from accelerate import Accelerator
import numpy as np
import torch.utils, gc
from datasets import Dataset
from torch import softmax
from torch.nn.functional import pad, log_softmax
from trl.core import stats_to_np, stack_dicts, WANDB_PADDING, masked_whiten, clip_by_value, masked_mean

from RLHF.utils import *


# Trainer Class
# File modified and derived from the TRL library.
# ~~~
class RLTrainer:
    # Initialises the RL trainer.
    def __init__(self, proj_name, model, ref_model, tokenizer, dataset, optimizer, data_collator):
        # Initialises configuration parameters. TODO: fix up naming, setup, etc.
        self.steps = 20000
        self.init_kl_coef = 0.2
        self.target = 6
        self.horizon = 10000
        self.gamma = 1
        self.lam = 0.95
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.vf_coef = 0.1
        self.batch_size = 1
        self.mini_batch_size = 1
        self.gradient_accumulation_steps = 6
        self.ppo_epochs = 64
        self.seed = 0
        self.total_ppo_epochs = int(np.ceil(self.steps/self.batch_size))
        self.current_device = torch.device("cuda:0")

        # Initialises model parameters.
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer

        self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler

        # Defines dataset and dataloader.
        self.dataset = dataset
        self.data_collator = data_collator
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            drop_last=True,
        )

        # Accelerator set to default parameters. TODO: may need to add config for tracking..
        self.accelerator = Accelerator(gradient_accumulation_steps=self.gradient_accumulation_steps)
        # self.accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=self.gradient_accumulation_steps)
        # self.accelerator.init_trackers(
        #     proj_name,
        # )

        # Prepares the accelerator.
        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.data_collator, self.dataloader
        )
        self.ref_model = self.accelerator.prepare(self.ref_model)

        # Sets up the KL controller and initialises the step count.
        self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.target, self.horizon)
        self.current_step = 0

    # Generates a numerical response for an input query and attention mask.
    def generate(self, query, mask):
        output = self.accelerator.unwrap_model(self.model)(input_ids=query.unsqueeze(0), attention_mask=mask.unsqueeze(0))
        return output

    # Performs one PPO step.
    def step(self, queries, masks, scores):
        # Clears up memory.
        gc.collect()
        torch.cuda.empty_cache()

        queries = [tensor.to(self.current_device) for tensor in queries]
        masks = [tensor.to(self.current_device) for tensor in masks]
        scores = [tensor.to(self.current_device) for tensor in scores]

        with torch.no_grad():
            all_logprobs, values = self.forward_pass(self.model, queries, masks)
            ref_logprobs, _ = self.forward_pass(self.ref_model, queries, masks)

        rewards, non_score_reward = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)

        mini_batch_dict = {
                "queries": queries,
                "logprobs": all_logprobs,
                "values": values,
                "rewards": rewards,
                "masks": masks,
        }

        def collator(data):
            return_dict = dict()
            for key in data[0]:
                return_dict[key] = [d[key] for d in data]
            return return_dict

        # mini_batch_dict.update(model_inputs)
        mini_batch_data = Dataset.from_dict(mini_batch_dict)
        mini_batch_data.set_format("torch")
        mini_batch_dataloader = torch.utils.data.DataLoader(
            mini_batch_data,
            batch_size=self.mini_batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        all_stats = []
        for _ in range(self.ppo_epochs):
            for i, batch in enumerate(mini_batch_dataloader):
                with self.accelerator.accumulate(self.model):
                    logprobs, logits = self.forward_pass(
                        self.model, batch["queries"], batch["masks"]
                    )

                    if (i % self.gradient_accumulation_steps) == 0:
                        self.optimizer.zero_grad()

                    train_stats = self.train_minibatch(
                        batch["logprobs"],
                        batch["values"],
                        batch["rewards"],
                        logprobs,
                        logits,
                        batch["masks"],
                    )

                    all_stats.append(train_stats)

        # Clears up memory.
        gc.collect()
        torch.cuda.empty_cache()

        # TODO: stats.
        stats = torch.tensor(all_stats).to(self.current_device)
        print("Mean Loss: ", float(stats.mean()))

        # TODO: Could use a function.
        gen_len = len(masks[-1][-1])

        new_logprobs = []
        for x, inner_list in enumerate(all_logprobs):
            new_logprobs.append([[] for _ in range(10)])
            for i, tensor in enumerate(inner_list):
                for j, element in enumerate(tensor):
                    new_logprobs[x][j].append(element)
        all_logprobs = [torch.stack(joke) for round_vals in new_logprobs for joke in round_vals]
        all_logprobs = torch.stack([pad(tensor, pad=(0, gen_len - len(tensor)), mode='constant', value=0) for tensor in all_logprobs]).to(self.current_device)

        new_logprobs_2 = []
        for x, inner_list in enumerate(ref_logprobs):
            new_logprobs_2.append([[] for _ in range(10)])
            for i, tensor in enumerate(inner_list):
                for j, element in enumerate(tensor):
                    new_logprobs_2[x][j].append(element)
        ref_logprobs = [torch.stack(joke) for round_vals in new_logprobs_2 for joke in round_vals]
        ref_logprobs = torch.stack([pad(tensor, pad=(0, gen_len - len(tensor)), mode='constant', value=0) for tensor in ref_logprobs]).to(self.current_device)
        masks = torch.stack([joke for round_vals in masks for joke in round_vals]).to(self.current_device)

        kl_list = ((all_logprobs - ref_logprobs) * masks).sum(axis=-1)
        mean_kl = kl_list.mean().cpu()

        self.kl_ctl.update(mean_kl, self.batch_size * self.accelerator.num_processes)

        return stats

    def train_minibatch(self, old_logprobs, values, rewards, logprobs, logits, mask):
        # Clears up memory.
        gc.collect()
        torch.cuda.empty_cache()

        loss_p, loss_v, train_stats = self.loss(old_logprobs, values, rewards, logprobs, logits, mask)
        loss = loss_p + loss_v
        
        self.accelerator.backward(loss)
        self.optimizer.step()

        return train_stats

    def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
        # Clears up memory.
        gc.collect()
        torch.cuda.empty_cache()

        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            # convert to tensors
            logprob, ref_logprob = torch.stack(logprob), torch.stack(ref_logprob)

            # compute KL penalty (from difference in logprobs)
            kl = logprob - ref_logprob
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward.to(self.current_device))
            reward = non_score_reward.clone().to(self.current_device)
            # [1:] as we ignore the BOS token, -1 as we ignore the EOS token
            last_non_masked_index = torch.tensor([(mask[i][1:].nonzero().max() - 1) for i in range(10)]).to(self.current_device)

            # reward is preference model score + KL penalty
            for i in range(10):
                reward[last_non_masked_index[i]][i] += score[i]
                
            rewards.append(reward)
        
        return rewards, non_score_rewards

    def loss(self, old_logprobs, values, rewards, logprobs, vpreds, mask):
        # Clears up memory.
        gc.collect()
        torch.cuda.empty_cache()

        lastgaelam = 0
        advantages_reversed = []

        values = [joke for round_vals in values for joke in round_vals]

        # TODO: Use function
        new_rewards = []
        for x, inner_list in enumerate(rewards):
            new_rewards.append([[] for _ in range(10)])
            for i, tensor in enumerate(inner_list):
                for j, element in enumerate(tensor):
                    new_rewards[x][j].append(element)
        rewards = [torch.stack(joke) for round_vals in new_rewards for joke in round_vals]

        gen_len = len(mask[-1][-1])

        values = torch.stack([pad(tensor, pad=(0, gen_len - len(tensor)), mode='constant', value=0) for tensor in values]).to(self.current_device)
        rewards = torch.stack([pad(tensor, pad=(0, gen_len - len(tensor)), mode='constant', value=0) for tensor in rewards]).to(self.current_device)
        mask = torch.stack([joke for round_vals in mask for joke in round_vals]).to(self.current_device)

        # remove last two 1s in mask (ignoring bos/eos)
        # TODO: depends on model architecture.
        for tensor in mask:
            indices = tensor.nonzero().flatten()[-2:]
            tensor[indices] = 0

        values = (values * mask)
        rewards = (rewards * mask)
        
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()

        vpreds = [joke for round_vals in vpreds for joke in round_vals]
        vpreds = torch.stack([pad(tensor, pad=(0, gen_len - len(tensor)), mode='constant', value=0) for tensor in vpreds])
        
        vpredclipped = clip_by_value(
            vpreds, values - self.cliprange_value, values + self.cliprange_value
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)

        # TODO: use function.
        new_logprobs = []
        for x, inner_list in enumerate(logprobs):
            new_logprobs.append([[] for _ in range(10)])
            for i, tensor in enumerate(inner_list):
                for j, element in enumerate(tensor):
                    new_logprobs[x][j].append(element)
        logprobs = [torch.stack(joke) for round_vals in new_logprobs for joke in round_vals]
        logprobs = torch.stack([pad(tensor, pad=(0, gen_len - len(tensor)), mode='constant', value=0) for tensor in logprobs]).to(self.current_device)

        new_logprobs_2 = []
        for x, inner_list in enumerate(old_logprobs):
            new_logprobs_2.append([[] for _ in range(10)])
            for i, tensor in enumerate(inner_list):
                for j, element in enumerate(tensor):
                    new_logprobs_2[x][j].append(element)
        old_logprobs = [torch.stack(joke) for round_vals in new_logprobs_2 for joke in round_vals]
        old_logprobs = torch.stack([pad(tensor, pad=(0, gen_len - len(tensor)), mode='constant', value=0) for tensor in old_logprobs]).to(self.current_device)

        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)

        loss = pg_loss + self.vf_coef * vf_loss

        stats = loss.detach()
        return pg_loss, self.vf_coef * vf_loss, stats

    def forward_pass(self, model, queries, masks):
        # Clears up memory.
        gc.collect()
        torch.cuda.empty_cache()

        bs = len(queries)
        fbs = self.mini_batch_size

        all_logprobs = []
        all_logits = []

        for i in range(int(bs / fbs)):
            # TODO: selects values within range of mini-bat
            batch_queries = queries[i * fbs: (i + 1) * fbs]
            batch_masks = masks[i * fbs: (i + 1) * fbs]
            logits = []

            # iterate over every batch/round in the mini-batch, additionally finding a logit for each word/token, etc.
            for j in range(fbs):
                round_logits = []
                # the current values will have a list of 10 jokes, we iterate through each joke.
                for x in range(10):
                    joke_logits = []
                    joke = batch_queries[j][x]
                    curr_joke = joke[0].reshape(1)

                    # all tokens in range of attention mask
                    for token in range(1, batch_masks[j][x].nonzero().max()):
                        curr_joke = torch.cat((curr_joke, joke[token].reshape(1)), 0)

                        # perform inference on the current built joke.
                        # TODO: this also differs when you change out of BERT (bos/eos tokens)
                        logit = model(input_ids=(torch.cat((curr_joke, joke[-1].reshape(1)))).unsqueeze(0).to(
                            self.current_device)).logits[0][0]

                        joke_logits.append(logit)
                    round_logits.append(torch.stack(joke_logits))
                logits.append(round_logits)

            # Softmax the logits to get probabilities.
            logprobs = []
            for round_logits in logits:
                round_logprobs = []
                max_joke_len = max(len(joke) for joke in round_logits)

                for index in range(max_joke_len):
                    logit_values = []
                    for joke in round_logits:
                        # use final value for smaller tensor/jokes
                        if index >= len(joke):
                            logit_values.append(joke[-1])
                        else:
                            logit_values.append(joke[index])

                    # softmax applied to values
                    softmax_values = log_softmax(torch.stack(logit_values), dim=0)
                    round_logprobs.append(softmax_values)

                logprobs.append(round_logprobs)

            all_logits.append(logits)
            all_logprobs.append(logprobs)

        # TODO: can be optimised.
        all_logits_c = []
        for item in all_logits:
            all_logits_c.extend(item)

        all_logprobs_c = []
        for item in all_logprobs:
            all_logprobs_c.extend(item)

        torch.cuda.empty_cache()

        return (
            all_logprobs_c,
            all_logits_c
        )

    # Saves the RL-trained model when called.
    def save_pretrained(self, save_directory: str) -> None:
        self.accelerator.unwrap_model(self.model).save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
