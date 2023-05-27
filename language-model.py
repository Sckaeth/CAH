import ast
import torch
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Config stuff
tqdm.pandas()
config = PPOConfig(
    model_name="gpt2",
    learning_rate=(1.47e-5) * 2,
    log_with="wandb",
    batch_size=8,
    forward_batch_size=1,
)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# Defaults to 0.
set_seed(config.seed)


# Sets up model, optimizer and tokenizer.
model = AutoModelForCausalLM.from_pretrained(config.model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
ref_model = create_reference_model(model)

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

tokenizer = AutoTokenizer.from_pretrained('Models/Tokenizer', local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset in.
dataset = load_dataset("csv", data_files="trimmed_cah_data.csv", split="train")


# Tokenize dataset for the training. Only the prompt row is utilised.
def tokenize(sample):
    info = ast.literal_eval(sample['info'])
    sample['input_ids'] = tokenizer.encode(info['prompt'], truncation=True)
    sample['query'] = tokenizer.decode(sample["input_ids"])
    return sample


# Apply to dataset, split into 80/20 split.
dataset = dataset.map(tokenize, batched=False)
dataset.set_format(type="torch")
dataset = dataset.train_test_split(shuffle=True, test_size=0.2)['train']
print(dataset)

# Set up trainer and reward model, via the local model.
ppo_trainer = PPOTrainer(
    config, model, ref_model=ref_model, tokenizer=tokenizer, dataset=dataset, data_collator=collator
)

reward_model = AutoModelForSequenceClassification.from_pretrained('Models/Model', num_labels=1, local_files_only=True).to(
    ppo_trainer.accelerator.device)

# Generation args.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_min_length = 4
output_max_length = 30
output_length_sampler = LengthSampler(output_min_length, output_max_length)

model_save_path = "/Models/CAH-Model"

# Training process
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Generates a response from the model.
    # An input queries list is generated to store the prompts for every generated punchline.
    response_tensors = []
    input_queries = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])

        # The punchlines are decoded/detokenized into text.
        input_queries.append(TreebankWordDetokenizer().detokenize([tokenizer.decode(x) for x in query]))

    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # A reward is computed from the model by fitting the same format the reward model trainer utilises.
    texts = batch["response"]
    new_texts = []
    for index, response in enumerate(texts):
        new_texts.append(response + " " + tokenizer.bos_token + " " + input_queries[index])

    reward_inputs = tokenizer(new_texts, padding=True, truncation=True, return_tensors="pt").to(
        ppo_trainer.accelerator.device
    )
    logits = reward_model(**reward_inputs).logits.float()
    reward_labels = (logits[:, 0]).tolist()

    rewards = [torch.tensor(output) for output in reward_labels]

    # A PPO step is completed.
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 100 epochs (?)
    if epoch % 100 == 0:
        if ppo_trainer.accelerator.is_main_process:
            print("Saving after 100 epochs.")
            ppo_trainer.save_pretrained(model_save_path)