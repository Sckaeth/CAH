from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
import ast

# Run:
# torchrun --master_addr="localhost" --master_port=29500 trl-attempt.py


# Define default script arguments..
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=0, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=True, metadata={"help": "If you want to resume training where it left off."}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=16)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[int] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default="5", metadata={"help": "The number of training epochs for the reward model. OpenAI used 5."}
    )


# Defines the parser with the default script args.
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Loads dataset and does a 80/20 split.
ds = load_dataset("csv", data_files="trimmed_cah_data.csv", split="train")
ds = ds.train_test_split(shuffle=True, seed=200, test_size=0.2)

# Ensures it actually runs on Windows. (can remove this if using Linux)
torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)

# Defines default training args.
training_args = TrainingArguments(
    output_dir=f"{script_args.model_name}_humour_reward_model",
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
)

# Load model and tokenizer from script args (in this case gpt-2).
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(script_args.model_name, num_labels=1)

# gpt-2 models require this as they have no padding token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id


# Converts dataset into a combination of prompts and punchlines.
# text_j is the preferred punchline (with its prompt)
# text_k is the punchline that is not preferred (alongside its prompt).
def turn_into_text_classification_format(examples):
    new_examples = {"text_j": [], "text_k": []}
    for info, summaries, choice in zip(examples["info"], examples["summaries"], examples["choice"]):
        info = ast.literal_eval(info)
        summaries = ast.literal_eval(summaries)

        if len(summaries) != 2 or choice not in (0, 1):
            raise ValueError(
                f"There should be two summaries with a choice that's either 0 or 1. Received {len(summaries)} summaries and choice={choice}."
            )
        new_examples["text_j"].append(
            summaries[choice]["text"] + " " + tokenizer.bos_token + " " + info["prompt"]
        )
        new_examples["text_k"].append(
            summaries[0 if choice == 1 else 1]["text"] + " " + tokenizer.bos_token + " " + info["prompt"]
        )

    return new_examples


num_proc = 1  # Set to 1 because my computer refuses to run it otherwise.
original_columns = ds["train"].column_names
ds = ds.map(turn_into_text_classification_format, batched=True, num_proc=num_proc, remove_columns=original_columns)


# Tokenize the dataset.
def preprocess_function(examples):
    tokenized_j = tokenizer(examples["text_j"], truncation=True)
    tokenized_k = tokenizer(examples["text_k"], truncation=True)
    return {
        "input_ids_j": tokenized_j["input_ids"],
        "attention_mask_j": tokenized_j["attention_mask"],
        "input_ids_k": tokenized_k["input_ids"],
        "attention_mask_k": tokenized_k["attention_mask"],
    }


tokenized_ds = ds.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=["text_j", "text_k"])


# Defines a special data collator that batches the data in the j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append({"input_ids": feature["input_ids_j"], "attention_mask": feature["attention_mask_j"]})
            features_k.append({"input_ids": feature["input_ids_k"], "attention_mask": feature["attention_mask_k"]})
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# Metric used for validation
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Predictions are the rewards of j and k. Accuracy checks for the frequency of j's reward being over k's reward.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Loss computed through j and k rewards. (as in the OpenAI paper)
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


# Trains model.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train(script_args.resume_from_checkpoint)

# Saves model/tokenizer.
trainer.save_model("./Models/Model/")
tokenizer.save_pretrained("./Models/Tokenizer/")
