import ast
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, AutoModelForSequenceClassification
import numpy as np
import random

dataset = load_dataset("csv", data_files="drive/MyDrive/CAH_Data/split_cah_data_1.csv", split="train")
dataset = dataset.train_test_split(test_size=0.2)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


def preprocess_function(examples):
    choice = [random.randint(0, 1) for x in range(len(examples['info']))]

    prompts = [ast.literal_eval(doc)['prompt'] for doc in examples['info']]
    punchlines = [ast.literal_eval(punchline) for punchline in examples['summaries']]
    combined = [prompts[index].replace("_____", input_text[choice[index]]['text']) for index, input_text in
                enumerate(punchlines)]

    with tokenizer.as_target_tokenizer():
        text = tokenizer(combined, max_length=512, padding="max_length", truncation=True)

    new_examples = text
    new_examples['label'] = choice
    return new_examples


original_columns = dataset["train"].column_names
tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=1, remove_columns=original_columns)
print(tokenized_dataset)

metric = evaluate.load("mse")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, squared=False)


training_args = TrainingArguments(output_dir="drive/MyDrive/FT-Model", evaluation_strategy="epoch", save_total_limit=2)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics
)

# trainer.train(resume_from_checkpoint=True)
trainer.train()
trainer.save_model()