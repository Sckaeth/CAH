import ast
import csv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import random


# Trims the CAH data and generates a new .csv with prompt/choice combinations.
# This grabs a random loss and a win, but ignores any rounds with more than one punchline selected.
# Generates rows from the model if "generate" is True.
# The choice associated with the generated prompt is a 50/50 pick between wins/losses.
def generate_data(generate=False):
    if generate:
        info_list = []
        prompt_list = []

    trimmed_data = open("extended_cah_data.csv", "w", encoding="utf8")
    writer = csv.writer(trimmed_data)

    # Write headers to file.
    writer.writerow(['info', 'summaries', 'choice'])

    with open("cah_data.csv", 'r', encoding="utf8") as file:
        file = list(csv.reader(file))[1:]

        game_id = 1

        grabbed_loss = None
        grabbed_win = None
        grabbed_prompt = None

        loss_index = -1
        loss_grab = random.randint(0, 6)

        for line in file:
            # Ignores anything with more than two prompts - for the sake of simplicity and time, we're ignoring them.
            # TODO: Eventually remove this constraint.
            if line[3].count("_____") > 1 or eval(line[2]):
                continue

            if int(line[0]) != game_id:
                new_row = [{'info': str(game_id), 'prompt': grabbed_prompt}, [{'text': grabbed_win}, {'text': grabbed_loss}], 0]
                writer.writerow(new_row)

                if generate:
                    info_list.append({'info': str(game_id), 'prompt': grabbed_prompt, 'grabbed_win': grabbed_win, 'grabbed_loss': grabbed_loss})
                    prompt_list.append(grabbed_prompt)

                game_id = int(line[0])
                grabbed_win = None
                grabbed_loss = None
                loss_index = -1
                loss_grab = random.randint(0, 6)

                print(f"Progress: {str((game_id/298955)*100)}%")

            if eval(line[6]) and not grabbed_win:
                grabbed_prompt = line[3]
                grabbed_win = line[5]
            elif not grabbed_loss:
                loss_index += 1
                if loss_index == loss_grab:
                    grabbed_loss = line[5]

        final_row = [{'info': str(game_id), 'prompt': line[3]}, [{'text': grabbed_win}, {'text': grabbed_loss}], 0]
        writer.writerow(final_row)

        game_id = int(line[0])
        print(f"Progress: {str((game_id / 298955) * 100)}%")

        if generate:
            tokenizer = AutoTokenizer.from_pretrained('Models/Tokenizer', local_files_only=True)
            tokenizer.pad_token = tokenizer.eos_token
            generator = pipeline("text-generation", model="Models/CAH-Model", tokenizer=tokenizer, max_new_tokens=30,
                                 min_new_tokens=4, device=0)

            start_n = 0
            end_n = 1000

            while end_n < len(prompt_list):
                outputs = generator(prompt_list[start_n:end_n])

                print(f"Begun saving {str(start_n)} to {str(end_n - 1)}.")
                for index, round in enumerate(info_list[start_n:end_n]):
                    output = outputs[index][0]['generated_text'].split(round['prompt'])[1]

                    grabbed_choice = random.sample([round['grabbed_win'], round['grabbed_loss']], 1)[0]
                    new_row = [{'info': round['info'] + "_g", 'prompt': round['prompt']}, [{'text': grabbed_choice}, {'text': output}], 0]
                    writer.writerow(new_row)

                print("Finished saving rows.")
                start_n = end_n
                end_n += 1000
                if end_n >= len(prompt_list):
                    end_n = prompt_list - 1


# Finds the maximum length of a punchline to get a rough estimate of how high the token limit should be for generations.
def find_max_len():
    with open("cah_data.csv", 'r', encoding="utf8") as file:
        file = list(csv.reader(file))[1:]
        max_len = 0

        for line in file:
            if len(line[5].split(" ")) > max_len:
                max_len = len(line[5].split(" "))

            print(f"Progress: {str((int(line[0]) / 298955) * 100)}%")

        print(max_len)


def get_reward(reward_model, tokenizer, output_r, prompt):
    output_d = {"text_j": output_r + " " + tokenizer.bos_token + " " + prompt}
    tokenized_j = tokenizer(output_d["text_j"], truncation=True, return_tensors="pt")

    logits = reward_model(**tokenized_j).logits.float()
    reward = (logits[:, 0]).tolist()[0]

    return reward


# Generates x model results with a reward.
def gen_model(num):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    generator = pipeline("text-classification", model="Models/FT-Model", tokenizer=tokenizer, device=0)
    reward_model = AutoModelForSequenceClassification.from_pretrained('Models/Model', num_labels=1,
                                                                      local_files_only=True)
    with open("split_cah_data_2.csv", 'r', encoding="utf8") as file:
        file = list(csv.reader(file))[1:]
        index_list = random.sample(range(len(file)), num)
        reach_count = 0

        for index in index_list:
            reach_count += 1
            prompt = ast.literal_eval(file[index][0])['prompt']
            choices = ast.literal_eval(file[index][1])

            print(f"Prompt: {prompt.replace('_____', choices[0]['text'])}")
            output = generator(prompt.replace('_____',choices[0]['text']))[0]['score']
            print(f"Generated score: {output}")
            # reward = get_reward(reward_model, tokenizer, choices[0]['text'], prompt)
            # print(f"Reward: {reward}\n")

            print(f"Prompt: {prompt.replace('_____', choices[1]['text'])}")
            output = generator(prompt.replace('_____',choices[1]['text']))[0]['score']
            print(f"Generated score: {output}")
            # reward = get_reward(reward_model, tokenizer, choices[1]['text'], prompt)
            # print(f"Reward: {reward}\n")

            print(f"Choice: {choices[0]['text']}")


# Generates a random 50% split of the row data and stores both splits separately as csv files.
def gen_split():
    split_data_1 = open("split_cah_data_1.csv", "w", newline='', encoding="utf8")
    writer_1 = csv.writer(split_data_1)

    split_data_2 = open("split_cah_data_2.csv", "w", newline='', encoding="utf8")
    writer_2 = csv.writer(split_data_2)

    # Write headers to files.
    writer_1.writerow(['info', 'summaries', 'choice'])
    writer_2.writerow(['info', 'summaries', 'choice'])

    with open("nogen_cah_data.csv", 'r', encoding="utf8") as file:
        file = list(csv.reader(file))[1:]

        index_list = random.sample(range(len(file)), int(len(file)*0.5))
        for index in range(len(file)):
            if len(file[index]) != 0:
                if index in index_list:
                    writer_1.writerow(file[index])
                else:
                    writer_2.writerow(file[index])
                print(str(index / len(file) * 100) + "% complete.")


# generate_data(True)
gen_model(100)
# gen_split()
