import matplotlib.pyplot as plt
import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSeq2SeqLM

# get index of of all the answers that are yes
# load the tokenizer & model for T5 & GPT2
tokenizer_GPTXL = AutoTokenizer.from_pretrained("gpt2")
model_GPTXL = AutoModelForCausalLM.from_pretrained("gpt2")


def generate(prompt, max_length=200):
    model = model_GPTXL
    tokenizer = tokenizer_GPTXL

    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # get output with standard parameters
    sample_output = model.generate(
        input_ids,        # context to continue
        do_sample=False,   # use sampling (not beam search (see below))
        # return maximally 50 words (including the input given)
        max_length=max_length,
        # top_k=0,          # just sample one word         # consider all options
        # temperature=0.7   # soft-max temperature
    )
    return (tokenizer.decode(sample_output[0], skip_special_tokens=True))


# Define the file path
file_path = "coin_questions.txt"

# Initialize an empty list to store questions
coin_questions_list = []

# Read the questions from the file
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        # Remove the number at the beginning and append the question to the list
        question = line.strip().split(' ', 1)[1]
        coin_questions_list.append(question)

# Display the list of questions
for i, question in enumerate(coin_questions_list, start=1):
    print(f"{i}. {question}")

# Open the file for reading
with open("coin_cot_3.txt", "r") as file:
    # Read the entire content of the file into a single string
    cot_coin = file.read()


# Irrelevant Prompts

# Open the file for reading
with open("coin_cot_irrelevant_3.txt", "r") as file:
    # Read the entire content of the file into a single string
    cot_coin_irr = file.read()
# False reasoning CoT

# Open the file for reading
with open("coin_cot_wrong_3.txt", "r") as file:
    # Read the entire content of the file into a single string
    cot_coin_false = file.read()
# "Lets think step by step"
# Open the file for reading
with open("step_by_step.txt", "r") as file:
    # Read the entire content of the file into a single string
    step_by_step = file.read()
# Standard few(3)-shot prompt
# Open the file for reading
with open("coin_std_few_shot_3.txt", "r") as file:
    # Read the entire content of the file into a single string
    few_shot = file.read()

# Define prompt templates and settings
prompt_settings = {
    "cot_irr": {"template": cot_coin_irr, "max_length": 205},
    "cot_think": {"template": step_by_step, "max_length": 47},
    "cot": {"template": cot_coin, "max_length": 335},
    "cot_false": {"template": cot_coin_false, "max_length": 320},
    "3shot": {"template": few_shot, "max_length": 154}
}

# Initialize a dictionary to hold the outputs
outputs = {key: [] for key in prompt_settings.keys()}

# Generate outputs for each question and each prompt setting
for coin_question in coin_questions_list:
    for key, settings in prompt_settings.items():
        prompt = settings["template"] + " Q: " + coin_question
        output = generate(prompt, max_length=settings["max_length"])
        outputs[key].append(output)

# Now, outputs will contain lists of generated outputs for each template

# outputs["cot_irr"]


def extract_yes_no(strings):
    """
    Extracts the last occurrence of 'yes' or 'no' in each string of the input list.
    The function is case-insensitive.

    :param strings: List of strings to process.
    :return: List of 'yes' or 'no' corresponding to each string.
    """
    results = []
    for string in strings:
        # Find the last occurrence of 'yes' or 'no', case-insensitive
        last_yes = string.lower().rfind('yes')
        last_no = string.lower().rfind('no')

        if last_yes == -1 and last_no == -1:
            # Neither 'yes' nor 'no' found in the string
            results.append(None)
        elif last_yes > last_no:
            # 'yes' is the last occurrence
            results.append('yes')
        else:
            # 'no' is the last occurrence or both are not found
            results.append('no')

    return results


final_outputs = {}

for key in outputs:
    final_outputs[key] = extract_yes_no(outputs[key])

# Initialize an empty list to store answers
coin_answer_list = []

# Define the file path
file_path = "coin_answers.txt"

# Read the questions from the file
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        # Remove the number at the beginning and append the question to the list
        answer = line.strip().split(' ', 1)[1]
        coin_answer_list.append(answer)


def compare_lists_detail(list1):
    """
    Compares each element of two lists and outputs a list of 'correct' or 'incorrect'.

    :param list1: First list of strings.
    :param list2: Second list of strings, of the same length as list1.
    :return: List of 'correct' or 'incorrect' based on element-wise comparison.
    """
    if len(list1) != len(coin_answer_list):
        raise ValueError("Both lists must be of the same length.")

    results = []
    for item1, item2 in zip(list1, coin_answer_list):
        if item1 == item2 and item1 == "yes":
            results.append("correct yes")
        if item1 == item2 and item1 == "no":
            results.append("correct no")
        if item1 != item2 and item1 == "yes":
            results.append("incorrect yes")
        else:
            results.append("incorrect no")

    return results


def compare_lists(list1):
    """
    Compares each element of two lists and outputs a list of 'correct' or 'incorrect'.

    :param list1: First list of strings.
    :param list2: Second list of strings, of the same length as list1.
    :return: List of 'correct' or 'incorrect' based on element-wise comparison.
    """
    if len(list1) != len(coin_answer_list):
        raise ValueError("Both lists must be of the same length.")

    results = []
    for item1, item2 in zip(list1, coin_answer_list):
        if item1 == item2:
            results.append("correct")
        else:
            results.append("incorrect")

    return results


outputs_eval = {}

for key in final_outputs:
    outputs_eval[key] = compare_lists(final_outputs[key])

outputs_eval_detail = {}

for key in final_outputs:
    outputs_eval_detail[key] = compare_lists_detail(final_outputs[key])


# outputs_eval["cot_irr"]


def count_responses_and_plot(data):
    """
    Counts the number of 'correct' and 'incorrect' responses in each list of the dictionary
    and plots a stacked bar graph with counts above each bar.

    :param data: Dictionary with keys as list names and values as lists of 'correct'/'incorrect' strings.
    """
    # Counting correct and incorrect in each list
    counts = {key: {"correct": lst.count("correct"), "incorrect": lst.count(
        "incorrect")} for key, lst in data.items()}

    # Data for plotting
    labels = counts.keys()
    correct_counts = [count["correct"] for count in counts.values()]
    incorrect_counts = [count["incorrect"] for count in counts.values()]

    # Creating the bar plot
    fig, ax = plt.subplots()
    bars1 = ax.bar(labels, correct_counts, label='Correct')
    bars2 = ax.bar(labels, incorrect_counts,
                   label='Incorrect', bottom=correct_counts)

    # Adding labels and title
    ax.set_xlabel('Lists')
    ax.set_ylabel('Counts')
    ax.set_title('Count of Correct and Incorrect Responses in Each List')
    ax.legend()

    # Add counts above each bar
    for bar1, bar2 in zip(bars1, bars2):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        total_height = height1 + height2
        ax.annotate(f'{height1}', xy=(bar1.get_x() + bar1.get_width() / 2, height1), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')
        ax.annotate(f'{height2}', xy=(bar2.get_x() + bar2.get_width() / 2, total_height), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')

    # Display the plot
    plt.xticks(rotation=45)
    plt.show()


# Generate and display the plot
count_responses_and_plot(outputs_eval)


#######################
# EASIER COIN DATASET
#######################

# Performance easier coin questions
# Define the file path
file_path = "simplified_coin_questions.txt"

# Initialize an empty list to store questions
simple_coin_questions_list = []

# Read the questions from the file
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        # Remove the number at the beginning and append the question to the list
        question = line.strip().split(' ', 1)[1]
        simple_coin_questions_list.append(question)

# Display the list of questions
for i, question in enumerate(simple_coin_questions_list, start=1):
    print(f"{i}. {question}")

# Open the file for reading
with open("coin_cot_3_easy.txt", "r") as file:
    # Read the entire content of the file into a single string
    cot_coin = file.read()

# Irrelevant Prompts
# Open the file for reading
with open("coin_cot_irrelevant_3_easy.txt", "r") as file:
    # Read the entire content of the file into a single string
    cot_coin_irr = file.read()
# False reasoning CoT

# Open the file for reading
with open("coin_cot_wrong_3_easy.txt", "r") as file:
    # Read the entire content of the file into a single string
    cot_coin_false = file.read()
# "Lets think step by step"
# Open the file for reading
with open("step_by_step.txt", "r") as file:
    # Read the entire content of the file into a single string
    step_by_step = file.read()
# Standard few(3)-shot prompt
# Open the file for reading
with open("coin_std_few_shot_3_easy.txt", "r") as file:
    # Read the entire content of the file into a single string
    few_shot = file.read()

# Initialize an empty list to store answers
simple_coin_answer_list = []

# Define the file path
file_path = "coin_questions_easier_answers.txt"

# Read the questions from the file
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        # Remove the number at the beginning and append the question to the list
        answer = line.strip().split(' ', 1)[1]
        simple_coin_answer_list.append(answer)

# Define prompt templates and settings
prompt_settings = {
    "cot_irr": {"template": cot_coin_irr, "max_length": 185},
    "cot_think": {"template": step_by_step, "max_length": 47},
    "cot": {"template": cot_coin, "max_length": 280},
    "cot_false": {"template": cot_coin_false, "max_length": 310},
    "3shot": {"template": few_shot, "max_length": 130}
}

# Initialize a dictionary to hold the outputs
simple_outputs = {key: [] for key in prompt_settings.keys()}

# Generate outputs for each question and each prompt setting
for coin_question in simple_coin_questions_list:
    for key, settings in prompt_settings.items():
        prompt = settings["template"] + " Q: " + coin_question
        output = generate(prompt, max_length=settings["max_length"])
        simple_outputs[key].append(output)

# Now, outputs will contain lists of generated outputs for each template
simple_final_outputs = {}

for key in simple_outputs:
    simple_final_outputs[key] = extract_yes_no(simple_outputs[key])

simple_outputs_eval = {}

for key in simple_final_outputs:
    simple_outputs_eval[key] = compare_lists(simple_final_outputs[key])

simple_outputs_eval_detail = {}

for key in simple_final_outputs:
    simple_outputs_eval_detail[key] = compare_lists_detail(
        simple_final_outputs[key])


# Generate and display the plot
count_responses_and_plot(simple_outputs_eval)

# for key in outputs:
#     print("next:")
#     print(outputs[key][0])
# for i in range(0, 50):
#     print(simple_final_outputs["cot_think"][i])
# simple_outputs_eval
