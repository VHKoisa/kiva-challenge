import os
import base64
import random
import argparse
import os
from utils_single import stitch_images_train,  stitch_images_test, read_image, stitch_final_images
import pandas as pd
import gc
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--concept', type=str, default="Counting", help='The concept to be tested')
parser.add_argument('--model', type=str, default="llava", help='model')
parser.add_argument('--api_key', type=str, default="API-KEY", help='gpt4_api_key')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images to process at once')
parser.add_argument('--max_trials', type=int, default=5, help='Maximum number of trials to run')
parser.add_argument('--max_regenerations', type=int, default=1, help='Maximum number of regenerations per trial')
parser.add_argument('--memory_efficient', action='store_true', help='Enable memory efficient mode')

args = parser.parse_args()

concept = args.concept
model_name = args.model
max_trials = args.max_trials
max_regenerations = args.max_regenerations

step_by_step_text = "step-by-step"

# Directory setup with memory-efficient paths
stimuli_directory = f"stimuli/KiVA/{concept}"
text_files_dir = f"stimuli/KiVA/trial_tracker/"
output_directory = f"output/single_image/output_{args.model}/{args.concept}"
stitched_images_directory = f"{output_directory}/{concept}_stitch"

os.makedirs(output_directory, exist_ok=True)
os.makedirs(stitched_images_directory, exist_ok=True)

# System prompts
system_prompt = ("You are an excellent visual puzzle solver! You will be given a visual puzzle that requires using visual analogical reasoning.")
system_prompt += f"You will think {step_by_step_text} and carefully examine the visual evidence before providing an answer."

initi_prompt = ("You are given a visual puzzle. The puzzle features a left-to-right transformation of an object on top and three left-to-right"
               "transformations of a different object on the bottom marked by (A) or (B) or (C)."
               "The transformations involve a change of either the size, orientation, number, or color of an object")

general_cross_rule_prompt = initi_prompt + ("Which one of the following rules {} best describes the left-to-right transformation on top of the"
                            "puzzle where the picture on the left transforms to the picture on the right? In your answer start with the correct rule number")
general_cross_rule_prompt += f"surrounded by parentheses, then provide a {step_by_step_text} reasoning for your choice."

general_within_rule_prompt = ("Which one of the following rules {} best describes the left-to-right transformation in the top of the puzzle where the picture"
                           "on the left transforms to the picture on the right?. In your answer start with the correct rule number surrounded by parentheses,")
general_within_rule_prompt += f"then provide a {step_by_step_text} reasoning for your choice."

extrapolation_prompt = ("Which one of three left-to-right object transformations (marked by either (A), (B) or (C) ) on the bottom of the puzzle is"
                      "the same as the left-to-right transformation on the top of the puzzle?"
                      "In your answer start with the correct letter surrounded by parentheses (or (D) if none of the options apply), ")
extrapolation_prompt += f"then provide a {step_by_step_text} reasoning for your choice."

concept_to_parameters = {
    "2DRotation": (["+90", "-90", 180]),
    "Counting": (["+1","+2","-1","-2"]),  
    "Colour": (["Red", "Green", "Blue"]),
    "Reflect": (["X", "Y"]), 
    "Resize": (["0.5XY", "2XY"])
}

def cleanup_memory():
    """Clean up GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def update_concept_result(param):
    concept_result = {
        "Variation": [],
        "Regeneration": [],
        "Train_input": [],
        "Train_output": [],
        "Test_input": [],
        "Test_output": [],
        "Test_wrong1": [],
        "Test_wrong2": [],
        "Full#1": [],
        "Full#2": [],
        "Full#3": [],
        "MCResponse#1": [],
        "MCResponse#2": [],
        "MCResponse#3": [],
    }
    return concept_result

def correct_cross_domain(concept):
    if concept == "2DRotation":
        return "Orientation of objects"
    elif concept == "Counting":
        return "Number of objects"
    elif concept == "Colour":
        return "Color of objects"
    elif concept == "Reflect":
        return "Orientation of objects"
    elif concept == "Resize":
        return "Size of objects"

def get_indexed_files(param):
    indexed_files = {}
    beginning = concept + str(param)
    for filename in os.listdir(stimuli_directory):
        if filename.startswith(beginning + "_"):
            index = int(filename.split('_')[1])
            if index not in indexed_files:
                indexed_files[index] = []
            indexed_files[index].append(filename)
    return indexed_files

def format_files_by_type(indexed_files, index, file_type):
    train_files = [filename for filename in indexed_files[index] if 'train' in filename]
    
    if file_type == 'train':
        input_filename = None
        output_filename = None
        for filename in train_files:
            if 'input' in filename:
                input_filename = filename
            elif 'output' in filename:
                output_filename = filename
        formatted_files = [input_filename, output_filename]
    elif file_type == 'test':
        test_files_input = [filename for filename in indexed_files[index] if 'test' in filename]
        formatted_files = sorted(test_files_input)
    
    return formatted_files

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def word_mc_options(selected_mc_options):
    worded_options = []
    for option in selected_mc_options:
        if concept == "2DRotation":
            if option == "-90" or option == "+90":
                option = 90
            worded_options += [f"Objects rotate by {option} degrees"]
        elif concept == "Counting":
            counting_type, option = option[0], option[1:]
            if counting_type == "+":
                worded_options += [f"Things go up by {option}"]
            elif counting_type == "-":
                worded_options += [f"Things go down by {option}"]
        elif concept == "Colour":
            worded_options += [f"Objects turn {option}"]
        elif concept == "Reflect":
            if option == "X":
                worded_options += [f"Objects flip upside down"]
            elif option == "Y":
                worded_options += [f"Objects flip sideways"]
            else:
                worded_options += [f"Objects rotate by {option} degrees"]
        elif concept == "Resize":
            if option == "0.5XY":
                worded_options += [f"Objects become smaller"]
            elif option == "2XY":
                worded_options += [f"Objects become bigger"]
    return worded_options

def eval_response(response, answers, all_choices):
    all_available_choices = {}
    for choice in all_choices:
        if choice in response:
            all_available_choices[choice] = response.index(choice)
    
    if len(all_available_choices) == 0:
        return False
    
    extracted_choice = min(all_available_choices, key=all_available_choices.get)
    return any(answer == extracted_choice for answer in answers)

if model_name == "llava":
    from models.llava_model import LLavaModel
    chat_model = LLavaModel(system_prompt, max_token=300)
    # Enable memory optimizations for the model
    if hasattr(chat_model.model_data["model"], "config"):
        chat_model.model_data["model"].config.use_cache = False
    if hasattr(chat_model.model_data["model"], "gradient_checkpointing_enable"):
        chat_model.model_data["model"].gradient_checkpointing_enable()
elif model_name == "gpt4":
    from models.gpt4_model import GPT4Model
    chat_model = GPT4Model(system_prompt, api_key=args.api_key, max_token=300)
else:
    raise ValueError("Model name not recognized.")

for param in concept_to_parameters[concept]:
    stimuli_set = get_indexed_files(param)
    output_file = output_directory + f"/{concept}{param}.csv"
    
    # Limit the number of trials
    query_repeats = min(len(stimuli_set), max_trials) if max_trials else len(stimuli_set)
    
    print("----------------------------------------------")
    print(f"Beginning Sub-Concept {concept} {param}")

    for query in list(range(query_repeats)):
        print("----------------------------------------------")
        print(f"Beginning Variation {query + 1} of {query_repeats}")

        # Limit regenerations
        for regeneration in range(max_regenerations):
            concept_result = update_concept_result(param)
            chat_model.init_history()
            
            cleanup_memory()  # Clean up memory before each trial
            
            # Rest of the processing code...
            # [Original code from lines 124-317]
            
            # Clean up memory after processing
            cleanup_memory()
            
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                df_to_add = pd.DataFrame(concept_result)
                df = pd.concat([df, df_to_add], ignore_index=True)
                df.to_csv(output_file, index=False)
                del df  # Free memory
                del df_to_add
            else:
                df = pd.DataFrame(concept_result)
                df.to_csv(output_file, index=False)
                del df
            
            cleanup_memory()
