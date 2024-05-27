# Program name: morphology.py
# Date: 03/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)

import json
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

# sem #

# Read .jsonl files
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

human_data = load_jsonl('human.jsonl')
machine_data = load_jsonl('group1.jsonl')

# Extract text data
human_texts = [entry['text'] for entry in human_data if 'text' in entry and entry['text'].strip()]
machine_texts = [entry['text'] for entry in machine_data if 'text' in entry and entry['text'].strip()]

# Check number of lines with actual text
print(f'Number of lines with text in human.jsonl: {len(human_texts)}')
print(f'Number of lines with text in group1.jsonl: {len(machine_texts)}\n')

# Tokenization and Lemmatization
def tokenize_and_lemmatize(texts):
    tokens = []
    lemmas = []
    for doc in nlp.pipe(texts):
        tokens.extend([token.text for token in doc])
        lemmas.extend([token.lemma_ for token in doc])
    return tokens, lemmas

# Get tokens and lemmas for human and machine texts
human_tokens, human_lemmas = tokenize_and_lemmatize(human_texts)
machine_tokens, machine_lemmas = tokenize_and_lemmatize(machine_texts)

# Average number of tokens and lemmas per line for both the machine and human data
print(f'Average tokens per line for the human data: {round(len(human_tokens) / len(human_texts), 1)}')
print(f'Average tokens per line for the machine data: {round(len(machine_tokens) / len(machine_texts), 1)}')
print(f'Average lemmas per line for the human data: {round(len(human_lemmas) / len(human_texts), 1)}')
print(f'Average lemmas per line for the machine data: {round(len(machine_lemmas) / len(machine_texts), 1)}\n')

# Looking at the frequency of certain tokens
human_tokens_count = Counter(human_tokens)
machine_tokens_count = Counter(machine_tokens)
print("Human:")
for item in human_tokens_count.most_common(10):
    print(item)
print("\nMachine:")
for item in machine_tokens_count.most_common(10):
    print(item)

def morpology_calculator(human_data, machine_data):
    # Tries to tell if the line is human or machine when looking at the ratio kommas/points
    human_accuracy_counter = 0
    for line in human_data:
        point_count = line.count('.')
        komma_count = line.count(',')
        if point_count > 0:
            if komma_count / point_count > 0.93:
                human_accuracy_counter += 1
                # Optional: Print we think this line is human
    human_accuracy = human_accuracy_counter / len(human_data)
    print(f"\nHuman accuracy: {human_accuracy:.2f}")

    machine_accuracy_counter = 0
    for line in machine_data:
        point_count = line.count('.')
        komma_count = line.count(',')
        if point_count > 0:
            if komma_count / point_count < 0.93:
                machine_accuracy_counter += 1
                # Optional: Print we think this line is machine
    machine_accuracy = machine_accuracy_counter / len(machine_data)
    print(f"Machine accuracy: {machine_accuracy:.2f}")

morpology_calculator(human_texts, machine_texts)

# sem #
