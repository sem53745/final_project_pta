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

# sem #