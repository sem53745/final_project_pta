# Program name: main.py
# Date: 03/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)

import json
import spacy

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
human_texts = [entry['text'] for entry in human_data]
machine_texts = [entry['text'] for entry in machine_data]

# Tokenization and Lemmatization
def tokenize_and_lemmatize(texts):
    tokens = []
    lemmas = []
    for doc in nlp.pipe(texts):
        tokens.append([token.text for token in doc])
        lemmas.append([token.lemma_ for token in doc])
    return tokens, lemmas

human_tokens, human_lemmas = tokenize_and_lemmatize(human_texts)
machine_tokens, machine_lemmas = tokenize_and_lemmatize(machine_texts)
# sem #
