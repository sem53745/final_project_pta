# Program name: morphology.py
# Date: 03/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)

from spacy.tokens import Doc
from collections import Counter
from preprocessor import get_and_parse_texts, Path
from typing import List, Tuple

# Sem wrote the code, Jasper added Type-hints #

# Tokenization and Lemmatization
def tokenize_and_lemmatize(texts: List[Doc]) -> Tuple[List[str], List[str]]:
    tokens: List[str] = []
    lemmas: List[str] = []
    for doc in texts:
        tokens.extend([token.text for token in doc])
        lemmas.extend([token.lemma_ for token in doc])
    return tokens, lemmas


def morpology_results(human_texts: List[Doc], machine_texts: List[Doc]):
    human_tokens, human_lemmas = tokenize_and_lemmatize(human_texts)
    machine_tokens, machine_lemmas = tokenize_and_lemmatize(machine_texts)

    # Number of tokens and lemmas in the human and machine
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


def morpology_calculator(human_data: List[Doc], machine_data: List[Doc]):
    # Tries to tell if the line is human or machine when looking at the ratio kommas/points
    human_accuracy_counter: int = 0
    for line in human_data:
        point_count = line.text.count('.')
        comma_count = line.text.count(',')
        if point_count > 0:
            if comma_count / point_count > 0.93:
                human_accuracy_counter += 1
                # Optional: Print we think this line is human
    human_accuracy: float = human_accuracy_counter / len(human_data)
    print(f"\nHuman accuracy: {human_accuracy:.2f}")

    machine_accuracy_counter: int = 0
    for line in machine_data:
        point_count = line.text.count('.')
        komma_count = line.text.count(',')
        if point_count > 0:
            if komma_count / point_count < 0.93:
                machine_accuracy_counter += 1
                # Optional: Print we think this line is machine
    machine_accuracy: float = machine_accuracy_counter / len(machine_data)
    print(f"Machine accuracy: {machine_accuracy:.2f}")


def main():

    # load the data
    human_data: Path = Path('human_sample.jsonl')
    machine_data: Path = Path('group1_sample.jsonl')

    # process the data
    human_texts, machine_texts = get_and_parse_texts(human_data, machine_data)

    morpology_results(human_texts, machine_texts)

    morpology_calculator(human_texts, machine_texts)


if __name__ == '__main__':
    main()

# sem #
