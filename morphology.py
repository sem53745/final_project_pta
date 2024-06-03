# Program name: morphology.py
# Date: 03/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)

from spacy.tokens import Doc
from collections import Counter
from preprocessor import parse_prompt_data, get_and_parse_texts, Path
from typing import List, Tuple, Dict
from sklearn.metrics import classification_report, confusion_matrix

DEBUG = False

# Sem wrote the code, Jasper added Type-hints #

# Tokenization and Lemmatization
def tokenize_and_lemmatize(texts: List[Doc]) -> Tuple[List[str], List[str]]:
    """
    Tokenise and lemmatize the texts
    :param texts: List[Doc], the texts to tokenize and lemmatize
    """

    tokens: List[str] = []
    lemmas: List[str] = []
    for doc in texts:
        tokens.extend([token.text for token in doc])
        lemmas.extend([token.lemma_ for token in doc])
    return tokens, lemmas


def calculate_ratios(texts: List[Doc]) -> Dict[str, float]:
    '''
    This function calculates the ratios of the data
    it calculates the comma/point ratio, the token/lemma ratio and the token/types ratio
    param texts: List[Doc], the data to calculate the ratios of
    '''
    ratios: Dict[str, float] = {}

    # Calculation comma/point ratio for human data
    point_count: int = 0
    comma_count: int = 0
    token_count: int = 0
    lemma_count: int = 0
    types_count: int = 0
    for line in texts:
        point_count += line.text.count('.')
        comma_count += line.text.count(',')
        tokens: List[str] = [token.text for token in line]
        token_count += len(tokens)
        lemma_count += len(set([token.lemma_ for token in line]))
        types_count += len(set(tokens))

    ratios['comma-point'] = comma_count / point_count
    ratios['token-lemma'] = token_count / lemma_count
    ratios['token-types'] = token_count / types_count

    return ratios


def do_morpology_analysis(human_texts: List[Doc], machine_texts: List[Doc]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Analyse the human and machine dataand calculates the ratios of the data
    :param human_texts: List[Doc], the human data
    :param machine_texts: List[Doc], the machine data
    """
    human_tokens, human_lemmas = tokenize_and_lemmatize(human_texts)
    machine_tokens, machine_lemmas = tokenize_and_lemmatize(machine_texts)

    if DEBUG:
        # Average number of tokens and lemmas per line for both the machine and human data
        print(f'Average tokens per line for the human data: {round(len(human_tokens) / len(human_texts), 1)}')
        print(f'Average tokens per line for the machine data: {round(len(machine_tokens) / len(machine_texts), 1)}')
        print(f'Average lemmas per line for the human data: {round(len(human_lemmas) / len(human_texts), 1)}')
        print(f'Average lemmas per line for the machine data: {round(len(machine_lemmas) / len(machine_texts), 1)}\n')
        print(f'Average types per line for the human data: {round(len(set(human_tokens)) / len(human_texts), 1)}')
        print(f'Average types per line for the machine data: {round(len(set(machine_tokens)) / len(machine_texts), 1)}')
        print(f'Average lemma types per line for the human data: {round(len(set(human_lemmas)) / len(human_texts), 1)}')
        print(f'Average lemma types per line for the machine data: {round(len(set(machine_lemmas)) / len(machine_texts), 1)}\n')

        # Looking at the frequency of tokens
        human_tokens_count = Counter(human_tokens)
        machine_tokens_count = Counter(machine_tokens)
        print("Human:")
        for item in human_tokens_count.most_common(10):
            print(item)
        print("\nMachine:")
        for item in machine_tokens_count.most_common(10):
            print(item)


    human_ratios = calculate_ratios(human_texts)
    machine_ratios = calculate_ratios(machine_texts)

    if DEBUG:
        print(f'Human data ratios: ')
        for key, value in human_ratios.items():
            print(f'{key}: {value}')
        print(f'Machine data ratios:')
        for key, value in machine_ratios.items():
            print(f'{key}: {value}')

    return human_ratios, machine_ratios


def get_morphology_results(prompts: List[Dict[str, str | Doc]], ratios: Tuple[Dict[str, float], Dict[str, float]]) -> List[str]:
    """
    Predicts if each line in the input file is written by a human or a machine.
    :param prompts: List[Dict[str, str | Doc]], the data to predict
    :param ratios: Tuple[Dict[str, float], Dict[str, float]], the ratios of the human and machine data
    """
    # Parse the prompt data
    texts: List[Doc] = [entry['text'] for entry in prompts] # type: ignore

    human_ratios, machine_ratios = ratios
    comma_point = human_ratios['comma-point'] - machine_ratios['comma-point']
    token_lemma = human_ratios['token-lemma'] - machine_ratios['token-lemma']
    token_types = human_ratios['token-types'] - machine_ratios['token-types']

    # Initialize the results list
    results: List[str] = []

    # Analyze each line using the calculated average ratio as threshold
    for line in texts:
        result: List[str] = []

        comma_point_ratio = line.text.count(',') / (line.text.count('.') + 0.0001)
        tokens = [token.text for token in line]
        token_lemma_ratio = len(tokens) / len(set([token.lemma_ for token in line]))
        token_types_ratio = len(tokens) / len(set(tokens))

        if comma_point_ratio > human_ratios['comma-point'] - (comma_point / 2):
            result.append('Human')
        else:
            result.append('AI')

        if token_lemma_ratio > human_ratios['token-lemma'] - (token_lemma / 2):
            result.append('Human')
        else:
            result.append('AI')

        if token_types_ratio > human_ratios['token-types'] - (token_types / 2):
            result.append('Human')
        else:
            result.append('AI')

        results.append(max(set(result), key=result.count))

    return results

def main():
    # Load the test data
    human_data_path: Path = Path('human.jsonl')
    machine_data_path: Path = Path('group1.jsonl')

    # Process the data
    human_data, machine_data = get_and_parse_texts(human_data_path, machine_data_path)

    # Perform morphology analysis on the human and machine data
    separation = do_morpology_analysis(human_data, machine_data)

    # Load the prompt data
    prompt_path: Path = Path('prompts.jsonl')
    prompt_data = parse_prompt_data(prompt_path)

    # Predict if each line in the prompt data is written by a human or a machine
    results = get_morphology_results(prompt_data, separation)
    print(results)

    # Print the classification report with sklearn
    true_labels: List[str] = [prompt['by'] for prompt in prompt_data] # type: ignore
    print(classification_report(true_labels, results))

    # Print the confusion matrix with sklearn
    matrix = confusion_matrix(true_labels, results)
    print('confusion matrix:\n', matrix)


if __name__ == '__main__':
    main()

# sem #
