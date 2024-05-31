# Program name: morphology.py
# Date: 03/06
# Contributors: Jasper Kleine (s5152372), Sem Bartels (s5374588)

from spacy.tokens import Doc
from collections import Counter
from preprocessor import parse_promt_data, get_and_parse_texts, Path
from typing import List, Tuple
from spacy.language import Language
from spacy import load as load_spacy_model
nlp = load_spacy_model("en_core_web_sm")

# Sem wrote the code, Jasper added Type-hints #

# Tokenization and Lemmatization
def tokenize_and_lemmatize(texts: List[Doc]) -> Tuple[List[str], List[str]]:
    """This function tokenizes the data"""
    tokens: List[str] = []
    lemmas: List[str] = []
    for doc in texts:
        tokens.extend([token.text for token in doc])
        lemmas.extend([token.lemma_ for token in doc])
    return tokens, lemmas


def morpology_analysis(human_texts: List[Doc], machine_texts: List[Doc]):
    """This function analysis the human and machine test data"""
    human_tokens, human_lemmas = tokenize_and_lemmatize(human_texts)
    machine_tokens, machine_lemmas = tokenize_and_lemmatize(machine_texts)

    # Average number of tokens and lemmas per line for both the machine and human data
    print(f'Average tokens per line for the human data: {round(len(human_tokens) / len(human_texts), 1)}')
    print(f'Average tokens per line for the machine data: {round(len(machine_tokens) / len(machine_texts), 1)}')
    print(f'Average lemmas per line for the human data: {round(len(human_lemmas) / len(human_texts), 1)}')
    print(f'Average lemmas per line for the machine data: {round(len(machine_lemmas) / len(machine_texts), 1)}\n')

    # Looking at the frequency of tokens
    human_tokens_count = Counter(human_tokens)
    machine_tokens_count = Counter(machine_tokens)
    print("Human:")
    for item in human_tokens_count.most_common(10):
        print(item)
    print("\nMachine:")
    for item in machine_tokens_count.most_common(10):
        print(item)

    # Calculation comma/point ratio for human data
    total_point_count = 0
    total_comma_count = 0
    human_accuracy_counter = 0
    for line in human_texts:
        point_count = line.text.count('.')
        comma_count = line.text.count(',')
        total_point_count += point_count
        total_comma_count += comma_count
        if point_count > 0:
            if comma_count / point_count > 0.815:
                human_accuracy_counter += 1
    human_accuracy = human_accuracy_counter / len(human_texts)
    print(f"\nComma/point ratio human data {(total_comma_count/total_point_count):.2f}")
    print(f"Human accuracy: {human_accuracy:.2f}")

    # Calculation comma/point ratio for machine data
    total_point_count = 0
    total_comma_count = 0
    machine_accuracy_counter = 0
    for line in machine_texts:
        point_count = line.text.count('.')
        comma_count = line.text.count(',')
        total_point_count += point_count
        total_comma_count += comma_count
        if point_count > 0:
            if comma_count / point_count < 0.815:
                machine_accuracy_counter += 1
    machine_accuracy = machine_accuracy_counter / len(machine_texts)
    print(f"\nComma/point ratio machine data {(total_comma_count/total_point_count):.2f}")
    print(f"Machine accuracy: {machine_accuracy:.2f}")

def morphology_results(prompt_file: Path) -> List[str]:
    """This function predicts if each line in the input file is written by a human or a machine."""
    # Parse the prompt data
    prompt_data = parse_promt_data(prompt_file)
    texts = [entry['text'] for entry in prompt_data]
    
    # Calculate the average comma/point ratio for the entire data
    total_point_count = 0
    total_comma_count = 0
    for line in texts:
        total_point_count += line.text.count('.')
        total_comma_count += line.text.count(',')
    
    if total_point_count == 0:
        average_ratio = 0  # Avoid division by zero
    else:
        average_ratio = total_comma_count / total_point_count
    
    # Initialize the results list
    results = []
    
    # Analyze each line using the calculated average ratio as threshold
    for line in texts:
        point_count = line.text.count('.')
        comma_count = line.text.count(',')
        
        if point_count > 0:
            ratio = comma_count / point_count
            if ratio > average_ratio:
                results.append("Human")
            else:
                results.append("AI")
        else:
            # If there are no points, we can't calculate the ratio. Default to "Human"
            results.append("Human")
    
    return results

def main():
    # Load the test data
    #human_data: Path = Path('human.jsonl')
    #machine_data: Path = Path('group1.jsonl')

    human_data_path: Path = Path('human_sample.jsonl')
    machine_data_path: Path = Path('group1.jsonl')

    # Perform morphology analysis on the human and machine data
    print(morphology_results(human_data_path))
    #morphology_results(machine_data_path)

    # Process the data
    #human_texts, machine_texts = get_and_parse_texts(human_data, machine_data)

    # Perform the analysis on the test data
    #morpology_analysis(human_texts, machine_texts)

if __name__ == '__main__':
    main()

# sem #
