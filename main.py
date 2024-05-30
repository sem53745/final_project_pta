# Program name: main.py
# Date: 03/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)

# Jasper #

# import our modules
from preprocessor import get_and_parse_texts, parse_promt_data, Path
from pragmatics import do_sentiment_analysis, get_sentiment_results
#from morphology import do_morpological_analysis, get_morpology_results
from syntax import do_syntactic_analysis, get_syntactic_results
#from semantics import do_semantic_analysis, get_semantic_results

# import the supporting packages
import argparse
import os
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from spacy.tokens import Doc
from typing import NewType, Tuple, List
Error = NewType('Error', str)


def create_final_prediction(*results: List[str], true_labels: List[str]) -> None:
    '''
    Function to create the final prediction of the results
    param results: List[str], the results of the different analysis
    '''

    transposed_results: List[List[str]] = [[results[j][i] for j in range(len(results))] for i in range(len(results[0]))]

    predictions: List[str] = []
    for idx, text_predictions in enumerate(transposed_results):
        human: float = -(text_predictions.count('Human') * 0.99)
        machine: float = (text_predictions.count('AI') * 1.01)
        prediction: float = human + machine
        prediction_str = 'Human' if prediction < 0 else 'AI'
        predictions.append(prediction_str)
        print(f'The final prediction is: {prediction_str} because it scored: {prediction:.3f}')
        print (f'The true label is: {true_labels[idx]}\n')

    print('The final classification report is: \n')
    print(classification_report(true_labels, predictions, labels=['AI', 'Human'], zero_division=0))
    matrix = confusion_matrix(true_labels, predictions, labels=['AI', 'Human'])
    print('the confusion matrix is: \n', matrix)
    



def make_report(true_labels: List[str], pred_labels: List[str], by: str) -> None:
    '''
    Function to make the classification report
    param true_labels: List[str], the true labels of the data
    param pred_labels: List[str], the predicted labels of the data
    '''

    print(f'The report for the {by} results are: \n')
    print(classification_report(true_labels, pred_labels))
    matrix = confusion_matrix(true_labels, pred_labels)
    print('the confusion matrix is: \n', matrix)
    print('\n')


def create_parser():
    '''
    Create the parser for the command line arguments
    There are 2 required arguments on the command line:
    1. The path to the human data 
    2. The path to the machine data
    '''
    parser = argparse.ArgumentParser(description='detection of AI generated text using NLP techniques')
    parser.add_argument('-t', '--training', metavar=('<human_data>', '<machine_data>'), nargs=2, type=str,
                        help='Path to the human and machine data jsonl file')
    parser.add_argument('prompt', metavar="prompt data", type=str,
                        help='Path to the prompt data jsonl file')
    return parser.parse_args()


def check_file(data_path: Path) -> None | Error:
    '''
    Check if the given files exist and is of the correct type
    param human: str, the path to the human data jsonl file
    param machine: str, the path to the machine data jsonl file
    '''
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'{data_path} does not exist')
    if not data_path.endswith('.jsonl'):
        raise ValueError(f'{data_path}, File must be a .jsonl file')
    

def test_data(data: Doc) -> None | Error:
    '''
    Function to test the functions in the main program
    Param data: Doc, a Doc object to test the basic spacy nlp functions
    '''

    # test if the data exists
    if not data:
        raise ValueError('No human data found')
    
    # test the basic spacy token functions
    for token in data:
        if not token:
            raise ValueError('No tokens found')
        if not token.text:
            raise ValueError('No text found')
        if not token.lemma_:
            raise ValueError('No lemmas found')
        if not token.pos_:
            raise ValueError('No POS tags found')
        if not token.dep_:
            raise ValueError('No dependencies found')
        break
        
    for chunk in data.noun_chunks:
        if not chunk:
            raise ValueError('No noun chunks found')
        if not chunk.text:
            raise ValueError('No text found')
        if not chunk.root:
            raise ValueError('No root found')
        if not chunk.root.pos_:
            raise ValueError('No root POS found')
        if not chunk.root.dep_:
            raise ValueError('No dependencies found')
        if not chunk.root.head:
            raise ValueError('No head found')
        if not chunk.root.head.pos_:
            raise ValueError('No head POS found')
        break

    for ent in data.ents:
        if not ent:
            raise ValueError('No entities found')
        if not ent.text:
            raise ValueError('No text found')
        if not ent.label_:
            raise ValueError('No entity label found')
        break

    for cluster in data._.coref_clusters:
        if not cluster:
            raise ValueError('No coreference clusters found')
        if not cluster[0]:
            raise ValueError('No cluster found')
        if not cluster[0][0]:
            raise ValueError('No start index found')
        if not cluster[0][1]:
            raise ValueError('No end index found')
        break
    
    if not data._.blob.polarity:
        raise ValueError('No polarity found')
    if not data._.blob.subjectivity:
        raise ValueError('No subjectivity found')
    if not data._.blob.sentiment_assessments.assessments:
        raise ValueError('No sentiment assessment found')


def main():

    args = create_parser()

    print('Loading the training data')
    if args.training:
        human_path = Path(args.training[0])
        machine_path = Path(args.training[1])

        check_file(human_path)
        check_file(machine_path)
        print('File paths are checked')

        # load the data from the jsonl files
        human, machine = get_and_parse_texts(human_path, machine_path)
        print('Data is loaded')

        test_data(human[0])
        print('All required spaCy-attributes are set')
    
    else:
        human, machine = get_and_parse_texts(Path('human.jsonl'), Path('group1.jsonl'))
        print('Data is loaded')

    print('\nLoading the prompt data')
    prompt_path = Path(args.prompt)

    check_file(prompt_path)

    # load the data from the jsonl files
    prompts = parse_promt_data(prompt_path)
    true_labels: List[str] = [prompt['by'] for prompt in prompts] # type: ignore

    # for the morphological analysis
    #something = do_morpological_analysis(human, machine)
    #morpology_prediction = get_morpology_results(something, prompts)
    #make_report(true_labels, semantic_prediction, 'morpological')

    # for the syntactic analysis
    ratios = do_syntactic_analysis(human, machine)
    syntactic_prediction = get_syntactic_results(ratios, prompts)
    #make_report(true_labels, syntactic_prediction, 'syntactic')

    # for the semantic analysis
    #something = do_semantic_analysis(human, machine)
    #semantic_prediction = get_semantic_results(human, machine, prompts)
    #make_report(true_labels, semantic_prediction, 'semantic')

    # for the pragmatic analysis
    polarity, subjectivity =  do_sentiment_analysis(human)
    comparison_data: Tuple[float, float, float, float] = (polarity[0], polarity[1], subjectivity[0], subjectivity[1])
    sentiment_prediction = get_sentiment_results(prompts, comparison_data)
    #make_report(true_labels, sentiment_prediction, 'pragmatic')

    # create the final prediction
    create_final_prediction(syntactic_prediction, sentiment_prediction, true_labels=true_labels)    


if __name__ == '__main__':
    main()

# Jasper #