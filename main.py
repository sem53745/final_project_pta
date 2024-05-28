# Program name: main.py
# Date: 03/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)

# Jasper #

# import the necessary packages
from preprocessor import get_and_parse_texts
# from morphology import tokenize_and_lemmatize

# import the supporting packages
import argparse
import os


def create_parser():
    '''
    Create the parser for the command line arguments
    There are 2 required arguments on the command line:
    1. The path to the human data 
    2. The path to the machine data
    '''
    parser = argparse.ArgumentParser(description='detection of AI generated text using NLP techniques')
    parser.add_argument('human', metavar="human data", type=str, help='Path to the human data jsonl file')
    parser.add_argument('machine', metavar="machine data", type=str, help='Path to the machine data jsonl file')
    return parser.parse_args()


def check_files(human: str, machine: str):
    '''
    Check if the given files exist and is of the correct type
    param human: str, the path to the human data jsonl file
    param machine: str, the path to the machine data jsonl file
    '''
    if not os.path.exists(human):
        raise FileNotFoundError('File path does not exist')
    if not human.endswith('.jsonl'):
        raise ValueError('File must be a .jsonl file')
    if not os.path.exists(machine):
        raise FileNotFoundError('File path does not exist')
    if not machine.endswith('.jsonl'):
        raise ValueError('File must be a .jsonl file')
    

def test_data(data: object):
    '''
    Function to test the functions in the main program
    '''

    if not data:
        raise ValueError('No human data found')
    
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

    print('All tests passed')


def main():

    args = create_parser()
    check_files(args.human, args.machine)

    # load the data from the jsonl files
    human, machine = get_and_parse_texts(args.human, args.machine)

    print(len(human))
    print(len(machine))

    test_data(human[0])


if __name__ == '__main__':
    main()

# Jasper #