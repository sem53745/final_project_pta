# Program name: preprocessor.py
# Date: 03/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)


# Jasper #

# import the supporting packages
import json
import subprocess
from typing import Tuple, List, Dict, NewType
Path = NewType('Path', str)

# import the necessary packages
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from fastcoref import spacy_component # type: ignore
import nltk # type: ignore

# try importing the nltk wordnet package, if it fails, download it
try:
    from nltk.corpus import wordnet as wn # type: ignore
except ImportError:
    nltk.download('wordnet') # type: ignore
    try:
        from nltk.corpus import wordnet as wn # type: ignore
    except ImportError:
        exit('Please install the nltk wordnet package')

# try importing the spacytextblob package, if it fails, download it
try:
    from spacytextblob.spacytextblob import SpacyTextBlob # type: ignore
except ImportError:
    subprocess.run('python3 -m textblob.download_corpora', shell = True, executable="/bin/bash")
    try:
        from spacytextblob.spacytextblob import SpacyTextBlob # type: ignore
    except ImportError:
        exit('Please install the textblob package')


def get_and_parse_texts(human_data: Path, machine_data: Path) -> Tuple[List[Doc], List[Doc]]:
    ''' 
    Function to load and parse the texts from the jsonl files
    param human_data: str, the path to the jsonl file with the human data
    param machine_data: str, the path to the jsonl file with the machine data
    '''

    # load the spacy model and add the necessary components
    nlp: Language = spacy.load("en_core_web_sm")
    nlp.add_pipe('spacytextblob')
    nlp.add_pipe('fastcoref')

    # subfunction to load the jsonl files
    def load_jsonl(file_path: Path) -> List[Dict[str, str]]:
        with open(file_path, 'r') as file:
            data: List[Dict[str, str]] = [json.loads(line) for line in file]
        return data

    # subfunction to process the data with spacy
    def process_data(data: List[Dict[str, str]]) -> List[Doc]:
        # Extract text data
        text_data: List[str] = [entry['text'] for entry in data if 'text' in entry and entry['text'].strip()]
        docs: List[Doc] = list(nlp.pipe(text_data))
        return docs
    
    # load the data
    human_data_list: List[Dict[str, str]] = load_jsonl(human_data)
    machine_data_list: List[Dict[str, str]] = load_jsonl(machine_data)

    # process the data
    human_docs: List[Doc] = process_data(human_data_list)
    machine_docs: List[Doc] = process_data(machine_data_list)

    return human_docs, machine_docs        


def main():
    
    # load the data
    human_data: Path = Path('human_sample.jsonl')
    machine_data: Path = Path('group1_sample.jsonl')

    # process the data
    human, machine = get_and_parse_texts(human_data, machine_data)

    # print the length of the data as a check
    print(len(human))
    print(len(machine))


if __name__ == '__main__':
    main()

# Jasper #