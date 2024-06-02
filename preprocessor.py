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
        subprocess.run('pip install -r requirements.txt', shell = True, executable="/bin/bash")
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
            try:
                subprocess.run('pip install -r requirements.txt', shell = True, executable="/bin/bash")
            except ImportError:
                exit('Please install the textblob package')


# subfunction to load the jsonl files
def load_jsonl(file_path: Path) -> List[Dict[str, str]]:
    """
    Loads a jsonl file
    :param file_path: str, the path to the jsonl file
    :return: list of dictionaries, with each dictionary being a line in the jsonl file
    """
    with open(file_path, 'r') as file:
        data: List[Dict[str, str]] = [json.loads(line) for line in file]
    return data


# subfunction to process the data with spacy
def process_prompt_data(data: List[Dict[str, str]], nlp: Language, annotation: str | None = None) -> List[Dict[str, Doc | str]]:
    """
    Processes the prompt data with spacy
    :param data: list of dictionaries, the data to process
    :param nlp: spacy model, the spacy model to use for processing
    :param annotation: str, the annotation for the data
    :return: list of dictionaries, the processed data
    """
    # Extract text data
    text_data: List[str] = [entry['text'] for entry in data if 'text' in entry and entry['text'].strip()]
    docs: List[Doc] = list(nlp.pipe(text_data))

    if annotation:
        return [{'text': doc, 'by': annotation} for doc in docs]
    else:
        text_by: List[str] = [entry['by'] for entry in data if 'by' in entry and entry['by'].strip()]
        return [{'text': doc, 'by': by} for doc, by in zip(docs, text_by)]



def process_data(data: List[Dict[str, str]], nlp: Language) -> List[Doc]:
    """
    Process the data into spacy docs
    :param data: list of dictionaries, the data to process
    :param nlp: spacy model, the spacy model to use for processing
    :return: list of spacy docs, the processed data
    """
    # Extract text data
    text_data: List[str] = [entry['text'] for entry in data if 'text' in entry and entry['text'].strip()]
    docs: List[Doc] = list(nlp.pipe(text_data))
    return docs


# subfunction to load the spacy model
def load_spacy_model() -> Language:
    """
    Loads the spacy model with all necessary components
    :return: spacy model, the loaded spacy model
    """
    nlp: Language = spacy.load("en_core_web_sm")
    nlp.add_pipe('spacytextblob')
    nlp.add_pipe('fastcoref')
    return nlp


def parse_prompt_data(prompt_file: Path) -> List[Dict[str, Doc | str]]:
    '''
    Function to parse the prompt data
    param prompt_file: str, the path to the jsonl file with the prompt data
    '''

    # load the spacy model
    nlp: Language = load_spacy_model()

    # load the prompt data from the jsonl file
    prompt_list: List[Dict[str, str]] = load_jsonl(prompt_file)

    # check if the prompt data is human or machine
    if 'human' in str(prompt_file).lower():
        prompt_data = process_prompt_data(prompt_list, nlp, 'Human')
    elif 'machine' in str(prompt_file).lower():
        prompt_data = process_prompt_data(prompt_list, nlp, 'AI')

    # if neither is found, the annotation is in the jsonl file itself
    else:
        prompt_data = process_prompt_data(prompt_list, nlp)

    return prompt_data


def get_and_parse_texts(human_data: Path, machine_data: Path) -> Tuple[List[Doc], List[Doc]]:
    '''
    Function to load and parse the texts from the jsonl files
    param human_data: str, the path to the jsonl file with the human data
    param machine_data: str, the path to the jsonl file with the machine data
    '''

    nlp: Language = load_spacy_model()

    # load the data
    human_data_list = load_jsonl(human_data)
    machine_data_list = load_jsonl(machine_data)

    # process the data
    human_docs = process_data(human_data_list, nlp)
    machine_docs = process_data(machine_data_list, nlp)

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

    promts = parse_prompt_data(Path('prompts.jsonl'))

    print(len(promts))


if __name__ == '__main__':
    main()

# Jasper #
