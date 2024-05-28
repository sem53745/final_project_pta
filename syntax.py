# Program name: syntax.py
# Date: 03/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)


# Mervyn #

# import preprocessor
from preprocessor import get_and_parse_texts, Path
from typing import List, Dict
from spacy.tokens import Doc

# import other necessary packages
from collections import Counter


def get_ratio_dict(data: List[Doc]) -> Dict[str, List[float]]:
    '''
    Function that takes data from the preprocessor and returns a dictionary with tags and their ratio values
    parameter data: list of docs from the preprocessor
    dictionary key: tag as a string
    dictionary value: list of ratios calculated per doc
    '''

    # initialize dictionary
    tag_dict: Dict[str, List[float]] = {}

    # loop for every doc in the data, which corresponds to: for every text line in the json file
    for doc in data:

        # get the text for the length and remove newlines
        text: str = doc.text
        text = text.replace('\n', '')
        pos_tags: List[str] = []

        # create a Counter dictionary of the frequency per tag
        for token in doc:
            pos_tags.append(token.tag_)
        tag_counter_dict = Counter(pos_tags)

        # calculate ratio and add tag and ratio to tag_dict
        for key, value in tag_counter_dict.items():
            ratio = value / len(text)
            if key not in tag_dict.keys():
                tag_dict[key] = [ratio]
            else:
                tag_dict[key] += [ratio]

    return tag_dict


def calculate_average_ratios(combined_ratios_dict: Dict[str, List[float]]) -> Dict[str, float]:
    '''
    Function that takes all the ratios and calculates the average which is returned in a dictionary.
    parameter combined_ratios_dict: dictionary with tags and a list of all the ratios calculated per text line in the json file
    dictionary key: tag as a string
    dictionary value: average ratio as a float
    '''

    # sub function for calculating the average ratio
    def calculate_average_ratio(values: List[float]) -> float:
        total_ratio = 0
        for value in values:
            total_ratio += value
        return total_ratio / len(values)

    # initialize dictionary
    average_ratio_dict: Dict[str, float] = {}

    # create a dictionary with every tag and their average ratio
    for key, value in combined_ratios_dict.items():
        ratio = calculate_average_ratio(value)
        average_ratio_dict[key] = ratio
    
    return average_ratio_dict


def main():

    # get data
    human_text, machine_text = get_and_parse_texts(Path('human_sample.jsonl'), Path('group1_sample.jsonl'))

    human_ratios = get_ratio_dict(human_text)
    machine_ratios = get_ratio_dict(machine_text)
    human_average_ratios = calculate_average_ratios(human_ratios)
    machine_average_ratios = calculate_average_ratios(machine_ratios)

    # showing ratios for testing
    print('Human')
    for key, value in human_average_ratios.items():
        print(f'{key} : {value}')
    print('\n\n')
    print('Machine')
    for key, value in machine_average_ratios.items():
        print(f'{key} : {value}')


if __name__ == '__main__':
    main()

# Mervyn #