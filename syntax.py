# Program name: syntax.py
# Date: 03/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)


# Mervyn #

# import preprocessor
from preprocessor import get_and_parse_texts

# import other necessary packages
from collections import Counter
import nltk


def get_ratio_dict(data: list) -> dict:
    '''
    Function that takes data from the preprocessor and returns a dictionary with tags and their ratio values
    parameter data: list of docs from the preprocessor
    dictionary key: tag as a string
    dictionary value: list of ratios calculated per doc
    '''

    # initialize dictionary
    tag_dict = {}

    # loop for every doc in the data, which corresponds to: for every text line in the json file
    for doc in data:

        pos_tags = []

        # create a Counter dictionary of the frequency per tag
        for token in doc:
            pos_tags.append(token.tag_)
        tag_counter_dict = Counter(pos_tags)

        # calculate ratio and add tag and ratio to tag_dict
        for key, value in tag_counter_dict.items():
            ratio = value / len(pos_tags)
            if key not in tag_dict.keys():
                tag_dict[key] = [ratio]
            else:
                tag_dict[key] += [ratio]

    return tag_dict


def calculate_average_ratios(combined_ratios: dict) -> dict:
    '''
    Function that takes all the ratios and calculates the average which is returned in a dictionary.
    parameter combined_ratios_dict: dictionary with tags and a list of all the ratios calculated per text line in the json file
    dictionary key: tag as a string
    dictionary value: average ratio as a float
    '''

    # sub function for calculating the average ratio
    def calculate_average_ratio(values: list):
        total_ratio = 0
        for value in values:
            total_ratio += value
        return value / len(values)

    # initialize dictionary
    average_ratio_dict = {}

    # create a dictionary with every tag and their average ratio
    for key, value in combined_ratios.items():
        ratio = calculate_average_ratio(value)
        average_ratio_dict[key] = ratio
    
    return average_ratio_dict


def calculate_measure_ratios(human_average_ratio : dict, machine_average_ratio: dict) -> dict:

    measure_ratios_dict = {}

    for key, value in human_average_ratio.items():
        if key in machine_average_ratio.keys():
            average_ratio = (value + machine_average_ratio[key]) / 2
            measure_ratios_dict[key] = average_ratio
    
    return measure_ratios_dict


def human_machine_decider(measure_ratios: dict, human_average_ratios: dict, machine_average_ratios: dict, unknown_average_ratios: dict):
    
    # initialize counters
    human_counter = 0
    machine_counter = 0

    # loop for every key/value in the measurement ratios
    for key, value in measure_ratios.items():

        # check if the key exists in the unknown dict
        if key in unknown_average_ratios.keys():

            # check if the ratio of the unknown file is bigger or smaller than the measure ratio
            if unknown_average_ratios[key] > value:
                if human_average_ratios[key] > machine_average_ratios[key]:
                    human_counter += 1
                elif human_average_ratios[key] < machine_average_ratios[key]:
                    machine_counter += 1
            elif unknown_average_ratios[key] < value:
                if human_average_ratios[key] > machine_average_ratios[key]:
                    machine_counter += 1
                elif human_average_ratios[key] < machine_average_ratios[key]:
                    human_counter += 1
            print(f'key measure: {key}, value measure: {value}')
            print(f'ratio for unknown file: {unknown_average_ratios[key]}')
            print(f'human ratio: {human_average_ratios[key]}, machine ratio: {machine_average_ratios[key]}')
            print(f'human counter: {human_counter}, machine counter: {machine_counter}')
            print('\n')

    print(f'human counter: {human_counter}, machine counter: {machine_counter}')

    # return human or machine according to the counters, if equal return undecided
    if human_counter > machine_counter:
        return 'Human'
    elif human_counter < machine_counter:
        return 'Machine'
    else:
        return 'Undecided'


def main():

    # get data
    human_text, machine_text = get_and_parse_texts('human.jsonl', 'group1.jsonl')

    human_ratios = get_ratio_dict(human_text)
    machine_ratios = get_ratio_dict(machine_text)
    human_average_ratios = calculate_average_ratios(human_ratios)
    machine_average_ratios = calculate_average_ratios(machine_ratios)
    measure_ratios = calculate_measure_ratios(human_average_ratios, machine_average_ratios)

    # for the unknown text
    unknown_text, unknown_text2 = get_and_parse_texts('human_sample.jsonl', 'group1.jsonl')
    unknown_ratios = get_ratio_dict(unknown_text)
    unknown_average_ratios = calculate_average_ratios(unknown_ratios)

    # test for human or machine
    answer = human_machine_decider(measure_ratios, human_average_ratios, machine_average_ratios, unknown_average_ratios)
    print(f'text1: {answer}')

    '''
    # showing ratios for testing
    print('Human')
    for key, value in sorted(human_average_ratios.items()):
        print(f'{key} : {value}')
    print('\n\n')
    print('Machine')
    for key, value in sorted(machine_average_ratios.items()):
        print(f'{key} : {value}')
    

    print('\n\n')
    print('Measure ratios:')
    for key, value in sorted(measure_ratios.items()):
        print(f'{key} : {value}')
    '''

if __name__ == '__main__':
    main()

# Mervyn #