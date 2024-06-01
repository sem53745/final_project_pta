# Program name: pragmatics.py
# Date: 03/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)


# Jasper #

from preprocessor import get_and_parse_texts, parse_prompt_data, Path
from spacy.tokens import Doc
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Tuple, Dict

# Jasper #
DEBUG = False


def pragmatic_predictor(text: Doc, comparison: Tuple[float, float, float, float]):
    '''
    Function to predict the pragmatic score of the data
    param data: Doc, the text to predict the pragmatic score of
    param comparison: Tuple[float, float, float, float], the comparison values to use
    '''

    # unpack the comparison values
    max_sent, min_sent, max_subj, min_subj = comparison

    # get the pragmatic values of the text
    pragmatic_polarity: float = text._.blob.polarity # type: ignore
    pragmatic_subjectivity: float = text._.blob.subjectivity # type: ignore

    # check if the pragmatic values are outside the "normal" values with a small margin
    ai_counter: int = 0
    if pragmatic_polarity > (max_sent * 1.025) or pragmatic_polarity < (min_sent * 1.025):
        ai_counter += 1
    if pragmatic_subjectivity > (max_subj * 1.025) or pragmatic_subjectivity < (min_subj * 1.025):
        ai_counter += 1

    # return the chance of the text being AI based on the pragmatic values being outside the "normal" values
    return ai_counter


def get_sentiment_results(prompts: List[Dict[str, Doc | str]], comparison_data: Tuple[float, float, float, float]) -> List[str]:
    '''
    function to write the sentiment results
    param prompts: List[Dict[str, Doc | str]], the data to check the sentiment of
    param comparison_data: Tuple[float, float, float, float], the "norm" values to use
    '''

    # get the data
    pred_list: List[str] = []
    for prompt in prompts:
        chance = pragmatic_predictor(prompt['text'], comparison_data) # type: ignore
        if chance > 0.0:
            pred_list.append('AI')
        else:
            pred_list.append('Human')

    return pred_list

def write_sentiment_results(prompts: List[Dict[str, Doc | str]], comparison_data: Tuple[float, float, float, float]) -> None:
    '''
    function to write the sentiment results
    param prompts: List[Dict[str, Doc | str]], the data to check the sentiment of
    param comparison_data: Tuple[float, float, float, float], the "norm" values to use
    '''

    # get the data
    true_list: List[str] = [prompt['by'] for prompt in prompts] # type: ignore
    pred_list: List[str] = get_sentiment_results(prompts, comparison_data)

    # print the classification report with sklearn
    print(classification_report(true_list, pred_list))

    # print the confusion matrix with sklearn
    matrix = confusion_matrix(true_list, pred_list)
    print('confusion matrix:\n', matrix)


def do_sentiment_analysis(data: List[Doc]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    '''
    Function to check the sentiment of the data
    param data: List[Doc], the data to check the sentiment of
    '''
    max_sentiment: float = 0.0
    min_sentiment: float = 0.0
    avg_sentiment: float = 0.0
    max_subjectivity: float = 0.0
    min_subjectivity: float = 0.0
    avg_subjectivity: float = 0.0

    for doc in data:

        if doc._.blob.polarity > max_sentiment:
            max_sentiment = doc._.blob.polarity
        if doc._.blob.polarity < min_sentiment:
            min_sentiment = doc._.blob.polarity
        avg_sentiment += doc._.blob.polarity

        if doc._.blob.subjectivity > max_subjectivity:
            max_subjectivity = doc._.blob.subjectivity
        if doc._.blob.subjectivity < min_subjectivity:
            min_subjectivity = doc._.blob.subjectivity
        avg_subjectivity += doc._.blob.subjectivity

    if DEBUG:
        print(f'The maximum positive sentiment is: {max_sentiment:.4f}')
        print(f'The minimum negative sentiment is: {min_sentiment:.4f}')
        print(f'The average sentiment is: {avg_sentiment / len(data):.4f}\n')

        print(f'The maximum subjectivity is: {max_subjectivity:.4f}')
        print(f'The minimum subjectivity is: {min_subjectivity:.4f}')
        print(f'The average subjectivity is: {avg_subjectivity / len(data):.4f}\n')

    return (max_sentiment, min_sentiment, avg_sentiment / len(data)), (max_subjectivity, min_subjectivity, avg_subjectivity / len(data))


def main():

    # get the data
    human, machine = get_and_parse_texts(Path('human.jsonl'), Path('group1.jsonl'))
    print('\n')

    # check the sentiment of the data
    polarity, subjectivity =  do_sentiment_analysis(human)
    do_sentiment_analysis(machine)

    comparison_data: Tuple[float, float, float, float] = (polarity[0], polarity[1], subjectivity[0], subjectivity[1])

    # get the prompt data
    prompts = parse_prompt_data(Path('prompts.jsonl'))

    write_sentiment_results(prompts, comparison_data)


if __name__ == '__main__':
    main()

# Jasper #
