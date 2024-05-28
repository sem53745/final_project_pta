# Program name: pragmatics.py
# Date: 03/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)


# Jasper #

from preprocessor import get_and_parse_texts
from spacy.language import Doc
from typing import List, Tuple


def get_sentiment_analysis(data: List[Doc]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    '''
    Function to check the sentiment of the data
    param data: List[Doc], the data to check the sentiment of
    '''
    max_sentiment: Tuple[float, float] = (0, 0)
    min_sentiment: Tuple[float, float] = (0, 0)
    avg_sentiment: Tuple[float, float] = (0, 0)
    for doc in data:
        if doc._.blob.polarity > max_sentiment[0]:
            max_sentiment = (doc._.blob.polarity, doc._.blob.subjectivity)
        if doc._.blob.polarity < min_sentiment[0]:
            min_sentiment = (doc._.blob.polarity, doc._.blob.subjectivity)
        avg_sentiment = (avg_sentiment[0] + doc._.blob.polarity, avg_sentiment[1] + doc._.blob.subjectivity)
  
    print(f'The maximum positive sentiment is: {max_sentiment[0]:.4f}, with a subjectivity of: {max_sentiment[1]:.4f}')
    print(f'The minimum negative sentiment is: {min_sentiment[0]:.4f}, with a subjectivity of: {min_sentiment[1]:.4f}')
    print(f'The average sentiment is: {avg_sentiment[0] / len(data):.4f}, with a average subjectivity of: {avg_sentiment[1] / len(data):.4f}\n')
    return max_sentiment, min_sentiment, avg_sentiment


def main():
    
    # get the data
    human, machine = get_and_parse_texts('human.jsonl', 'group1.jsonl')
    print('\n\n\n')

    # check the sentiment of the data
    print('human sentiment analysis:')
    human_sentiment = get_sentiment_analysis(human)
    print('machine sentiment analysis:')
    machine_sentiment = get_sentiment_analysis(machine)


if __name__ == '__main__':
    main()

# Jasper #