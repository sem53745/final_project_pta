# Program name: pragmatics.py
# Date: 03/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)


# Jasper #

from preprocessor import get_and_parse_texts, Path
from spacy.tokens import Doc
from typing import List, Tuple

def get_sentiment_analysis(data: List[Doc]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
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
    print('human sentiment analysis:')
    get_sentiment_analysis(human)
    print('machine sentiment analysis:')
    get_sentiment_analysis(machine)


if __name__ == '__main__':
    main()

# Jasper #