# Program name: semantics.py
# Date: 02/06
# Contributors: Joris van Bruggen (s5723752),  Tieme Boerema (s5410762)

from preprocessor import get_and_parse_texts, Path, parse_prompt_data
from typing import Tuple, List, Dict, Literal
from spacy.tokens import Doc
from fastcoref import spacy_component
from collections import Counter
from nltk.corpus import wordnet as wn
import nltk


def perform_analysis(texts: List[Doc]):
    ''' This function performs a few analyses on each doc within a list of docs.
       These are: calculating the amount of references per coreference cluster,
       calculating the average amount of Named Entities per sentence, and
       calculating the average synsets per verb. These values are then returned,
       to be used to compute the separator values. '''

    total_reference_amount = 0
    total_coref_amount = 0
    amount_of_sentences = 0
    amount_of_NEs = 0
    amount_of_synsets = 0
    amount_of_verbs = 0

    # For each doc, retrieve a bunch of values, and add them to their specific total.
    # These totals are then used to calculate the variables that will be returned by this function.
    for doc in texts:
        coref_amount, reference_amount, sentence_amount, NE_amount, verb_amount, synset_amount = perform_analysis_single(doc)
        amount_of_sentences += sentence_amount
        amount_of_NEs += NE_amount
        total_coref_amount += coref_amount
        total_reference_amount += reference_amount
        amount_of_verbs += verb_amount
        amount_of_synsets += synset_amount

    # Calculate values which will later be used to compute separator values.
    references_per_cluster = total_reference_amount / total_coref_amount
    average_NE_sentence = amount_of_NEs / amount_of_sentences
    synsets_per_verb = amount_of_synsets / amount_of_verbs

    return references_per_cluster, average_NE_sentence, synsets_per_verb


def perform_analysis_single(doc:Doc):
    ''' This function is similar to perform_analysis, but is specific to a single doc.
       The main difference lies in the fact that this single analysis function is used
       to analyze test data one by one, while the average values calculated in
       perform_analysis are calculated across all of the test data.'''

    # Retrieve a bunch of values, which are later compared to separator values
    # if test data is used, in order to determine whether a text is human or AI.
    # If the function is used by perform_analysis, the data is used to create these separators.
    synset_amount = 0
    sentence_amount = len(list(doc.sents))
    NE_amount = len(list(doc.ents))
    coref_amount = len(list(doc._.coref_clusters))
    reference_amount = len(list(reference for cluster in doc._.coref_clusters for reference in cluster))
    verb_amount = len(list(word for word in doc if word.pos_ == "VERB"))

    for word in doc:
        if word.pos_ == "VERB":
            lemma = word.lemma_
            synsets = wn.synsets(lemma, pos=wn.VERB)
            synset_amount += len(synsets)

    return coref_amount, reference_amount, sentence_amount, NE_amount, verb_amount, synset_amount


def do_semantic_analysis(human_texts: List[Doc], machine_texts: List[Doc]):
    ''' This function uses values calculated by perform_analysis 
       to calculate separator values based on the average of the human
       and machine text values.'''
    
    # Retrieve values for both machine and human texts
    # which will then be used to calculate separator values.
    machine_reference_amount, machine_NE_sentence, machine_synsets_verb = perform_analysis(machine_texts)
    human_reference_amount, human_NE_sentence, human_synsets_verb = perform_analysis(human_texts)

    # Calculate separator values for each classification category.
    separator_value_NE_sentence = (machine_NE_sentence + human_NE_sentence) / 2
    separator_value_references = (machine_reference_amount + human_reference_amount) / 2
    separator_value_synsets_verb = (machine_synsets_verb + human_synsets_verb) / 2

    return (separator_value_NE_sentence, separator_value_references, separator_value_synsets_verb)


def human_or_ai(value_to_divide_by:int, value_to_divide:int, separator:float):
    ''' This function is used for each classification category. It calculates a value
       comparable to the separator value for that specific category, and then
       compares it to the separator. In each case, if the value is higher than
       the separator, it is more likely to be a human text and human is returned.
       Otherwise, AI is returned.'''

    if value_to_divide_by != 0:
        if (value_to_divide / value_to_divide_by) >= separator:
            return "Human"
        else:
            return "AI"


def get_semantic_results(separators:tuple[float, float, float], prompts: List[Dict[str, Doc | str]]) -> List[Tuple[Literal['Human', 'Unsure', 'AI'], float]]:
    ''' This function takes as input a list of prompts (test data). These prompts are then
       analyzed individually and compared to the patterns found in the training data.
       The function then makes a decision of Human or AI based on the values found;
       if two or more of the classification categories point in one direction, that label will be
       assigned. In the end, the list of answers is returned, to be used by the main program
       in order to assign a definitive label. '''

    human_counter = 0
    ai_counter = 0
    separator_NE_sentence, separator_references, separator_synsets_verb = separators
    answers = []
    for prompt in prompts:
        # Retrieve required values for this prompt.
        coref_current, references_current, sentences_current, NE_current, verbs_current, synsets_current = perform_analysis_single(prompt['text'])

        if human_or_ai(sentences_current, NE_current, separator_NE_sentence) == "Human":
            human_counter +=1
        else:
            ai_counter += 1

        if human_or_ai(coref_current, references_current, separator_references) == "Human":
            human_counter +=1
        else:
            ai_counter += 1

        if human_or_ai(verbs_current, synsets_current, separator_synsets_verb) == "Human":
            human_counter +=1
        else:
            ai_counter += 1

        # Here, certainty is calculated based on the amount of classification 
        # categories that assigned the same label to the text.
        max_score = 3
        guessed = max(human_counter, ai_counter)
        certainty = guessed / max_score
        if human_counter > ai_counter:
            answers.append(("Human", certainty))
        elif human_counter == ai_counter:
            answers.append(("Unsure", certainty))
        else:
            answers.append(("AI", certainty))

        human_counter = 0
        ai_counter = 0

    return answers


def main():

    # Load training data
    human_text, machine_text = get_and_parse_texts(Path('human.jsonl'), Path('group1.jsonl'))

    # Compute the three separator values
    NE_sentence_separator, references_amount_separator, synsets_verb_separator = do_semantic_analysis(human_text, machine_text)

    # Load test data.
    prompts = parse_prompt_data(Path('prompts.jsonl'))
    
    # Compute results
    results = get_semantic_results((NE_sentence_separator, references_amount_separator, synsets_verb_separator), prompts)


if __name__ == "__main__":
    main()
