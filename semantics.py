# Program name: semantics.py
# Date: 01/06
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)

from preprocessor import get_and_parse_texts, Path, parse_prompt_data
from typing import Tuple, List, Dict, Literal
from spacy.tokens import Doc
from fastcoref import spacy_component
from collections import Counter
from nltk.corpus import wordnet as wn
import nltk


def perform_analysis(texts: List[Doc]):

    total_reference_amount = 0
    total_coref_amount = 0
    amount_of_sentences = 0
    amount_of_NEs = 0
    amount_of_synsets = 0
    amount_of_verbs = 0

    for doc in texts:
        coref_amount, reference_amount, sentence_amount, NE_amount, verb_amount, synset_amount = perform_analysis_single(doc)
        amount_of_sentences += sentence_amount
        amount_of_NEs += NE_amount
        total_coref_amount += coref_amount
        total_reference_amount += reference_amount
        amount_of_verbs += verb_amount
        amount_of_synsets += synset_amount

    references_per_cluster = total_reference_amount / total_coref_amount
    average_NE_sentence = amount_of_NEs / amount_of_sentences
    synsets_per_verb = amount_of_synsets / amount_of_verbs

    return references_per_cluster, average_NE_sentence, synsets_per_verb


def perform_analysis_single(doc:Doc):

    coref_amount = 0
    reference_amount = 0
    NE_amount = 0
    sentence_amount = 0
    synset_amount = 0
    verb_amount = 0

    sentence_amount += len(list(doc.sents))
    NE_amount += len(list(doc.ents))
    coref_amount += len(list(doc._.coref_clusters))
    reference_amount += len(list(reference for cluster in doc._.coref_clusters for reference in cluster))
    #synset_amount += len(list(wn.synsets(word.lemma_, pos=wn.VERB) for word in doc if word.pos_ == "VERB"))
    verb_amount += len(list(word for word in doc if word.pos_ == "VERB"))

    for word in doc:
        if word.pos_ == "VERB":
            #verb_amount += 1
            lemma = word.lemma_
            synsets = wn.synsets(lemma, pos=wn.VERB)
            synset_amount += len(synsets)

    return coref_amount, reference_amount, sentence_amount, NE_amount, verb_amount, synset_amount


def do_semantic_analysis(human_texts: List[Doc], machine_texts: List[Doc]):

    machine_reference_amount, machine_NE_sentence, machine_synsets_verb = perform_analysis(machine_texts)
    human_reference_amount, human_NE_sentence, human_synsets_verb = perform_analysis(human_texts)

    separator_value_NE_sentence = (machine_NE_sentence + human_NE_sentence) / 2
    separator_value_references = (machine_reference_amount + human_reference_amount) / 2
    separator_value_synsets_verb = (machine_synsets_verb + human_synsets_verb) / 2

    return (separator_value_NE_sentence, separator_value_references, separator_value_synsets_verb)


def human_or_ai(value_to_divide_by:int, value_to_divide:int, separator:float):

    if value_to_divide_by != 0:
        if (value_to_divide / value_to_divide_by) >= separator:
            return "Human"
        else:
            return "AI"


def get_semantic_results(seperators:tuple[float, float, float], prompts: List[Dict[str, Doc | str]]) -> List[Tuple[Literal['Human', 'Unsure', 'AI'], float]]:
    human_counter = 0
    ai_counter = 0
    separator_NE_sentence, separator_references, separator_synsets_verb = seperators
    answers = []
    for prompt in prompts:
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

        max_score = 3
        guessed = max(human_counter, ai_counter)
        certainty = guessed / max_score
        if human_counter > ai_counter:
            answers.append(("Human", certainty))
            #print(prompt['by'], end = ", ")
            #print("Human")
        elif human_counter == ai_counter:
            answers.append(("Unsure", certainty))
            #print(prompt['by'], end = ", ")
            #print("Unsure")
        else:
            answers.append(("AI", certainty))
            #print(prompt['by'], end = ", ")
            #print("AI")

        human_counter = 0
        ai_counter = 0



    return answers


def main():

    human_text, machine_text = get_and_parse_texts(Path('human.jsonl'), Path('group1.jsonl'))
    NE_sentence_separator, references_amount_separator, synsets_verb_separator = do_semantic_analysis(human_text, machine_text)

    prompts = parse_prompt_data(Path('prompts.jsonl'))
    results = get_semantic_results((NE_sentence_separator, references_amount_separator, synsets_verb_separator), prompts)


if __name__ == "__main__":
    main()
