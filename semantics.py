# Program name: semantics.py
# Date: 30/05
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)

from preprocessor import get_and_parse_texts, Path, parse_promt_data
import spacy
from fastcoref import spacy_component
from collections import Counter
from nltk.corpus import wordnet as wn
import nltk


def perform_analysis(texts):

    total_reference_amount = 0
    total_coref_amount = 0
    amount_of_sentences = 0
    amount_of_NEs = 0
    amount_of_synsets = 0
    amount_of_verbs = 0
    for doc in texts:
        for sent in doc.sents:
            amount_of_sentences += 1
            
        for ent in doc.ents:
            amount_of_NEs += 1
            
        for cluster in doc._.coref_clusters:
            total_coref_amount += 1
            for reference in cluster:
                total_reference_amount += 1

        for word in doc:
            if word.pos_ == "VERB":
                amount_of_verbs += 1
                lemma = word.lemma_
                synsets = wn.synsets(lemma, pos=wn.VERB)
                amount_of_synsets += len(synsets)

    references_per_cluster = total_reference_amount / total_coref_amount
    average_NE_sentence = amount_of_NEs / amount_of_sentences
    synsets_per_verb = amount_of_synsets / amount_of_verbs
    
    # Some print statements here to confirm values later

    return references_per_cluster, average_NE_sentence, synsets_per_verb


def perform_analysis_single(doc):

    coref_amount = 0
    reference_amount = 0
    NE_amount = 0
    sentence_amount = 0
    synset_amount = 0
    verb_amount = 0
    
    for sentence in doc.sents:
        sentence_amount += 1

    for ent in doc.ents:
        NE_amount += 1

    for cluster in doc._.coref_clusters:
        coref_amount += 1
        for reference in cluster:
            reference_amount += 1
    
    for word in doc:
        if word.pos_ == "VERB":
            verb_amount += 1
            lemma = word.lemma_
            synsets = wn.synsets(lemma, pos=wn.VERB)
            synset_amount += len(synsets)

    return coref_amount, reference_amount, sentence_amount, NE_amount, verb_amount, synset_amount


def do_semantic_analysis(human_texts, machine_texts):

    machine_reference_amount, machine_NE_sentence, machine_synsets_verb = perform_analysis(machine_texts)    
    human_reference_amount, human_NE_sentence, human_synsets_verb = perform_analysis(human_texts)

    separator_value_NE_sentence = (machine_NE_sentence + human_NE_sentence) / 2
    separator_value_references = (machine_reference_amount + human_reference_amount) / 2
    separator_value_synsets_verb = (machine_synsets_verb + human_synsets_verb) / 2
    
    return separator_value_NE_sentence, separator_value_references, separator_value_synsets_verb


def get_semantic_results(separator_NE_sentence, separator_references, separator_synsets_verb, prompts):

    human_counter = 0
    ai_counter = 0
    answers = []
    for prompt in prompts:
        coref_current, references_current, sentences_current, NE_current, verbs_current, synsets_current = perform_analysis_single(prompt['text'])
            
        if (references_current / coref_current) > separator_references:
            human_counter += 1
        else:
            ai_counter += 1
            
        if (NE_current / sentences_current) > separator_NE_sentence:
            human_counter += 1
        else:
            ai_counter += 1
            
        if (synsets_current / verbs_current) > separator_synsets_verb:
            human_counter += 1
        else:
            ai_counter += 1

        if human_counter > ai_counter:
            answers.append("Human")
            print("Human")
        else:
            answers.append("AI")
            print("AI")

    return answers


def main():

    human_text, machine_text = get_and_parse_texts(Path('human.jsonl'), Path('group1.jsonl'))
    NE_sentence_separator, references_amount_separator, synsets_verb_separator = do_semantic_analysis(human_text, machine_text)

    prompts = parse_promt_data(Path('prompts.jsonl'))
    results = get_semantic_results(NE_sentence_separator, references_amount_separator, synsets_verb_separator, prompts)


if __name__ == "__main__":
    main()
