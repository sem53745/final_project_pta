# Program name: maybefinal.py
# Date: 30/05
# Contributors: Joris van Bruggen (s5723752), Mervyn Bolhuis (s5119103), Tieme Boerema (s5410762), Jasper Kleine (s5152372), Sem Bartels (s5374588)

from preprocessor import get_and_parse_texts, Path, parse_promt_data
import spacy
from fastcoref import spacy_component
from collections import Counter


def perform_analysis(texts):

    total_coref_amount = 0
    for doc in texts:
        #amount_of_sentences = 0
        #amount_of_NEs = 0
        #total_reference_amount = 0
        #for sent in doc.sents:
            #amount_of_sentences += 1
            
       #for ent in doc.ents:
            #amount_of_NEs += 1
            
        for cluster in doc._.coref_clusters:
            total_coref_amount += 1
            #for reference in cluster:
                #total_reference_amount += 1

    average_coref_amount = total_coref_amount / len(texts)

    return average_coref_amount


def perform_analysis_single(doc):

    coref_amount = 0
    for cluster in doc._.coref_clusters:
        coref_amount += 1

    return coref_amount


def do_semantic_analysis(human_texts, machine_texts):

    machine_data = perform_analysis(machine_texts)    
    human_data = perform_analysis(human_texts)

    separator_value = (machine_data + human_data) / 2
    
    return separator_value


def get_semantic_results(separator_coref_amount, prompts):

    human_counter = 0
    ai_counter = 0
    answers = []
    for prompt in prompts:
        coref_current = perform_analysis_single(prompt['text'])
        if coref_current > separator_coref_amount:
            human_counter += 1
        else:
            ai_counter += 1

        if human_counter > ai_counter:
            answers.append("Human")
        else:
            answers.append("AI")

    return answers


def main():

    human_text, machine_text = get_and_parse_texts(Path('human.jsonl'), Path('group1.jsonl'))
    coref_amount_separator = do_semantic_analysis(human_text, machine_text)

    prompts = parse_promt_data(Path('prompts.jsonl'))
    results = get_semantic_results(coref_amount_separator, prompts)


if __name__ == "__main__":
    main()
