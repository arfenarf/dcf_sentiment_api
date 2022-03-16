import pickle
from multiprocessing import freeze_support

import gensim
import numpy as np
import pandas as pd
import pyLDAvis
import spacy
import tqdm
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
from pyLDAvis import gensim_models

from nlp_utils.text_processing import prepare_text, prepare_save_corpus_dict


# Compute Coherence Score

def compute_coherence(corpus, dictionary, k, b, a):
    ldamodel = gensim.models.ldamulticore.LdaModel(corpus, num_topics=k, id2word=dictionary, passes=15,
                                               alpha=a, eta=b, chunksize=100)
    coherence = CoherenceModel(model=ldamodel, texts=text_data, dictionary=dictionary, coherence='c_v')
    return coherence.get_coherence()


if __name__ == '__main__':
    GENERATE_DATA = True
    spacy.load('en_core_web_sm')
    freeze_support()
    if GENERATE_DATA:
        text_data = prepare_text(input_type='file', file_path='data/student_responses.tsv',
                                 train_percent=0.25)
        prepare_save_corpus_dict(text_data)
        dictionary = gensim.corpora.Dictionary(text_data)
        corpus = [dictionary.doc2bow(text) for text in text_data]

    else:
        text_data= pickle.load(open('data/text_data.pkl', 'rb'))
        dictionary = gensim.corpora.Dictionary.load('data/dictionary.gensim')
        corpus = pickle.load(open('data/corpus.pkl', 'rb'))

    ## calculating out possible coherence scores and metaparameters

    grid = {}
    grid['Validation_Set'] = {}
    # Topics range
    min_topics = 2
    max_topics = 11
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)
    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')
    # Validation sets
    num_of_docs = len(corpus)
    corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
                   corpus]
    corpus_title = ['75% Corpus', '100% Corpus']
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                     }
    # Can take a long time to run
    if 1 == 1:
        pbar = tqdm.tqdm(total=len(corpus_sets) * len(topics_range) * len(alpha) * len(beta))
        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                # iterate through alpha values
                for a in alpha:
                    # iterare through beta values
                    for b in beta:
                        # get the coherence score for the given parameters
                        cv = compute_coherence(corpus=corpus_sets[i], dictionary=dictionary,
                                               k=k, a=a, b=b)
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)
                        pbar.update(1)
        pd.DataFrame(model_results).to_csv('models/lda_tuning_results.csv', index=False)
        pbar.close()



# lda_display = gensim_models.prepare(ldamodel, corpus, dictionary, sort_topics=False)
# ldavis_text = pyLDAvis.save_html(lda_display, 'static/ldavis.html')
