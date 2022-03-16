import gensim
import pandas as pd
from gensim import corpora

import pickle
import nltk
from nltk import WordNetLemmatizer

from spacy.lang.en import English


def tokenize(text):

    parser = English()
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_stopwords():
    en_stop = set(nltk.corpus.stopwords.words('english'))
    new_stopwords = {'appointment', 'would', 'patient', 'thanks', 'thank', 'better', 'question'}
    en_stop = en_stop.union(new_stopwords)
    return en_stop


def tokenize_text_for_lda(text, stop):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in stop]
    tokens = [get_lemma2(token) for token in tokens]
    return tokens


def prepare_text(input_type = 'file',
                 input_text = None,
                 train_percent = 0.005,
                 file_path='data/student_responses.tsv'):

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    stop = prepare_stopwords()

    if input_type == 'file':
        text_data = []
        resp_df = pd.read_csv(file_path, sep='\t')
        train_df = resp_df.sample(frac = train_percent)
        for index, row in train_df.iterrows():
            tokens = tokenize_text_for_lda(row['response_value'], stop)
            if len(tokens) > 1:
                print(tokens)
                text_data.append(tokens)
        train_df['response_id'].to_csv('data/train_ids.csv', index=False)
        resp_df.loc[~resp_df.response_id.isin(train_df.response_id), 'response_id'].to_csv('data/test_ids.csv', index = False)


    else:
        text_data = [tokenize_text_for_lda(input_text, stop)]
    # Build the bigram model
    bigram = gensim.models.Phrases(text_data, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    text_data = make_bigrams(text_data)
    return text_data

def prepare_save_corpus_dict(text_data):
    pickle.dump(text_data, open('data/text_data.pkl', 'wb'))
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('data/corpus.pkl', 'wb'))
    dictionary.save('data/dictionary.gensim')

def calculate_save_model(corpus, k, dictionary, passes, a, b):
    ldamodel = gensim.models.ldamodel.LdaModel(corpus,
                                               num_topics=k,
                                               id2word=dictionary,
                                               passes=passes,
                                               alpha=a, eta=b)
    ldamodel.save(f'../models/model_{k}.gensim')
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)
