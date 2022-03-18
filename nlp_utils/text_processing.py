import pickle

import gensim
import nltk
import pandas as pd
import pyLDAvis
from gensim import corpora
from gensim.models import CoherenceModel
from nltk import WordNetLemmatizer
from pyLDAvis import gensim_models
from spacy.lang.en import English
from tqdm import tqdm


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


def prepare_stopwords():
    en_stop = set(nltk.corpus.stopwords.words('english'))
    new_stopwords = {'appointment', 'would', 'patient', 'thanks', 'thank', 'better', 'question'}
    en_stop = en_stop.union(new_stopwords)
    return en_stop


def tokenize_text_for_lda(text, stop):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in stop]
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    return tokens


def prepare_text(input_type='file',
                 input_text=None,
                 train_percent=0.005,
                 file_path='data/student_responses.tsv'):
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    stop = prepare_stopwords()

    if input_type == 'file':
        print('tokenizing training data')
        text_data = []
        resp_df = pd.read_csv(file_path, sep='\t')
        resp_df = resp_df.loc[(resp_df['question'] == 'improvement') & (resp_df['student_year'] == 'D3'), :]
        train_df = resp_df.sample(frac=train_percent)
        for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
            tokens = tokenize_text_for_lda(row['response_value'], stop)
            if len(tokens) > 1:
                text_data.append(tokens)
        train_df['response_id'].to_csv('data/train_ids.csv', index=False)
        resp_df.loc[~resp_df.response_id.isin(train_df.response_id), 'response_id'].to_csv('data/test_ids.csv',
                                                                                           index=False)


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
    dictionary.filter_extremes(no_below=5)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('data/corpus.pkl', 'wb'))
    dictionary.save('data/dictionary.gensim')


def calculate_save_model(corpus, k, dictionary, passes, a, b):
    ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus,
                                                       num_topics=k,
                                                       id2word=dictionary,
                                                       passes=passes,
                                                       alpha=a, eta=b)
    ldamodel.save(f'models/model_{k}_{a}_{b}.gensim')
    topics = ldamodel.print_topics(num_words=6)
    for topic in topics:
        print(topic)
    return ldamodel


def predict_topic(sent, model_path):
    model = gensim.models.ldamulticore.LdaMulticore.load(model_path)
    dictionary = corpora.Dictionary.load('data/dictionary.gensim')
    sent_processed = prepare_text(input_type='string', input_text=sent)
    sent_corp = [dictionary.doc2bow(text) for text in sent_processed][0]
    pred = dict(model[sent_corp])
    return pred


if __name__ == '__main__':
    # this will generate a new model and accompanying LDAVis for a given set of parameters
    corpus = pickle.load(open('data/corpus.pkl', 'rb'))
    dictionary = corpora.Dictionary.load('data/dictionary.gensim')

    # parameters from initial pass: a = 0.1, b = 0.05, k = 5
    k = 8
    a = 'symmetric'
    b = 0.91
    passes = 15
    ldamodel = calculate_save_model(corpus=corpus, k=k, dictionary=dictionary,
                                    a=a, b=b, passes=passes)

    # save the plot. Bear in mind that gensim is using antiquated modules that squawk harmlessly
    lda_display = gensim_models.prepare(ldamodel, corpus, dictionary, sort_topics=False)
    ldavis_text = pyLDAvis.save_html(lda_display, 'static/ldavis.html')
    text_data = pickle.load(open('data/text_data.pkl', 'rb'))
    coherence = CoherenceModel(model=ldamodel, texts=text_data, dictionary=dictionary, coherence='c_v')
    coherence.get_coherence()

    sent = "extracted patient's wisdom teeth used anaesthesia nobody died"
    clean_sent = prepare_text(input_type='text', input_text=sent)[0]
    predict_corpus = dictionary.doc2bow(clean_sent)
    ldamodel.get_document_topics(predict_corpus)

    print('wait if debugging')
