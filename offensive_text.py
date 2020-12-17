import numpy as np
import json
import sys
import joblib
import csv
import re
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import ML_models

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

raw_data_train = []
raw_data_test = []
sentence_data = []
labels = []
label_texts = []

# number of vectors to use for encoding each word
embed_size = 100 
# max number of unique words 
max_features = 20000
# max number of words from review to use
maxlen = 75

# in_file_folder = './parsed_dataset/'
in_file_folder = './toxicity_dataset/'
save_folder = './trained_models/'
GloVe_txt = './glove/glove.twitter.27B.100d.txt'
tokenize_type = 'keras'
val_split = 0.3

model = None
classifier_RF = None
classifier_AdB = None


def load_text():
    print('Loading in feature data and labels...')
    word_lengths = []

    if(in_file_folder == './parsed_dataset/'):
        with open(in_file_folder + in_file, newline = '') as file:                                                                                          
            tweets = csv.reader(file)
            next(tweets)
            for tweet in tweets:
                # clean tweet of RT, usernames, extraneous symbols
                x = tweet[2].strip()
                x = re.sub('@.*? ', '', x)
                x = re.sub('RT ', '', x)
                x = re.sub('http:.*?D', '', x)
                x = re.sub('http?://\S+', '', x)
                x = re.sub('\*', '', x)
                cleaned_tweet = re.sub('https?://\S+', '', x)
                word_lengths.append(len(cleaned_tweet.split()))

                # filter out a few examples which are garbage data
                if(word_lengths[-1] < 2000):
                    data = {'index': tweet[0],
                            'index2': tweet[1], 
                            'text': cleaned_tweet,
                            'bully_type' : tweet[3],
                            'label' : int(tweet[4])}

                    raw_data_train.append(data)
                else:
                    word_lengths.pop(-1)

    if(in_file_folder == './toxicity_dataset/'): 
        in_file = 'train_data.json'
        with open(in_file_folder + in_file, 'r') as file:
            for row in file:
                data = json.loads(row)
                x = data['text'].strip()
                x = re.sub('http:.*?D', '', x)
                x = re.sub('http?://\S+', '', x)
                x = re.sub('\*', '', x)
                data['text'] = re.sub('https?://\S+', '', x)

                raw_data_train.append(data)

        in_file = 'test_data.json'  
        with open(in_file_folder + in_file, 'r') as file:
            for row in file:
                data = json.loads(row)
                x = data['text'].strip()
                x = re.sub('http:.*?D', '', x)
                x = re.sub('http?://\S+', '', x)
                x = re.sub('\*', '', x)
                data['text'] = re.sub('https?://\S+', '', x)
                raw_data_test.append(data)

def texts_to_tokens(ind_texts, tokenizer, tokenize_type):
    if(tokenize_type == 'keras'):
        tokenizer.fit_on_texts(ind_texts)
        sent_seq = tokenizer.texts_to_sequences(ind_texts)

        return tokenizer.word_index, sent_seq

    else:
        # tokenization with spacy, deals with contractions differently
        en_nlp = spacy.load("en")

        word_index = {True: 1}
        num_unique_words = 0

        print('tokenizing...')
        for i,text in enumerate(ind_texts):
            if(i % 1000 == 0):
                print(i)
            tokenized = en_nlp(text)
            ind_texts[i] = [x.text for x in tokenized]

        print('Creating word_index...')
        # create word_index dict manually using spacy tokenization
        for text in ind_texts:
            for token in text:
                if(num_unique_words > (max_features-10)):
                    break
                if token not in word_index:
                    word_index[token] = num_unique_words+2
                    num_unique_words += 1

        print('Transform text to sequences...')
        # transform words to token values
        sent_seq = []
        for text in ind_texts:
            ind_seq = [word_index[token] if token in word_index else 0 for token in text]
            sent_seq.append(ind_seq)

        return word_index, sent_seq

# load pretrained GloVe embeddings
def loadGloVeEmbedding(GloVe_txt, word_index_dict):
    embedding_matrix = np.zeros((max_features, embed_size))
    embeddings_index = dict()
    with open(GloVe_txt, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    for word, index in word_index_dict.items():
        if (index > max_features - 1):
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return embedding_matrix

def train():
    # GloVe embedding used for neural network
    print('Loading GloVe Embedding...')
    embed_matrix = loadGloVeEmbedding(GloVe_txt, word_index_dict)
    # standardize shape for passing into ML Models
    data_text = pad_sequences(sent_seq, maxlen=maxlen)

    print('Loading models...')

    # model = ML_models.LSTM_model(max_features, embed_size, maxlen)
    model, es = ML_models.initialize_model(max_features, embed_size, maxlen, embed_matrix)

    classifier_RF = ML_models.init_RandomForest(num_trees = 100)
    classifier_AdB = ML_models.init_BoostingModel(num_estims = 100)

    print('Training Neural Network(s)...')
    print(model.summary())

    train_results = model.fit(data_text,
                                np.array(label_texts), 
                                validation_split=val_split, 
                                batch_size = 10000,   
                                epochs = 1)
    print()

    print('Training Random Forest...')
    classifier_RF.fit(tfidf_weights, trainy)
    predictions = classifier_RF.predict(test_weights)
    accuracy = metrics.accuracy_score(predictions, valy)
    print('Validation Accuracy (RF): ', accuracy)
    print()

    print('Training AdaBoost...')
    classifier_AdB.fit(tfidf_weights, trainy)
    predictions = classifier_AdB.predict(test_weights)
    accuracy = metrics.accuracy_score(predictions, valy)
    print('Validation Accuracy (AdaBoost): ', accuracy)


    save_weights = input('Do you want to save weights (y/n): ').strip().lower()
    if(save_weights == 'y'):
        print('Saving new model weights...')

        model.save_weights(save_folder + 'pretrained_NN')

        with open(save_folder + 'train_history', 'wb') as file:
            joblib.dump(train_results.history, file)


    save_classifiers = input('Do you want to save RF and AdaBoost (y/n): ').strip().lower()
    if(save_classifiers == 'y'):
        print('Saving classifiers...')
        with open(save_folder + 'RF_model.pkl', 'wb') as file_out:
            joblib.dump(classifier_RF, file_out)

        with open(save_folder + 'AdaBoost_model.pkl', 'wb') as file_out:
            joblib.dump(classifier_AdB, file_out)

    return model, classifier_RF, classifier_AdB

def evaluate(tokenizer, tfidf, model, classifier_RF, classifier_AdB):
    ind_texts_test = [x['text'] for x in raw_data_test]
    test_labels = [x['label'] for x in raw_data_test]

    embed_matrix = np.zeros((max_features, embed_size))

    sent_seq_test = tokenizer.texts_to_sequences(ind_texts_test)
    test_data = pad_sequences(sent_seq_test, maxlen=maxlen)

    test_weights = tfidf.transform(ind_texts_test)


    print('Loading previously saved models...')
    if(model is None):
        model, es = ML_models.initialize_model(max_features, embed_size, maxlen, embed_matrix)
        model.load_weights(save_folder + 'pretrained_NN').expect_partial()

    # joblib and pickle.load having erroneous predictions when loading saved pkl
    # models, retraining from scratch here on train data
    classifier_RF = ML_models.init_RandomForest(num_trees = 100)
    classifier_AdB = ML_models.init_BoostingModel(num_estims = 100)

    print('Training Random Forest...')
    classifier_RF.fit(tfidf_weights, trainy)

    print('Training AdaBoost...')
    classifier_AdB.fit(tfidf_weights, trainy)

    # if(classifier_RF is None):
    #     with open(save_folder + 'RF_model.pkl', 'rb') as input_file:
    #         classifier_RF = joblib.load(input_file)
    # if(classifier_AdB is None):
    #     with open(save_folder + 'AdaBoost_model.pkl', 'rb') as input_file:
    #         classifier_AdB = joblib.load(input_file)


    predictions = model.predict(test_data)
    predictions_RF = classifier_RF.predict(test_weights)
    predictions_AdB = classifier_AdB.predict(test_weights)

    # view 1st 5 predictions from test
    for i in range(5):
        print('Text: ', ind_texts_test[i])
        print('Label NN: ', predictions[i])
        print('Label RF: ', predictions_RF[i])
        print('Label AdB: ', predictions_AdB[i])
        print()


load_text()

ind_texts_train = [x['text'] for x in raw_data_train]

if(in_file_folder == './parsed_dataset/'):
    label_texts_train = [x['label'] for x in raw_data_train]
else:
    # in Toxicity dataset, comments can be given a 1 in any of 6 categories for offensive behavior
    # comments with 1 for any of labels given 1, otherwise 0
    label_texts = [1 if sum(x['label']) > 0 else 0 for x in raw_data_train]

tokenizer = Tokenizer(num_words= max_features, oov_token=True)
word_index_dict, sent_seq = texts_to_tokens(ind_texts_train, tokenizer, tokenize_type)

# tf-idf (term frequency, inverse doc frequency) used for RF and AdaBoost
# GloVe embedding achieved poor results when tested
print('Creating tf-idf matrix...')
trainX, valX, trainy, valy = train_test_split(ind_texts_train, 
                                        np.array(label_texts), 
                                        test_size = val_split, 
                                        shuffle=True
                                        )

tfidf = TfidfVectorizer(max_features = max_features)
# fit vocabulary to train data
tfidf_weights = tfidf.fit_transform(trainX)
test_weights = tfidf.transform(valX)

model, classifier_RF, classifier_AdB = train()

evaluate(tokenizer, tfidf, model, classifier_RF, classifier_AdB)



