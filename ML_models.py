from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Input, Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.layers.merge import concatenate

from sklearn import ensemble

def initialize_model(max_features, embed_size, maxlen, embedding_matrix):
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen, weights=[embedding_matrix], trainable=False))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 6, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(Bidirectional(LSTM(embed_size, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    return model, es

def init_RandomForest(num_trees = 100):
    classifier = ensemble.RandomForestClassifier(n_estimators=num_trees)
    return classifier

def init_BoostingModel(num_estims = 50000):
    classifier = ensemble.AdaBoostClassifier(n_estimators = num_estims, learning_rate = 1)
    return classifier