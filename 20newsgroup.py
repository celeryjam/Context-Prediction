# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:52:19 2023

@author: lbhpham
"""

from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import TextVectorization, Dense, LSTM
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split

newsgroup_train = fetch_20newsgroups(subset='train')
newsgroup_test = fetch_20newsgroups(subset='test')

import re

def clean_str(string):
    string = re.sub(r"\n","", string) #remove new-line character
    string = re.sub(r"[^A-Za-z]", " ", string) #remove numbers and symbols
    string = string.strip().lower()
    return string

cleaned_articles = []
for article in newsgroup_train.data:
    cleaned_articles.append(clean_str(article))

cleaned_articles_test = []
for article in newsgroup_test.data:
    cleaned_articles_test.append(clean_str(article))
    
targets=[]
for target_name in newsgroup_train.target_names:
    targets.append(target_name)
targets=keras.utils.to_categorical(targets, dtype='uint8')

X_train, X_val, y_train, y_val = train_test_split(cleaned_articles,targets, train_size=0.8)

targets_test=[]
for target_name in newsgroup_test.target_names:
    targets_test.append(target_name)
    
    
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250
def tokenisation_by_keras(list_string,binary_vectorize=True, vocab_size=VOCAB_SIZE, max_sequence_length=MAX_SEQUENCE_LENGTH):
    
    '''
    Keras TextVectorization application of tokenization
    clean the text and vectorize it
    '''
    #string = Input(shape=(1,))
    #print(string)
    if binary_vectorize:
        vectorize_layer = TextVectorization(output_mode='binary') #max_tokens=vocab_size,
    else:
        vectorize_layer = TextVectorization(output_mode='int',max_sequence_length=1000) #max_tokens=vocab_size, output_sequence_length=max_sequence_length
        
    vectorize_layer.adapt(list_string)
    
    #Create a model that uses the vectorize text layer
    model = Sequential()
    # Start by creating an explicit input layer. It needs to have a shape of
    # (1,) (because we need to guarantee that there is exactly one string
    # input per batch), and the dtype needs to be 'string'.
    model.add(Input(shape=(1,), dtype=tf.string))
    # The first layer in our model is the vectorization layer. After this
    # layer, we have a tensor of shape (batch_size, max_len) containing
    # vocab indices.
    model.add(vectorize_layer)
    return model, vectorize_layer.vocabulary_size()

def build_lstm_model():
    model = Sequential()
    model.add(LSTM(1000, return_sequences=True))
    model.add(LSTM(1000))
    model.add(Dense(1000, activation="relu"))
    return model
    
def build_model():
    model = Sequential()
    tokenizer, vocab_size = tokenisation_by_keras(cleaned_articles)
    model.add(tokenizer)
    model.add(build_lstm_model())
    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))
    return model

train_hist = build_model().fit(X_train, y_train, validation_data=(X_val, y_val), epoch=20, batch_size=2, verbose=1)