# from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
# import numpy as np

import tensorflow as tf
import theano
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional
from sklearn.preprocessing import OneHotEncoder
from keras.engine import Input, Model, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,confusion_matrix
import os
import pydot
from keras.models import load_model
import graphviz

# Convert the letters to numerical format
def letter_to_index(letter):
    _alphabet = 'ATGCN'
    if letter not in _alphabet:
        print ("Anomaly Found")
        print (letter)
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)

# Format the Sequence
def format_sequences(sequences):
    max_sequence_len = 250
    formatted_sequence = []
    for sequence in sequences:
        formatted_sequence.append([int(letter_to_index(charecter)) for charecter in sequence ])
    formatted_sequence = np.array(formatted_sequence)
    return pad_sequences(formatted_sequence, maxlen = max_sequence_len)

def make_prediction(formatted_sequence):
    path = "rnn_24_model.h5"
    model_loaded = load_model("rnn_24_model.h5")
    y_score = model_loaded.predict_proba(formatted_sequence) 
    print (y_score)

# Predict the label for the given Riboswitch Sequences 
def predict(sequences):
    formatted_sequence = format_sequences(sequences)
    make_prediction(formatted_sequence)
    print ("Done")

# Input Sequence
sequences = [
    "TTTTTTTTGCAGGGGTGGCTTTAGGGCCTGAGAAGATACCCATTGAACCTGACCTGGCTAAAACCAGGGTAGGGAATTGCAGAAATGTCCTCATT",
    "CTCTTATCCAGAGCGGTAGAGGGACTGGCCCTTTGAAGCCCAGCAACCTACACTTTTTGTTGTAAGGTGCTAACCTGAGCAGGAGAAATCCTGACCGATGAGAG",
    "CCACGATAAAGGTAAACCCTGAGTGATCAGGGGGCGCAAAGTGTAGGATCTCAGCTCAAGTCATCTCCAGATAAGAAATATCAGAAAGATAGCCTTACTGCCGAA"
]

predict(sequences)
