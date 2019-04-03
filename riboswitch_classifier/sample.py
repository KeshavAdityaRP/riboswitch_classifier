from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

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

def construct_output(class_wise_probabilty, predicted_classes):
    riboswitch_names = [
        'RF00050',
        'RF00059',
        'RF00162',
        'RF00167',
        'RF00168',
        'RF00174',
        'RF00234',
        'RF00380',
        'RF00504',
        'RF00521',
        'RF00522',
        'RF00634',
        'RF01051',
        'RF01054',
        'RF01055',
        'RF01057',
        'RF01725',
        'RF01726',
        'RF01727',
        'RF01734',
        'RF01739',
        'RF01763',
        'RF01767',
        'RF02683'
    ]
    result = []
    for riboswitch_classes, predicted_class in zip(class_wise_probabilty,predicted_classes):
        squence_component = {}
        print (riboswitch_classes.size)
        print (len(riboswitch_names))
        for riboswitch_class, name in zip(riboswitch_classes, riboswitch_names):
            squence_component[name] = riboswitch_class
        squence_component["perdicted_class"] = predicted_class
        result.append(squence_component)
    print ("done")
    print (result)

# Make a prediction
def make_prediction(formatted_sequence):
    path = "model/rnn_24_model.h5"
    model_loaded = load_model(path)
    class_wise_probabilty = model_loaded.predict_classes(formatted_sequence) 
    predicted_classes = model_loaded.predict(formatted_sequence)
    construct_output(class_wise_probabilty, predicted_classes)
    

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
