import numpy as np
import tensorflow as tf
import tflearn
import string
import re
from tflearn.data_utils import load_csv

max_document_length = 20
min_word_freq = 20

def word_to_vocab(message): 
	return np.array(list(vp.fit_transform([message])))

def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return(text_string)

data, labels = load_csv('spam.csv', target_column=0, categorical_labels=True, n_classes=2)

text_messages = np.concatenate(data, axis=0)
text_messages = [clean_text(x) for x in text_messages]

vp = tflearn.data_utils.VocabularyProcessor(max_document_length, min_frequency=min_word_freq)
trainX = np.array(list(vp.fit_transform(text_messages)))
trainY = labels

net = tflearn.input_data([None, max_document_length])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)

del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]

model.fit(trainX, trainY, n_epoch=10, batch_size=16, show_metric=True)

model.save('model/model')