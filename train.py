## Embedding
#********************************************************************************************************************************************

from tensorflow.keras.preprocessing.text import one_hot

#sentence
sent=[  'I am Rohit Tayde',
     'I like to drink juice',
     'the cup of tea',
    'The boy is king',
     'I am a good developer',
     'understand the meaning of words which i speak',
     'your videos are good',]

#define the vocablary size
voc_size=10000

#one hot representation
one_hot_rep=[one_hot(words,voc_size)for words in sent]
print(one_hot_rep)

## word embedding representation
from tensorflow.keras.layers import Embedding,Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

import numpy as np

sent_length=8
embedded_docs=pad_sequences(one_hot_rep,padding='pre',maxlen=sent_length)
print(embedded_docs)

# feature representation
dim=10

## compile the model
model=Sequential()
model.add(Input(shape=(sent_length,)))
model.add(Embedding(voc_size,10,input_length=sent_length))
model.compile('adam','mse')

model.summary()

# model prediction
model.predict(embedded_docs)

#_____________________________________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________________________________

# Simple RNN
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,Embedding, Dense,Input

# load the imdb dataset
max_features=10000
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)

print('Training data shape',x_train.shape)
print('Testing data shape',x_test.shape)

x_train[0]

# sample review and its label
sample_review=x_train[0]
label=y_train[0]
print('Sample review:',sample_review)
print('Label:',label)

# sample review and its label
sample_review=x_train[2]
label=y_train[2]
print('Sample review:',sample_review)
print('Label:',label)

# mapping of words index
word_index=imdb.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])

reverse_word_index

decoded_review=' '.join(reverse_word_index.get(i-3,'?') for i in sample_review)
decoded_review
# it convert the numerical review back into readable english word reverse_word_index join them into a sentence and print the decoded review

from tensorflow.keras.preprocessing import sequence
max_len=500
x_train=sequence.pad_sequences(x_train,maxlen=max_len)
x_test=sequence.pad_sequences(x_test,maxlen=max_len)

model=Sequential() # for creating sequential  model
model.add(Input(shape=(max_len,))) # input layer
model.add(Embedding(max_features,64,input_length=max_len))# Embedding layer
model.add(SimpleRNN(64))# RNN Hidden layer
model.add(Dense(1,activation='sigmoid'))# output layer

model.summary()

# compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Create an instant  of EarlyStopping callback
from tensorflow.keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

# train the model with early stopping
history=model.fit(x_train,y_train,batch_size=128,epochs=5,validation_data=(x_test,y_test),callbacks=[early_stopping])

# save the model
model.save('rnn_model.keras')