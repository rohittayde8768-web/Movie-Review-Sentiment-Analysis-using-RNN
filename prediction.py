import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# mapping of words index
word_index=imdb.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])

# load the keras model
model=load_model('rnn_model.keras')
model.summary()

# helper function
#1 function  to decode the review
def decode_review(encoded_review):
    return ' '.join(reverse_word_index.get(i-3,'?') for i in review)

#2 function to preprocess user input
def preprocess_test(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=800)
    return padded_review

#3 prediction function
def predict_sentiment(review):
  preprocessed_input=preprocess_test(review)
  prediction=model.predict(preprocessed_input)[0][0]
  sentiment='positive' if prediction>0.5 else 'negative'
  return sentiment,prediction
  ##[0] [0] used because model return the prediction in 2D array format in which first[0] select the first review(first row) second[0] select the prediction score

## user input and prediction we pass example review for prediction
example_review='''
This movie is absolutely amazing and wonderful'''
sentiment,score=predict_sentiment(example_review)
print('Predicted Sentiment:',sentiment)
print('Prediction Score:',score)