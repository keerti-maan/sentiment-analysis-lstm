import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
import re

data = pd.read_csv('Sentiment.csv',usecols=['text','sentiment'])
data = data[data.sentiment != "Neutral"]

#cleaning data
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
data['text'] = data['text'].str.replace('rt','')

#preprocessing
X=np.array(data['text'])
y=pd.get_dummies(data['sentiment']).values
X_tokenizer = Tokenizer(num_words=2000,split=' ') 
X_tokenizer.fit_on_texts(list(X))
X = X_tokenizer.texts_to_sequences(X) 
#padding zero upto maximum length
X = pad_sequences(X)
#size of vocabulary ( +1 for padding token)
X_voc   =  X_tokenizer.num_words + 1

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0,shuffle=True)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
