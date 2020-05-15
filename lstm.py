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

##lstm model
embed=128
latent_dim=256
model = Sequential()
model.add(Embedding(X_voc, embed,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(latent_dim, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

model.fit(X_train, y_train, epochs = 20, batch_size=128, verbose = 1)

y_pred = model.predict_classes(X_test,batch_size = 128)
df_test = pd.DataFrame({'true': y_test, 'pred':y_pred})
df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))
cm=confusion_matrix(df_test.true, df_test.pred)
print(cm)
print(classification_report(df_test.true, df_test.pred))

twt = ['keep up the good work']
twt = X_tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")