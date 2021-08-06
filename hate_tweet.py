

import numpy as np # pip install numpy
import pandas as pd # pip install pandas
import random

import tensorflow as tf #pip install tensorflow==2.2
#os.environ["PATH"] += os.pathsep + 'C:/Programmi/Graphviz/bin/'
import os

from keras import backend as K


'''
# Get the same result after each execution
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)
os.environ['PYTHONHASHSEED'] = '0'
session_conf = tf.compat.v1.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)
'''


### DATASET OPENING
df = pd.read_csv('train_hate.csv', encoding='utf-8')
print(df['tweet'])

# Take only the usefull columns
df = df[['tweet', 'label']]
print(df)



#### STUDY THE DATASET ####
total_rows = df.shape[0]
print("Total Rows: ", total_rows)

# Count the number of labels
list_of_labels = df['label'].unique()
print("Labels: ", list_of_labels)
print("Number of Labels: ", len(df['label'].unique()))


# Print the number of records for each labels
for i in list_of_labels:
    n = len(df[df["label"] == i])
    print("Label: [",i,"] Records:", n , "/",total_rows )


#### TEXT CLEANING ####
import nltk # pip install nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
import re
import string

def clean_text(text):  
    
  # delete all links
  text = re.sub(r"http\S+", "", text) 
  text = re.sub(r"html\S+", "", text) 
  text = re.sub(r"https\S+", "", text) 
  
  # Delete all punctuation
  text = re.sub('[^A-Za-z0-9]+', ' ', text)
  
  # Delete all stopword
  text = text.lower().split()
  stops = set(stopwords.words("english"))
  text = [w for w in text if not w in stops]
  
  
  text = " ".join(text)

  return text

print(df['tweet'])

df['Text_Clean'] = df['tweet'].map(lambda x: clean_text(x))

row = 100 



print("\n\nOriginal Text")
print(df['tweet'].iloc[row])
print("\n\nCleaned Text")
print(df['Text_Clean'].iloc[row])


######### FEATURE SELECTION

# Numbers of word inside a tweet
df['Feature_1'] = df['tweet'].apply(lambda x: len(str(x).split()))

# Numbers of letters inside a tweet
df['Feature_2'] = df['tweet'].apply(lambda x: len(str(x)))

# Mean length of word
df['Feature_3'] = df['Feature_2'] / df['Feature_1']

# Numbers of stopword inside the tweet
stop_words = set(stopwords.words('english'))
df['Feature_4'] = df["tweet"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))


# Numbers of punctuation inside the tweet
df['Feature_5'] = df['tweet'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]) )


# WORK WITH THE TEXT

from sklearn.utils import shuffle #pip install sklearn
from keras.preprocessing.text import Tokenizer #pip install keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

df = shuffle(df)

# We will work only with the cleaned Text, Labels and calculated features
# so let's drop the original Text

df = df.drop(columns=['tweet'])


# Creating a list that contain all the tweet of the dataset
lista_testo = df["Text_Clean"].fillna('').to_list() 


# Cast every value to String
lista_testo = [str(i) for i in lista_testo] 


# Initialize the Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lista_testo)

vocab_size = len(tokenizer.word_index) + 1
print("Word inside the vocabulary ", vocab_size) 
lista_testo_tokenizer = tokenizer.texts_to_sequences(lista_testo)
max_len = max(len(x) for x in lista_testo_tokenizer)


# max_len rappresent the number of words inside the longest tweet inside the dataset
print(max_len)

df['testo_token'] = tokenizer.texts_to_sequences(df['Text_Clean'])
print("Length before post padding: ", len(df['testo_token'].iloc[1]))
#i want all the world to be at the same length, so let's add zero padding.
df['testo_token_padding'] = pad_sequences(df['testo_token'], padding = "post", maxlen = max_len).tolist()
print("Length after post padding: ", len(df['testo_token_padding'].iloc[1]))




# let's filter the columns that are usefull for training the model.
df = df[['testo_token_padding','Feature_1','Feature_2','Feature_3','Feature_4',
         'Feature_5', 'label']]


#Dataframe values
X = df.iloc[:,0:6].values
Y = df.iloc[:,6].values



# Splitting phase, 80% to training, 20% to testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)



X_train_embedding = np.array([item[0] for item in X_train]).astype(np.float32)

X_train_feature = np.array([item[1:] for item in X_train]).astype(np.float32)

#Let's do the same thing for the Test Set
X_test_embedding = np.array([item[0] for item in X_test]).astype(np.float32)
X_test_feature = np.array([item[1:] for item in X_test]).astype(np.float32)


# class 1 became [0 1] and class 0 became [1 0]. it's the format required by Keras
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2) # there are 2 classes


## MODEL CREATION

from keras.layers import Flatten, Input, Concatenate, Dense
from keras.utils.vis_utils import plot_model
from keras.layers.embeddings import Embedding
from keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix

embedding_dim = max_len #is 41 for this problem


input_testo = Input(shape=(max_len,))

x = Embedding(vocab_size, embedding_dim, input_length = max_len, trainable = True)(input_testo)
x = Flatten()(x)

input_feature = Input(shape=(X_train_feature.shape[1],))

model_final = Concatenate()([x, input_feature])
model_final = Dense(300, activation='relu', bias_initializer='zeros')(model_final)
model_final = Dense(150, activation = "relu", bias_initializer='zeros') (model_final)
model_final = Dense(2, activation='softmax', bias_initializer='zeros')(model_final)
model_final = Model([input_testo,input_feature], model_final)
model_final.compile(loss="categorical_crossentropy", optimizer = 'adam', metrics = ["accuracy"])


#pip install pydot
#pip install pydotplus
#pip install graphviz
plot_model(model_final, to_file="model_plot.png", show_shapes = True, show_layer_names = True)




import matplotlib.pyplot as plt #pip install matplotlib
#Function to sshow the performance of the model during the training.
def plot_history(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5), dpi = 130)
    
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Accuracy during training and validation')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Loss during training and validation')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()




history = model_final.fit(x=[X_train_embedding,X_train_feature], y = np.array(y_train),
                          batch_size = 64, epochs=3, verbose = 1, validation_split=0.2)


# TESTING

y_pred = model_final.predict([X_test_embedding, X_test_feature])
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
print(cm)
plot_history(history)




