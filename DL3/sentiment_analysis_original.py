from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.callbacks import TensorBoard
import time
df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values


#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# Number of features
# print(input_dim)
model = Sequential()

#correct input_dim
model.add(layers.Dense(300,input_dim=2000, activation='relu'))
model.add(layers.Dense(300,input_dim=2000, activation='tanh'))
model.add(layers.Dense(300,input_dim=2000, activation='sigmoid'))
#change the output to softmax
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)


[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
print(history.history.keys())

plt.figure(1)
plt.subplots_adjust(hspace=.5)
plt.subplot(221)
plt.plot(history.history['val_loss'])
plt.title('Val Loss')
plt.subplot(222)
plt.plot(history.history['loss'])
plt.title('Loss')
plt.subplot(223)
plt.plot(history.history['val_acc'])
plt.title('Val Accuracy')
plt.subplot(224)
plt.plot(history.history['acc'])
plt.title('Accuracy')

plt.show()