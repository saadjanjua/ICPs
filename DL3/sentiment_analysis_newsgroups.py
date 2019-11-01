from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups
newsgroups_train =fetch_20newsgroups(subset='train', shuffle=True)
sentences = newsgroups_train.data
y = newsgroups_train.target

# tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
# getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)


# Embedding
#max_review_len = max([len(s.split()) for s in sentences])
#vocab_size = len(tokenizer.word_index) + 1
#sentences = tokenizer.texts_to_sequences(sentences)
#padded_docs = sequence.pad_sequences(sentences, maxlen=max_review_len)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)


# Number of features
# print(input_dim)
model = Sequential()
# Add Embedding
#model.add(Embedding(vocab_size, 100, input_length=max_review_len))
#model.add(Flatten())
# corrected input_dim
model.add(layers.Dense(300, input_dim=2000, activation='relu'))
# change the activation to softmax
model.add(layers.Dense(20, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test), batch_size=256)

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
print(model.predict_classes(X_test[[28],:]))
print(X_test[[28],:])