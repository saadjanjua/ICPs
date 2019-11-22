import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.models import load_model
import numpy as np

from sklearn.preprocessing import LabelEncoder

model = load_model('model.h5')
test_text = [["A lot of good things are happening. We are respected again throughout the world, and that's a great thing."]]
# test_text = [["I am the best"]]
test_df = pd.DataFrame(test_text)
print(test_df.values)
test_df[0] = test_df[0].apply(lambda x: x.lower())
test_df[0] = test_df[0].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(test_df[0].values)
X = tokenizer.texts_to_sequences(test_df[0].values)

X = pad_sequences(X, maxlen=28)
#test_df['text'] =
print(np.argmax(model.predict(X)))