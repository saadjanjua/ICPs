import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("Breas Cancer.csv")
dataset.drop(dataset.columns[32], axis=1, inplace=True)
print(dataset.head())
print(dataset.shape)

# Nulls after last step
nulls = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

dataset['diagnosis'] = dataset['diagnosis'].map(
    {'B': 0, 'M': 1}).astype(int)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x = dataset.drop(['diagnosis', 'id'], axis=1)
sc.fit(x)
x_scaler = sc.transform(x)
x_scaled = pd.DataFrame(x_scaler, columns=x.columns)
y = dataset['diagnosis'].values

X_train, X_test, Y_train, Y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=87)
#X_train, X_test, Y_train, Y_test = train_test_split(x, y,
#                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(15, input_dim=30, activation='relu')) # hidden layer
#my_first_nn.add(Dense(10, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test))