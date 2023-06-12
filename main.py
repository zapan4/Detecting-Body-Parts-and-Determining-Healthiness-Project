from tqdm.notebook import tqdm

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, InputLayer
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape, Conv2D, GlobalAveragePooling2D
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import shutil
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

from hypopt import GridSearch

import cv2
from numpy import asarray
from PIL import Image
import glob

'''
X = np.zeros((2400, 4096))
Y = np.zeros(2400)
j = 0
for i in range(6):
    paths = ['Hand', 'HeadCT', 'AbdomenCT', 'ChestCT', 'CXR', 'BreastMRI']
    for filename in glob.glob('/Users/kids/PycharmProjects/research/archive/' + paths[i] + '/*.jpeg'):
        if j > 399+i*400:
            break
        im = Image.open(filename)
        data = asarray(im)
        data = data.flatten()
        X[j] = data
        Y[j] = i
        j += 1

X = X.reshape(-1, 64, 64, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

Y_train = keras.utils.to_categorical(Y_train, 6)
Y_test = keras.utils.to_categorical(Y_test, 6)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu', input_shape=(64, 64, 1)))
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(6, activation="softmax"))
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=16, epochs=6)

X_train = X_train.reshape(-1, 4096, 1)
X_test = X_test.reshape(-1, 4096, 1)

clf = LogisticRegression(random_state=0, max_iter=3000).fit(X_train, Y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(accuracy)

#first model
model1 = Sequential()
model1.add(InputLayer(input_shape=(4096,)))
model1.add(Dense(40, activation = 'relu'))
model1.add(Dense(40, activation = 'relu'))
model1.add(Dense(6, activation = 'softmax'))
model1.compile(loss='categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])
model1.fit(X_train, Y_train,
                    validation_data=(X_test, Y_test),
                    epochs=40)
y_pred1 = model1.predict(X_test)

#second model
model2 = Sequential()
model2.add(InputLayer(input_shape=(4096,)))
model2.add(Dense(40, activation = 'relu'))
model2.add(Dense(40, activation = 'relu'))
model2.add(Dense(40, activation = 'relu'))
model2.add(Dense(40, activation = 'relu'))
model2.add(Dense(6, activation = 'softmax'))
model2.compile(loss='categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])
model2.fit(X_train, Y_train,
                    validation_data=(X_test, Y_test),
                    epochs=50)
y_pred2 = model2.predict(X_test)

#third model
model3 = Sequential()
model3.add(InputLayer(input_shape=(4096,)))
model3.add(Dense(80, activation = 'relu'))
model3.add(Dense(80, activation = 'relu'))
model3.add(Dense(80, activation = 'relu'))
model3.add(Dense(80, activation = 'relu'))
model3.add(Dense(6, activation = 'softmax'))
model3.compile(loss='categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])
model3.fit(X_train, Y_train,
                    validation_data=(X_test, Y_test),
                    epochs=40)
y_pred3 = model3.predict(X_test)


weight1 = 1
weight2 = 1
weight3 = 1

number = 0

for i in range(600):
    weight1 = weight1/(pow(2.718, 1 - y_pred1[i][np.argmax(Y_test[i])]))
    weight2 = weight2/(pow(2.718, 1 - y_pred2[i][np.argmax(Y_test[i])]))
    weight3 = weight3/(pow(2.718, 1 - y_pred3[i][np.argmax(Y_test[i])]))
    sum = weight1 + weight2 + weight3
    weight1 = weight1 / sum
    weight2 = weight2 / sum
    weight3 = weight3 / sum
    weights = np.zeros(6)
    weights[np.argmax(y_pred1[i])] += weight1
    weights[np.argmax(y_pred2[i])] += weight2
    weights[np.argmax(y_pred3[i])] += weight3
    if np.argmax(weights) == np.argmax(Y_test[i]):
        number += 1

print(number/600)

y_pred6 = model2.predict(X_train)

blackbox = np.zeros(1800)
xnew1 = []
ynew1 = []
for i in range(1800):
    if np.argmax(y_pred6[i]) == np.argmax(Y_train[i]):
        blackbox[i] = 0
    else:
        xnew1.append(i)
        ynew1.append(i)
        blackbox[i] = 1

xnew = np.zeros((len(xnew1), 4096))
ynew = np.zeros((len(ynew1), 6))

for i in range(len(xnew1)):
    xnew[i] = X_train[xnew1[i]]
    ynew[i] = Y_train[ynew1[i]]

model4 = Sequential()
model4.add(InputLayer(input_shape=(4096,)))
model4.add(Dense(80, activation = 'relu'))
model4.add(Dense(80, activation = 'relu'))
model4.add(Dense(80, activation = 'relu'))
model4.add(Dense(80, activation = 'relu'))
model4.add(Dense(6, activation = 'softmax'))
model4.compile(loss='categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

model4.fit(xnew, ynew,
                    epochs=40)

y_pred4 = model4.predict(X_test)

blackbox = keras.utils.to_categorical(blackbox, 2)

model5 = Sequential()
model5.add(InputLayer(input_shape=(4096,)))
model5.add(Dense(80, activation = 'relu'))
model5.add(Dense(80, activation = 'relu'))
model5.add(Dense(80, activation = 'relu'))
model5.add(Dense(80, activation = 'relu'))
model5.add(Dense(2, activation = 'softmax'))
model5.compile(loss='categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

model5.fit(X_train, blackbox, epochs=40)

y_pred5 = model5.predict(X_test)
print(y_pred5)

number1 = 0

for i in range(600):
    if np.argmax(y_pred5[i]) == 0:
        if np.argmax(y_pred1[i]) == np.argmax(Y_test[i]):
            number1 = number1 + 1
    else:
        if np.argmax(y_pred4[i]) == np.argmax(Y_test[i]):
            number1 = number1 + 1

print(number1/600)
'''

X_train = np.zeros((5216, 4096))
Y_train = np.zeros(5216)

X_test = np.zeros((624, 4096))
Y_test = np.zeros(624)

i = 0
for filename in glob.glob('/Users/kids/PycharmProjects/research/chest_xray/chest_xray/train/NORMAL/*.jpeg'):
    if i > 1340:
        break
    im = Image.open(filename)
    im = im.resize((64, 64))
    data = asarray(im)
    data = data.flatten()
    X_train[i] = data
    Y_train[i] = 0
    i += 1

for filename in glob.glob('/Users/kids/PycharmProjects/research/chest_xray/chest_xray/train/PNEUMONIA/*.jpeg'):
    if i > 5215:
        break
    im = Image.open(filename)
    im = im.resize((64, 64))
    data = asarray(im)
    data = data.flatten()
    if data.shape[0] != 4096:
        continue
    X_train[i] = data
    Y_train[i] = 1
    i += 1

i = 0

for filename in glob.glob('/Users/kids/PycharmProjects/research/chest_xray/chest_xray/test/NORMAL/*.jpeg'):
    if i > 233:
        break
    im = Image.open(filename)
    im = im.resize((64, 64))
    data = asarray(im)
    data = data.flatten()
    X_test[i] = data
    Y_test[i] = 0
    i += 1

for filename in glob.glob('/Users/kids/PycharmProjects/research/chest_xray/chest_xray/test/PNEUMONIA/*.jpeg'):
    if i > 623:
        break
    im = Image.open(filename)
    im = im.resize((64, 64))
    data = asarray(im)
    data = data.flatten()
    X_test[i] = data
    Y_test[i] = 1
    i += 1

Y_train = keras.utils.to_categorical(Y_train, 2)
Y_test = keras.utils.to_categorical(Y_test, 2)


X_train = X_train.reshape(-1, 64, 64, 1)
X_test = X_test.reshape(-1, 64, 64, 1)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu', input_shape=(64, 64, 1)))
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=16, epochs=6)


