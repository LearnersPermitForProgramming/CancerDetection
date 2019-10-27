
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

NewX = pickle.load(open('X.pickle', 'rb'))
Newy = pickle.load(open('y.pickle', 'rb'))

NewX = NewX/255.0

model = Sequential()
model.add(Conv2D((64), (3,3),input_shape=NewX.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D((64), (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(NewX, Newy, batch_size=32, epochs=10, validation_split=0.1)