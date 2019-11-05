
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import os
import time
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from cv2 import cv2
NAME = "BENIGNMALIGNANTTUMOR-{}".format(int(time.time()))


tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

NewX = pickle.load(open('X.pickle', 'rb'))
Newy = pickle.load(open('y.pickle', 'rb'))

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)


NewX = NewX/255.0

model = Sequential()
model.add(Conv2D((64), (3,3),input_shape=NewX.shape[1:]))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D((64), (3,3)))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.save_weights(checkpoint_path.format(epoch=0))

model.fit(NewX, Newy, batch_size=32, epochs=10, validation_split=0.1, callbacks=[cp_callback], verbose=0)

model.save('mynewthing.h5')

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

prediction = model.predict([prepare("test9.png")])
print(int(prediction[0][0]))

# model.load_weights(checkpoint_path)
# loss, acc = model.evaluate(NewX, Newy)
# print("Restored Model, accuracy: {:5.2f}%".format(100*acc))
