#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
#loading the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#plotting the first image in the dataset
X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('float32')
#normalizing inputs from 0-255 to 0-1
X_train /= 255
X_test /= 255
#one hot encoding outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#defining the model
model = Sequential()
#adding layers to the model
model.add(Dense(512, input_shape=(28 * 28,), activation='relu'))
#reducing the overfitting of values
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
#compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#fitting the model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))
#evaluating the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])
#predicting the image in the test set
predictions = model.predict(X_test)
#plotting the image in the test set
plt.plot(predictions[0], color='green')
plt.title('Prediction for first test image')
plt.xlabel('Digit')
plt.ylabel('Probability')
plt.show()
print(mnist)