import pandas as pd
import os
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.datasets import mnist

save_path = "."

# Load the MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the images and one-hot encode the labels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Build the neural network
model = Sequential()
model.add(Dense(392, input_dim = x_train.shape[1], activation='relu')) # Hidden 1
model.add(Dense(196, activation='relu')) # Hidden 2
model.add(Dense(10, activation='softmax')) # Output
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# Use Early Stopping to avoid overfitting
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto', restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=2, callbacks=[monitor], epochs=100)

# Evaluate the model's performance
predictions = model.predict(x_test)
predict_classes = np.argmax(predictions, axis=1)
expected_classes = np.argmax(y_test, axis=1)
correct = accuracy_score(expected_classes,predict_classes)
print(f"Accuracy: {correct}")

# save neural network structure to JSON (no weights)
model_json = model.to_json()
with open(os.path.join(save_path,"mnist-digit-network.json"), "w") as json_file: json_file.write(model_json)

# save entire network to HDF5 (save everything, suggested)
model.save(os.path.join(save_path,"mnist-digit-network.h5"))

#Dump variables into a pickle file to load later
with open("variables.pkl", "wb") as f:
    pickle.dump(x_test, f)
    pickle.dump(y_test, f)
    pickle.dump(x_train, f) 
    pickle.dump(y_train, f)
