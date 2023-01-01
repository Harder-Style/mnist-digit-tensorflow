import pandas as pd
import os
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping

#The goal here is to adopt the digit recognition algorithim developed by Smason Zhang
#and adopt it to utilize Tensorflow in hopes of achieving greater accuracy. This will use 
#a Classification NN

save_path = "."

df = pd.read_csv("./mnist_train_small.csv", na_values=['NA', '?'])

# Convert to numpy - Classification
data=np.array(df)
m,n = data.shape
np.random.shuffle(data)

print(m,n) #19999, 785
print(data) #label, [binary representation of the text]

x = data[:, 1:]
dummies = pd.get_dummies(data[:,0]) # Classification
label = dummies.columns
y = dummies.values
print(x)
print(y)

# Split into validation and training sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    
# Build neural network
model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], activation='relu')) # Hidden 1
model.add(Dense(25, activation='relu')) # Hidden 2
model.add(Dense(10, input_dim=y.shape[1], activation='softmax')) # Output
model.compile(loss='categorical_crossentropy', optimizer='adam')

#establish Early Stopping to avoid 'overfitting'
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto', restore_best_weights=True)
model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=2, callbacks=[monitor], epochs=1000)

pred = model.predict(x_test)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y_test,axis=1)
correct = accuracy_score(expected_classes,predict_classes)
print(f"Accuracy: {correct}")
print(f"Possible Characters: {label}")

# save neural network structure to JSON (no weights)
model_json = model.to_json()
with open(os.path.join(save_path,"mnist-digit-network.json"), "w") as json_file: json_file.write(model_json)

# save entire network to HDF5 (save everything, suggested)
model.save(os.path.join(save_path,"mnist-digit-network.h5"))

#Dump variables into a pickle file to load later
with open("variables.pkl", "wb") as f:
    pickle.dump(x, f)
    pickle.dump(y, f)
    pickle.dump(x_test, f)
    pickle.dump(y_test, f)
    pickle.dump(x_train, f) 
    pickle.dump(y_train, f)
    pickle.dump(label, f)
