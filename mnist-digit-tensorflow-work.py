import os
import numpy as np
import pickle
import subprocess
import sys
from sklearn.metrics import accuracy_score
from keras.models import load_model
from matplotlib import pyplot as plt

#The goal here is to adopt the digit recognition algorithim developed by Smason Zhang
#and adopt it to utilize Tensorflow in hopes of achieving greater accuracy. This will use 
#a Classification NN

save_path = "."

if os.path.exists('./mnist-digit-network.keras'):
    #load variables with pickle
    with open("variables.pkl", "rb") as f:
        x_test = pickle.load(f)
        y_test = pickle.load(f)
        x_train = pickle.load(f)
        y_train = pickle.load(f)

    model = load_model(os.path.join(save_path,"mnist-digit-network.h5"))
    pred = model.predict(x_test)
    print(pred)
    predict_classes = np.argmax(pred,axis=1)
    expected_classes = np.argmax(y_test,axis=1)
    correct = accuracy_score(expected_classes,predict_classes)

    # Select 10 random images to predict
    num_samples = 10
    random_indices = np.random.choice(x_test.shape[0], num_samples)
    random_samples = x_test[random_indices]

    pred = model.predict(random_samples)
    predict_classes = np.argmax(pred, axis=1)

    # Sample the entire sheet
    for i in range(num_samples):
        sample_digit = random_samples[i]
        pred = model.predict([sample_digit.reshape(1, -1)])
        print(pred)
        pred = np.argmax(pred,axis=1)
        print(f"Predict that this image {expected_classes[random_indices[i]]} ")
        print(f"is: {predict_classes[i]}")
        current_image = sample_digit.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

    print(f"Accuracy: {correct}")

else:
    print("Now building model. This may take several minutes: ")
    result = subprocess.run(["python", "./mnist-digit-tensorflow-build_model"])

    if result.returncode == 0:
        print("Script ran successfully")
    os.execl(sys.executable, sys.executable, *sys.argv)
