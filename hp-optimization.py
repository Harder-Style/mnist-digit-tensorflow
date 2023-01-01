import tensorflow as tf
import kerastuner as kt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the inputs
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the model
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    
    # Tune the number of units in the first dense layer
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Use the tuner to search for the best hyperparameters
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model on the test set
_, test_acc = best_model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Get the summary of the results
best_hps = tuner.get_best_hyperparameters()[0]
print(f'Best hyperparameters: {best_hps}')

results = tuner.results_summary()
print(results)
