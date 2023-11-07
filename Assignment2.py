# Implementing Feedforward neural networks with Keras and TensorFlow
# a. Import the necessary packages
# b. Load the training and testing data (MNIST/CIFAR10)
# c. Define the network architecture using Keras
# d. Train the model using SGD
# e. Evaluate the network
# f. Plot the training loss and accuracy

# Step 1: Import necessary packages
import numpy as np # Imports the NumPy library as 'np', which is commonly used for numerical operations and array handling in Python.
import tensorflow as tf # Imports TensorFlow, a popular deep learning framework.
from keras.datasets import cifar10 # Imports the CIFAR-10 dataset from Keras, as mentioned earlier.
from tensorflow.keras.models import Sequential # Imports the Sequential model from TensorFlow's Keras API. The Sequential model is a linear stack of layers, which is commonly used for building feedforward neural networks.
from tensorflow.keras.layers import Dense, Flatten # Imports the Dense and Flatten layers from TensorFlow's Keras API. The Dense layer represents a fully connected layer in a neural network, and the Flatten layer is used to flatten the input data. 
import matplotlib.pyplot as plt # Imports the Matplotlib library for creating plots and visualizations.

# Step 2: Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data() # Loads the CIFAR-10 dataset and splits it into training and testing sets. x_train contains the training images, y_train contains the training labels, x_test contains the testing images, and y_test contains the testing labels.
x_train = x_train.astype('float32') / 255 #  Normalizes the pixel values in the training and testing images by converting them to float32 and dividing by 255. This scales the pixel values to the range [0, 1].
x_test = x_test.astype('float32') / 255 #  Normalizes the pixel values in the training and testing images by converting them to float32 and dividing by 255. This scales the pixel values to the range [0, 1].

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10) # One-hot encodes the training and testing labels using the to_categorical function from TensorFlow's Keras utilities. This converts class labels into binary vectors for each class, making them suitable for training a neural network.
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10) # One-hot encodes the training and testing labels using the to_categorical function from TensorFlow's Keras utilities. This converts class labels into binary vectors for each class, making them suitable for training a neural network.

# Step 3: Define the network architecture # This is a comment indicating the beginning of the third step, which is about defining the network architecture. 
model = Sequential([            # Defines a sequential model. The layers are defined inside a list that is passed to the Sequential constructor. The model consists of:
    Flatten(input_shape=(32, 32, 3)),  # Adjust input shape for CIFAR-10 # A Flatten layer that converts the 32x32x3 input images into a 1D array.
    Dense(64, activation='relu'), # A fully connected (Dense) hidden layer with 64 neurons and ReLU activation function.
    Dense(64, activation='relu'), # Another fully connected hidden layer with 64 neurons and ReLU activation.
    Dense(10, activation='softmax') #  The output layer with 10 neurons (matching the number of classes in CIFAR-10) and a softmax activation function for classification.
])

# Step 4: Compile the model # This is a comment indicating the beginning of the fourth step, which is about compiling the model.
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) # Compiles the model with the specified settings:
# optimizer='sgd': Uses stochastic gradient descent (SGD) as the optimization algorithm.
# loss='categorical_crossentropy': Sets the loss function to categorical cross-entropy, which is commonly used for multi-class classification tasks.
# metrics=['accuracy']: Evaluates the model's performance using accuracy as the metric.

# Step 5: Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test)) # Trains the model on the training data for 10 epochs with a batch size of 32. The training progress and results are stored in the history variable for later analysis.

# Step 6: Evaluate the network
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0) # Evaluates the model's performance on the testing data and stores the test loss and test accuracy. The verbose=0 parameter means that the evaluation is performed without printing progress information.
print(f'Test accuracy: {test_accuracy*100:.2f}%') # Prints the test accuracy as a percentage.

# Step 7: Plot the training loss and accuracy
plt.figure(figsize=(10, 4)) # 

# Plot training & validation accuracy values
plt.subplot(1, 2, 1) # Sets up the first subplot in a 1x2 grid.
plt.plot(history.history['accuracy']) #  Plots the training accuracy values from the history object.
plt.plot(history.history['val_accuracy']) # Plots the validation accuracy values from the history object.
plt.title('Model accuracy') # 
plt.ylabel('Accuracy') # 
plt.xlabel('Epoch') # 
plt.legend(['Train', 'Test'], loc='upper left') #   

# Plot training & validation loss values
plt.subplot(1, 2, 2) # Sets up the second subplot in a 1x2 grid.
plt.plot(history.history['loss']) # Plots the training loss values from the history object.
plt.plot(history.history['val_loss']) # Plots the validation loss values from the history object.
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout() # Adjusts the subplot layout for better spacing.
plt.show()

# What is the purpose of this code?
# The purpose of this code is to train a neural network for image classification on the CIFAR-10 dataset and evaluate its performance.

# What dataset is being used in this code, and why is it commonly used in deep learning?
# The code uses the CIFAR-10 dataset, which is commonly used in deep learning for image classification tasks due to its variety of objects and classes.

# How is the CIFAR-10 dataset preprocessed in this code, and why is preprocessing important?
# The CIFAR-10 dataset is preprocessed by normalizing pixel values to the range [0, 1]. Preprocessing is essential to prepare data for neural network training and to improve convergence.

# Can you explain the structure of the neural network model defined in this code?
# The model is a feedforward neural network with three layers: input (flattened for images), two hidden layers with ReLU activation, and an output layer with softmax for classification.

# What is the role of the compile method, and what parameters are specified during compilation?
# The compile method configures the model for training by specifying the optimizer (SGD), loss function (categorical cross-entropy), and accuracy as the evaluation metric.

# How is the model trained, and what information is stored in the history variable?
# The model is trained on the training data for 10 epochs with a batch size of 32, and training history, including accuracy and loss, is stored in the history variable.

# How is the model's performance evaluated, and what metric is used for evaluation?
# The model's performance is evaluated on the test data, and test accuracy is used as the evaluation metric.

# What do the accuracy and loss plots in the final step of the code represent?
# The accuracy plots show how well the model is learning, and the loss plots indicate how well the model is minimizing its error during training.

# Can you describe the significance of one-hot encoding the labels in the CIFAR-10 dataset?
# One-hot encoding transforms class labels into a binary format, allowing the model to understand class relationships and perform multi-class classification.

# What could be potential improvements or modifications to this code for better model performance?
# Potential improvements may include adding more complex layers, adjusting hyperparameters, data augmentation, and implementing regularization techniques for better model performance.