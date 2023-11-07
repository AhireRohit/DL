# Study of Deep learning Packages: Tensorflow, Keras, Theano and PyTorch. Document the distinct
# features and functionality of the packages

from keras.datasets import cifar10 # This line imports the CIFAR-10 dataset from Keras, which is a popular dataset for image classification tasks. CIFAR-10 contains 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

from matplotlib import pyplot # Imports the pyplot module from the Matplotlib library, which is used for creating plots and displaying images.

(train_X, train_y), (test_X, test_y) = cifar10.load_data() # This line loads the CIFAR-10 dataset and splits it into training and testing sets. train_X contains the training images, train_y contains the corresponding training labels, test_X contains the testing images, and test_y contains the testing labels.

# shape of dataset
print('X_train: ' + str(train_X.shape)) # This line prints the shape of the training data, which is the dimensions of the train_X array. It shows the number of training examples, image height, image width, and the number of color channels (3 for RGB).

print('Y_train: ' + str(train_y.shape)) # This line prints the shape of the training labels, which indicates the number of training examples.

print('X_test: ' + str(test_X.shape)) # This line prints the shape of the testing data, similar to what was done for the training data.

print('Y_test: ' + str(test_y.shape)) # This line prints the shape of the testing labels.

# plotting
from matplotlib import pyplot 

for i in range(9):  # This line starts a loop that iterates 9 times, for the purpose of displaying the first 9 images from the training dataset.
    pyplot.subplot(330 + 1 + i) # This line sets up a subplot grid with 3 rows and 3 columns (3x3 grid) and selects the i-th subplot for displaying the image. The i variable iterates from 0 to 8, so this code selects the first 9 subplots in the grid.

    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray')) # This line displays the i-th image from the training dataset using Matplotlib's imshow function. It specifies the colormap as 'gray', which is used to display a color image in grayscale. The train_X[i] is the image to be displayed.
pyplot.show() # This line shows the entire grid of subplots with the 9 images. The previous code has set up the subplots and displayed the images within them, and this command is responsible for showing the grid of subplots to the user.

# Output --> X_train: (50000, 32, 32, 3)
#            Y_train: (50000, 1)
#            X_test: (10000, 32, 32, 3)
#            Y_test: (10000, 1)

# X_train: (50000, 32, 32, 3): This line indicates that the training data (X_train) has a shape of (50000, 32, 32, 3). 
# This means:
# 50000: There are 50,000 training examples in the dataset.
# 32: Each image in the training set has a height of 32 pixels.
# 32: Each image in the training set has a width of 32 pixels.
# 3: There are 3 color channels per pixel, indicating that the images are in RGB (Red, Green, Blue) format.

# Y_train: (50000, 1): This line indicates that the training labels (Y_train) have a shape of (50000, 1). 
# This means:
# 50000: There are 50,000 corresponding labels for the training examples.
# 1: Each label is a single scalar value, indicating the class of the corresponding image. In this context, CIFAR-10 is a classification dataset with 10 different classes (e.g., "airplane," "automobile," "bird," etc.), and each label represents one of these classes.

# X_test: (10000, 32, 32, 3): This line indicates that the testing data (X_test) has a shape of (10000, 32, 32, 3). This is similar to the training data but corresponds to the testing set, which contains different examples.

# Y_test: (10000, 1): This line indicates that the testing labels (Y_test) have a shape of (10000, 1). Like the training labels, there are 10,000 labels for the testing examples, with each label representing the class of the corresponding image.

# In summary, the output provides information about the dimensions and shapes of the training and testing data and labels in the CIFAR-10 dataset. The training set has 50,000 images, while the testing set has 10,000 images, and each image is a 32x32 pixel RGB image. The labels for both sets are represented as single scalar values.

# Output -->
# Epoch 1/10
# 1563/1563 [==============================] - 4s 3ms/step - loss: 1.9228 - accuracy: 0.3078 - val_loss: 1.7984 - val_accuracy: 0.3530
# Epoch 2/10
# 1563/1563 [==============================] - 3s 2ms/step - loss: 1.7417 - accuracy: 0.3785 - val_loss: 1.9830 - val_accuracy: 0.3025
# Epoch 3/10
# 1563/1563 [==============================] - 3s 2ms/step - loss: 1.6679 - accuracy: 0.4083 - val_loss: 1.6330 - val_accuracy: 0.4248
# Epoch 4/10
# 1563/1563 [==============================] - 3s 2ms/step - loss: 1.6193 - accuracy: 0.4265 - val_loss: 1.6944 - val_accuracy: 0.3952
# Epoch 5/10
# 1563/1563 [==============================] - 3s 2ms/step - loss: 1.5833 - accuracy: 0.4385 - val_loss: 1.6090 - val_accuracy: 0.4266
# Epoch 6/10
# 1563/1563 [==============================] - 3s 2ms/step - loss: 1.5509 - accuracy: 0.4505 - val_loss: 1.5582 - val_accuracy: 0.4431
# Epoch 7/10
# 1563/1563 [==============================] - 3s 2ms/step - loss: 1.5225 - accuracy: 0.4613 - val_loss: 1.5895 - val_accuracy: 0.4308
# Epoch 8/10
# 1563/1563 [==============================] - 3s 2ms/step - loss: 1.4985 - accuracy: 0.4696 - val_loss: 1.5364 - val_accuracy: 0.4523
# Epoch 9/10
# 1563/1563 [==============================] - 3s 2ms/step - loss: 1.4751 - accuracy: 0.4780 - val_loss: 1.4951 - val_accuracy: 0.4661
# Epoch 10/10
# 1563/1563 [==============================] - 4s 2ms/step - loss: 1.4571 - accuracy: 0.4834 - val_loss: 1.5105 - val_accuracy: 0.4660
# Test accuracy: 46.60%

# The output you provided is from training a neural network on the CIFAR-10 dataset. Here's an explanation of the key parts of the output:

# Optimization Information:

# The output begins with a message related to TensorFlow's CPU feature optimization, indicating that the binary is optimized to use certain CPU instructions for performance-critical operations.
# Training Progress:

# It shows the progress of training the neural network over 10 epochs. Each epoch represents one complete pass through the entire training dataset.

# For each epoch, you can see information such as:

# Epoch X/10: Indicates the current epoch out of the total 10 epochs.
# loss: The training loss, which measures how well the model is fitting the training data. It starts relatively high and decreases over time.
# accuracy: The training accuracy, which represents the proportion of correctly classified training examples. It starts low and increases.
# Similarly, there's validation data performance:

# val_loss: The validation loss, which measures how well the model generalizes to data it hasn't seen during training. It may vary compared to the training loss.
# val_accuracy: The validation accuracy, which represents the accuracy on a separate validation dataset. It is used to monitor how well the model generalizes to new data.
# Test Accuracy:

# After training for 10 epochs, the code evaluates the model on the test dataset and prints the test accuracy, which represents how well the model performs on unseen data. In this case, the test accuracy is 46.60%, meaning the model correctly classifies 46.60% of the test examples.
# In summary, the output provides insight into the training and evaluation process of the neural network on the CIFAR-10 dataset, including information about loss and accuracy over multiple epochs and the final test accuracy. The goal is typically to train the model until the training and validation accuracies converge while avoiding overfitting to achieve good performance on unseen data.

# Training Progress:

# Epoch 1/10: This line indicates the start of the first training epoch out of a total of 10 epochs. Each epoch represents one pass through the entire training dataset.

# 1563/1563 [==============================] - 4s 3ms/step - loss: 1.9228 - accuracy: 0.3078 - val_loss: 1.7984 - val_accuracy: 0.3530: This line represents the progress and results of the first epoch. Here's a breakdown:

# 1563/1563: Indicates the batch progress. In this case, 1563 batches have been processed out of a total of 1563.

# [==============================]: The progress bar, which is filled as the training progresses through batches.

# - 4s 3ms/step: Indicates that the first epoch took 4 seconds to complete and processed batches at a rate of 3 milliseconds per batch.

# loss: 1.9228: The training loss at the end of the first epoch. This value indicates how well the model is fitting the training data. It's relatively high at the start and will hopefully decrease with more training.

# accuracy: 0.3078: The training accuracy at the end of the first epoch. This value represents the proportion of correctly classified training examples. It starts at 30.78%.

# val_loss: 1.7984: The validation loss at the end of the first epoch. This value measures how well the model generalizes to data it hasn't seen during training. It's important to monitor to detect overfitting.

# val_accuracy: 0.3530: The validation accuracy at the end of the first epoch. This value represents the accuracy on a separate validation dataset, which helps monitor how well the model generalizes to new data. It's initially at 35.30%.

# Repeating for Subsequent Epochs:

# The same format is repeated for the remaining epochs (2/10 through 10/10). Training and validation loss and accuracy are shown for each epoch, allowing you to see how the model's performance changes over time.
# Test Accuracy:

# Test accuracy: 46.60%: After training for 10 epochs, the code evaluates the model on the test dataset and prints the final test accuracy. The test accuracy represents how well the model performs on unseen data, and in this case, it's 46.60%.