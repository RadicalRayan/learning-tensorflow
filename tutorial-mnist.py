from __future__ import absolute_import, division, print_function, unicode_literals

# TF and Keras, a high level API for TF
import tensorflow as tf
from tensorflow import keras

# other helpful libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# ----- Import and Load Data -----

fashion_mnist = keras.datasets.fashion_mnist
# the data has already been separated into training and test data with appropriate labels
# these variables hold the corresponding data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class name options for each of the data images for respective index
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# dimensions of matrix of input images
print(train_images.shape)  # (60000, 28, 28)
# number of labels (one for each image)
print(len(train_labels))   # 60000
# each label is an integer between 0 and 9
print(train_labels)

# shape and number of labels for test data
print(test_images.shape)  # (10000, 28, 28)
print(len(test_labels))   # 10000

# ----- Pre-process the Data -----

# shows the first training image. notice the pixel values are between 0 and 255
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.set_cmap(cmap=plt.cm.binary)  # comment out this line to see values in color
plt.grid(False)
plt.show()

# the data must be scaled to be from 0 to 1 before being fed to the model
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# ----- Build the Model -----

# set up the layers -> deep learning consists of chaining together layers
model = keras.Sequential([
    # transforms the 2d array of 28x28 into a 1d array of 784 pixels
    keras.layers.Flatten(input_shape=(28, 28)),
    # densely-connected/fully-connected neural layer of 128 nodes. Rectified linear activation
    keras.layers.Dense(128, activation=tf.nn.relu),
    # 10 node softmax -> produces array of 10 probabilities (likelihood for each class) that sum to 1
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile the model
model.compile(optimizer='adam',  # how the model is updates based on loss function
              loss='sparse_categorical_crossentropy',  # measures accuracy and model tries to minimize loss
              metrics=['accuracy'])  # used to monitor training. accuracy = fraction of images correctly classified

# ----- Train the Model -----

model.fit(train_images, train_labels, epochs=5)

# ----- Evaluate Accuracy -----

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)  # Test accuracy: 0.8671
# overfitting -> model performs worse on new data than training data

# ----- Make Predictions -----

predictions = model.predict(test_images)
print(predictions[0])  # shows the confidence the model has for each class
print(np.argmax(predictions[0]))  # prints the index of the class with the highest probability

# helper methods for visualizing predictions
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# shows the predictions for the first image
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# shows the predictions for the 12th image
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# plot the first X test images, their predicted label, and the true label
# color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()
