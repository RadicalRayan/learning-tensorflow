from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

"""
Rayan Krishnan 7/9/19
Based on tutorial found : https://www.tensorflow.org/tutorials/keras/basic_text_classification
"""

print(tf.__version__)

# ----- Download the Data -----
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Training entries: 25000, labels: 25000
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
print(len(train_data[0]), len(train_data[1]))  # shows the number of words in two different reviews: (218, 189)
# neural nets must be of the same length so this must be resolved

# ----- Helper Dictionary to Convert Integers to Words -----
# dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved for certain keys
word_index = {k: (v+3) for k, v in word_index.items()}  # everything is shifted by 3
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# test of converting the first training data
print(decode_review(train_data[0]))

# ----- Prepare the Data -----
"""
The array of integers must be converted to tensors to be fed into the NN. There are two methods
    * Convert the arrays into vectors of 0s and 1s for word occurrence (one-hot encoding)
      The sequence of [3, 5] would, for example, be stored as a 10,000 dimensional vector with all 0s
      except 1s for 3 and 5. This requires a matrix of size num_words * num_reviews
    * Pad all the arrays so they have the same length. Make a tensor of shape max_length * num_reviews.
      This is the method that will be implemented
"""
# pad_sequences standardizes the length of the arrays
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(len(train_data[0]), len(train_data[1]))  # (256, 256)
print(train_data[0])  # this is what the first padded review looks like

# ----- Build the Model -----
# the vocabulary size is 10,000 words for movie reviews
vocab_size = 10000
# number of hidden input and outputs between layers. Amount of freedom when learning
# more units may lead to learning more complex representations -> more computational intensive and leads to overfitting
hidden_units = 16

model = keras.Sequential()
# takes the integer-encoded vocabulary and looks up the vector for each index.
# the resulting dimensions are (batch, sequence, embedding)
model.add(keras.layers.Embedding(vocab_size, hidden_units))
# returns a fix-length output vector for each example, averaging over the sequence dimension
# this effectively handles the issue of inputs of variable length
model.add(keras.layers.GlobalAveragePooling1D())
# fully connected layer with 16 hidden units using rectified linear activation
model.add(keras.layers.Dense(hidden_units, activation=tf.nn.relu))
# fully connected and returns one value. Because it has a sigmoid activation function, the output is from 0 to 1
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

# create a validation set (separate from train or test data to fine tune our model)
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# ----- Train the Model -----
# 40 epochs in mini-batches of 512 samples
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# ----- Evaluate the Model -----
results = model.evaluate(test_data, test_labels)
print(results)

history_dict = history.history
print(history_dict.keys())

# ----- Create a Graph of Accuracy and Loss Over Time -----
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


