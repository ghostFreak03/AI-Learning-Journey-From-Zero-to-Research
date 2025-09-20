import numpy as np
import sys
from keras.datasets import mnist
np.random.seed(1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = (x_train[0:1000].reshape(1000, 28 * 28) / 255, y_train[0:1000])

one_hot_labels = np.zeros((len(labels), 10))

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28 * 28) / 255
test_labels = np.zeros((len(y_test), 10))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

# activation functions - AF

# hidden layer AF
def tanh(x): # replacement for relu
    return np.tanh(x)
def tanh2Deriv(output):
    return 1 - (output ** 2)

# output layer
def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis = 1, keepdims = True)
    
# relu = lambda x: (x >= 0) * x
# relu2Deriv = lambda x: x >= 0

alpha, iterations = (2, 300)
pixels_per_image, num_labels = (784, 10)
batch_size = 128 # for batched gradient descent

input_rows = 28
input_cols = 28

kernel_rows = 3
kernel_cols = 3
num_kernels = 16

hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels

kernel_weights = 0.02 * np.random.random((kernel_rows * kernel_cols, num_kernels)) - 0.01
# weights_0_1 = 0.02 * np.random.random((pixels_per_image, hidden_size)) - 0.01
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

# convolutional layer
def get_image_section(layer, row_from, row_to, col_from, col_to) :
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to-row_from, col_to-col_from)

for j in range(iterations): 
    correct_count = 0

    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
        
        layer_0 = images[batch_start:batch_end]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        layer_0.shape

        sects = list()
        for row_start in range(layer_0.shape[1] - kernel_rows) :
            for col_start in range(layer_0.shape[2] - kernel_cols) :
                sect = get_image_section(layer_0, row_start, row_start + kernel_rows, col_start, col_start + kernel_cols)
                sects.append(sect)

        expanded_input = np.concatenate(sects, axis = 1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], - 1)
        kernel_output = flattened_input.dot(kernel_weights)
        
        layer_1 = tanh(kernel_output.reshape(es[0], -1))

        # Regularization method - droupout 
        # this generated a matrix of same shape as layer_1 with 0s and 1s (with 50% chance for both)
        dropout_mask = np.random.randint(2, size = layer_1.shape)

        layer_1 *= dropout_mask * 2 # each node layer_1 will become either 0 or remain the same - thus droupout achieved
        # when half of layer_1 node are dropped(made 0), layer_2 will get only half weighted sum from layer_1. but layer_2 requires full value 
        # because without it, it will cause inconsistency while predicting using test_data where we don't use dropout method.
        # so we multiply the layer_1 value by 2 after dropout
        
        layer_2 = softmax(np.dot(layer_1, weights_1_2))

        # error += np.sum((labels[batch_start:batch_end] - layer_2) ** 2)

        for k in range(batch_size) :
            correct_count += int(np.argmax(layer_2[k:k+1]) == np.argmax(labels[batch_start + k : batch_start + k + 1]))
            
        layer_2_delta = (labels[batch_start:batch_end] - layer_2) / (batch_size * layer_2.shape[0])
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * tanh2Deriv(layer_1)
        layer_1_delta *= dropout_mask # not required but best practice. you make delta 0 of nodes which are 0 because of dropout

        weights_1_2 += alpha * np.dot(layer_1.T, layer_2_delta)
        
        # weights_0_1 += alpha * np.dot(layer_0.T, layer_1_delta)

        lld_reshape = layer_1_delta.reshape(kernel_output.shape)
        kernel_delta = flattened_input.T.dot(lld_reshape)
        kernel_weights -= alpha * kernel_delta

    test_correct_count = 0
    
        # test_error = 0.0

    for i in range(len(test_images)) :
        layer_0 = test_images[i:i+1]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        layer_0.shape
        sects = list()
        for row_start in range(layer_0.shape[1] - kernel_rows) :
            for col_start in range(layer_0.shape[2] - kernel_cols) :
                sect = get_image_section(layer_0, row_start, row_start + kernel_rows, col_start, col_start + kernel_cols)
                sects.append(sect)

        expanded_input = np.concatenate(sects, axis = 1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], - 1)
        kernel_output = flattened_input.dot(kernel_weights)
        
        layer_1 = tanh(kernel_output.reshape(es[0], -1))
        layer_2 = np.dot(layer_1, weights_1_2)

        # test_error += np.sum((test_labels[i:i+1] - layer_2) ** 2)
        test_correct_count += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

    if(j % 1 == 0) :
        sys.stdout.write(
            "\n" + \
            "I:" + str(j) + \
            # "Test-Err:" + str(test_error / float(len(test_images)))[0:5] + \
            " Test-Acc:" + str(test_correct_count / float(len(test_images))) + \
            # "Train-Err:" + str(error / float(len(images)))[0:5] + \
            " Train-Acc:" + str(correct_count / float(len(images)))
        )
        







