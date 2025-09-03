import numpy as np

np.random.seed(1) # seed(1) makes sure that numpy generate same set of random numbers in each run

def relu(x) :
    return (x > 0) * x # in numpy x > 0 can behave as 1 if true or 0 if false. so it comes down to 1 * x if x > 0 or 0 * x if x <= 0

#We need this because if relu() set a node to be 0, then its contribution to the error is also 0. so we set the delta of that node to be 0
def relu2deriv(output) :
    return output > 0

street_light = np.array([[1, 0, 1],
                        [0, 1, 1],
                        [0, 0, 1],
                        [1, 1, 1]])
walk_vs_stop = np.array([[1],
                        [1],
                        [0],
                        [0]]) # Goal
hidden_size = 4
alpha = 0.1

# weights between layer 0 and layer 1
# why 2 * and -1 ? 
# np.random.random generates random values in [0,1)
# multiplied by 2 it generates in range [0, 2)
# with -1 it generated in range [-1, 1] thus we have both positive and negative numbers

weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1 # input is 1 x 3 and weight is 3 x 4, so dot would return 1 x 4 which is 4 predictions

#weights between layer 1 and layer 2
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1 # input is 1 x 4 and weight is 4 x 1, so dot would return 1 x 1 which is 1 predictions

for iter in range(60) :
    layer_2_error = 0
    for i in range(len(street_light)) :
        layer_0 = street_light[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1)) # this returns matrix of 1 x 4
        layer_2 = np.dot(layer_1, weights_1_2) # this return matrix of 1 x 1 (1 x 4 * 4 x 1 = 1 x 1)

        layer_2_delta = walk_vs_stop[i:i+1] - layer_2
        layer_2_error += np.sum(layer_2_delta ** 2)

        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu2deriv(layer_1)

        weights_0_1 += alpha * (np.dot(layer_0.T, layer_1_delta))
        weights_1_2 += alpha * (np.dot(layer_1.T, layer_2_delta))

        print("pred - ", str(layer_2))


def stochastic_grad_descent() :
    for i in range(40) :
        
        error_for_all_lights = 0
        
        for row in range(len(walk_vs_stop)) :
            input = street_light[row]
            true = walk_vs_stop[row]
            pred = input.dot(weights)
            delta = pred - true
            error = delta ** 2
            error_for_all_lights += error
            weight_delta = delta * input
        
            print("Pred - ", str(pred))
            print("Delta - ", str(delta))
            print("Error - ", str(error))
        
            weights = weights - (alpha * weight_delta)
        
            print("Updated weight - ", str(weights))
        print("Overall error - ", str(error_for_all_lights))
