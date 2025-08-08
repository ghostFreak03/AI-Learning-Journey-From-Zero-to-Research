import numpy as np

number_of_sixes_np = np.array([8.5, 9.5, 9.9, 9]) # number of sixes hit by every team till now in each season
win_loss_record_np = np.array([0.65, 0.8, 0.8, 0.9]) # win/lose percent of every team in each season
number_of_fans_np = np.array([1.2, 1.3, 0.5, 1.0]) # number of fans for every team in each season (in millions)

number_of_sixes = [8.5, 9.5, 9.9, 9] # number of sixes hit by every team till now in each season
win_loss_record = [0.65, 0.8, 0.8, 0.9] # win/lose percent of every team in each season
number_of_fans = [1.2, 1.3, 0.5, 1.0] # number of fans for every team in each season (in millions)

weight = [0.1, 0.2, 0]
weight_np = [0.1, 0.2, 0]

weights_np = [[0.1, 0.1, -0.3], # to predict hurt player count
              [0.1, 0.2, 0], # to predict win/loss %
              [0, 1.3, 0.1]] # to predict if players are sad

def multi_input_single_output() :
    weight = [0.1, 0.2, 0]
    def neural_network(input, weight) :
        return weighted_sum(input, weight)
    
    def weighted_sum(a, b) :
        assert(len(a) == len(b))
        output = 0
        for i in range(len(a)) :
            output += a[i] * b[i]
        return output
    
    input = [number_of_sixes[0], win_loss_record[0], number_of_fans[0]]
    return neural_network(input, weight)

# Same using NumPy

def multi_input_single_output_numpy() :
    
    def nueral_network(input) :
        return input.dot(weight_np)
    
    input = np.array([number_of_sixes_np[0], win_loss_record_np[0], number_of_fans_np[0]])
    pred = nueral_network(input)
    return pred

# Multiple weights to predict multiple things(outputs) with same input

def multi_input_multi_output() :

    def vector_matrix_mul (vector) :
        assert(len(vector) == len(weights))
        output = [0, 0, 0]
    
        for i in range(len(vector)) :
            output[i] = weighted_sum(vector, weights[i])
    
        return output
    
    def weighted_sum(a, b) :
        assert(len(a) == len(b))
        output = 0
        for i in range(len(a)) :
            output += a[i] * b[i]
        return output
    
    input = [number_of_sixes[0], win_loss_record[0], number_of_fans[0]]
    return vector_matrix_mul(input)

# There can multiple networks ie., multiple weight matrix. One set of prediction can be the input of next network. 
# This is mainly used in image classification

def multi_input_multi_output_two_layers() :
    def nueral_network(input, weights) :
        hidden = input.dot(weights[0])
        final = hidden.dot(weights[1])
        return final

    hidden_layer_weight = np.array(
        [[0.1, 0.2, -0.1], # to predict hurt player count
        [-0.1, 0.1, 0.9], # to predict win/loss %
        [0.1, 0.4, 0.1]] # to predict if players are sad
    )
    final_layer_weight = np.array(
        [[0.3, 1.1, -0.3], # to predict hurt player count
        [0.1, 0.2, 0], # to predict win/loss %
        [0, 1.3, 0.1]] # to predict if players are sad
    )

    weights = [hidden_layer_weight, final_layer_weight]
    input = np.array([number_of_sixes[0], win_loss_record[0], number_of_fans[0]])
    
    pred = nueral_network(input, weights)
    return pred


print("multi_input_single_output - ", multi_input_single_output())
print("multi_input_single_output_numpy - ", multi_input_single_output_numpy())
print("multi_input_multi_output - ", multi_input_multi_output())
print("multi_input_multi_output_two_layers - ", multi_input_multi_output_two_layers())

# Different useful functions of numpy - 
print(np.zeros((2, 4))) # prints 2 x 4 matrix of zeros
print(np.random.rand(2, 4)) # prints 2 x 4 matrix of random numbers from 0 to 1

# Dot product mechanism - 

# Number of col of matrix1 should be equal to number of row of matrix2 in order to do dot product and output with shape of [#rows in m1, #col in m2]

a = np.zeros((1, 4))
b = np.zeros((4, 3))
c = a.dot(b)
print(c.shape) #[1, 3]

d = np.zeros((5, 4)).T # T is transpose - makes [5, 4] as [4, 5]
e = np.zeros((5, 6))
f = d.dot(e)
print(f.shape) #[4, 6]

g = np.zeros((6, 4))
h = np.zeros((5, 6))
#i = g.dot(h)
#print(i.shape) #Error
j = h.dot(g) # reversing so that it'll be (5, 6)*(6, 4) instead of (6, 4)*(5, 6)
print(j.shape)
# thus dot product thumb rule - when two matrices which are being dot product put side by side, the neighbouring values of each matrix should be same
# (1, 4).(4, 3) -> (1, 3) - here 4,4 is the neighbouring values and they are same
