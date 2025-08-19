def neural_network(inputs, weights) :
    output = 0
    for i in range(len(inputs)) :
        output += inputs[i] * weights[i]
    return output

def neural_network_single_input(input, weights) :
    return ele_mul(input, weights)

def ele_mul(scalar, vector) :
    output = [0 for _ in range(len(vector))] 
    for i in range(len(vector)) :
        output[i] = scalar * vector[i]
    return output

def vector_matrix_mul (vector, matrix) :
    assert(len(vector) == len(matrix))
    output = [0, 0, 0]

    for i in range(len(vector)) :
        output[i] = weighted_sum(vector, matrix[i])

    return output
    
def weighted_sum(a, b) :
    assert(len(a) == len(b))
    output = 0
    for i in range(len(a)) :
        output += a[i] * b[i]
    return output

def outer_prod(vec_a, vec_b) :
    out = [[0 for _ in range(len(vec_a))] for _ in range(len(vec_b))]
    for i  in range(len(vec_a)) :
        for j in range(len(vec_b)) :
            out[i][j] = vec_a[i] * vec_b[j]
    return out

def grad_descent_multi_input_single_output() :
    number_of_sixes = [8.5, 9.5, 9.9, 9] # number of sixes hit by every team till now in each season
    win_loss_record = [0.65, 0.8, 0.8, 0.9] # win/lose percent of every team in each season
    number_of_fans = [1.2, 1.3, 0.5, 1.0] # number of fans for every team in each season (in millions)
    
    win_or_lose_binary = [1, 1, 0, 0] # goal
    true = win_or_lose_binary[0]

    alpha = 0.01
    weights = [0.1, 0.2, -.1]
    inputs = [number_of_sixes[0], win_loss_record[0], number_of_fans[0]]

    for iter in range(3) :
        pred = neural_network(inputs, weights)
        delta = pred - true
        error = delta ** 2
        weight_delta = ele_mul(delta, inputs)

        print("Iteration - ", str(iter + 1))
        print("Pred - ", str(pred))
        print("Delta - ", str(delta))
        print("Error - ", str(error))
        print("Weight delta - ", str(weight_delta))
        print("Weights before - ", str(weights))
        
        for i in range(len(weights)) :
            weights[i] -= weight_delta[i] * alpha

        print("Weights after - ", str(weights))

def grad_descent_single_input_multiple_output() :
    win_loss_record = [0.65, 0.8, 0.8, 0.9] # win/lose percent of every team in each season

    hurt = [0.1, 0, 0, 0.1]
    sad = [0.1, 0, 0.1, 0.2]
    win = [1, 1, 0, 1]
    true = [hurt[0], win[0], sad[0]]

    alpha = 0.01
    weights = [0.3, 0.2, 0.9]
    input = win_loss_record[0]

    for iter in range(30) :
        pred = neural_network_single_input(input, weights)
        delta = [0 for _ in range(len(pred))]
        error = [0 for _ in range(len(pred))]
        for j in range(len(true)) :
            delta[j] = pred[j] - true[j]
            error[j] = delta[j] ** 2
            
        weight_delta = ele_mul(input, delta)

        print("Iteration - ", str(iter + 1))
        print("Pred - ", str(pred))
        print("Delta - ", str(delta))
        print("Error - ", str(error))
        print("Weight delta - ", str(weight_delta))
        print("Weights before - ", str(weights))
        
        for i in range(len(weights)) :
            weights[i] -= weight_delta[i] * alpha

        print("Weights after - ", str(weights))

def grad_descent_multiple_input_multiple_output() :
    number_of_sixes = [8.5, 9.5, 9.9, 9] # number of sixes hit by every team till now in each season
    win_loss_record = [0.65, 0.8, 0.8, 0.9] # win/lose percent of every team in each season
    number_of_fans = [1.2, 1.3, 0.5, 1.0] # number of fans for every team in each season (in millions)

    hurt = [0.1, 0, 0, 0.1]
    sad = [0.1, 0, 0.1, 0.2]
    win = [1, 1, 0, 1]
    true = [hurt[0], win[0], sad[0]]

    alpha = 0.01
    weights = [[0.1, 0.1, -0.3],
               [0.1, 0.2, 0],
               [0, 1.3, 0.1]]
    inputs = [number_of_sixes[0], win_loss_record[0], number_of_fans[0]]

    for iter in range(30) :
        pred = vector_matrix_mul(inputs, weights)
        delta = [0 for _ in range(len(pred))]
        error = [0 for _ in range(len(pred))]
        for j in range(len(true)) :
            delta[j] = pred[j] - true[j]
            error[j] = delta[j] ** 2
            
        weight_delta = outer_prod(inputs, delta)

        print("Iteration - ", str(iter + 1))
        print("Pred - ", str(pred))
        print("Delta - ", str(delta))
        print("Error - ", str(error))
        print("Weight delta - ", str(weight_delta))
        print("Weights before - ", str(weights))
        
        for i in range(len(weights)) :
            for j in range(len(weights)) :
                weights[i][j] -= weight_delta[i][j] * alpha

        print("Weights after - ", str(weights))

grad_descent_multi_input_single_output()
grad_descent_single_input_multiple_output()
grad_descent_multiple_input_multiple_output()
        

    
