
def neural_network(input, weight) :
    return input * weight

# hot & cold learning
def part_1() :
    knob_weight = 0.5
    input = 0.5
    prediction_goal = 0.8
    
    pred = input * knob_weight
    
    error = (pred - prediction_goal) ** 2 
    # why square? - in order to keep the error positive. an archer hitting 2 inch above target and 2 inch below target should have same error 2 inch, not +2 inch and -2 inch.
    # also squaring helps make big errors(>1) bigger and small errors(<1) smaller. thus the model can learn to ignore small errors and focus on big errors
    
    print(error)
    
    weight = 0.1
    lr = 0.01
    
    number_of_sixes = [8.5]
    win_or_lose = [1] # win - this is the expected output
    input = number_of_sixes[0]
    goal = win_or_lose[0]
    
    pred = neural_network(input, weight)
    error = (pred - goal) ** 2
    
    # move weight up and down by lr and see which has less error
    pred_up = neural_network(input, weight + lr)
    error_up = (pred_up - goal) ** 2
    
    pred_down = neural_network(input, weight - lr)
    error_down = (pred_down - goal) ** 2
    
    if(error_down < error_up) :
        weight -= lr
    else :
        weight += lr
    
    
    print("error - ", error)
    print("error_up - ", error_up)
    print("error_down - ", error_down)

# hot & cold learning
def part_2() :
    weight = 0.5
    input = 0.5
    prediction_goal = 0.8

    step_amount = 0.10 # adjustment value based on error

    for iteration in range(1101) :
        pred = neural_network(input, weight)
        error = (pred - prediction_goal) ** 2
        print("Error - " + str(error) + ", Prediction - " + str(pred))
        pred_up = neural_network(input, weight + step_amount)
        error_up = (pred_up - prediction_goal) ** 2

        pred_down = neural_network(input, weight - step_amount)
        error_down = (pred_down - prediction_goal) ** 2

        if(error_up < error_down) :
            weight += step_amount
        else :
            weight -= step_amount

def gradient_descent() :

    weight = 0
    alpha = 0.01 # in order to control how fast the network learns and avoid overshooting
    
    number_of_sixes = [8.5]
    win_or_lose = [1] # win - this is the expected output
    input = number_of_sixes[0]
    prediction_goal = win_or_lose[0]
    

    for iteration in range(20) :
        pred = neural_network(input, weight)
        error = (pred - prediction_goal) ** 2
        delta = (pred - prediction_goal) # delta is how much we missed (previously known as pure error)
        weight_delta = input * delta # previously known as direction_and_amount (mutiply input to apply scaling, negative reversing & stopping)
        
        print("Error - " + str(error) + ", Prediction - " + str(pred))
        weight -= weight_delta * alpha
    

# part_1()
# part_2()
gradient_descent()



    
