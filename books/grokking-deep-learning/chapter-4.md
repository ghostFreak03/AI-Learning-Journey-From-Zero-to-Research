Chapter 4 - 

How down tweak weight knobs? 
	We find the error in the prediction and learn from it by adjusting the weights in order to have a more perfect prediction. Error is a way to find out how much you missed.

One of the method to find error - mean squared error
One of the method of learning - gradient descent

Mean squared error - 
- Here we square the difference between the prediction and the expected result. 
- why square? - in order to keep the error positive. an archer hitting 2 inch above target and 2 inch below target should have same error 2 inch, not +2 inch and -2 inch. 
- Also our goal here is to reduce the average error to 0. This will be a problem if error goes positive and negative because avg of +1000 error and -1000 error would be 0 but we haven’t actually reached 0 error state. False positive
- squaring helps make big errors(>1) bigger and small errors(<1) smaller. thus the model can learn to ignore small errors and focus on big errors


Learning - 

Hot & Cold method - 
- Adjusting the weight knob both up and down, seeing which reduces the error more and finally moving the knob in that direction. Repeat this till error becomes 0 eventually
- Main drawback in hot & cold method, you have to predict three times in an iteration(actual pred, weight up red, weight down) in order to compare and move weight in a particular direction. 
- Also the amount(set_amount) you increase/decrease the weights by is a set amount and has no relation with the input or prediction. This makes this method of learning keep on going without reaching the goal prediction when the set_amount is very high or very low. This creates uncertainty.

Gradient Descent - 
- This method involves applying three types of effects on pure error by multiplying it with input to get the direction_and_amount by which the weight should change.
-  pure error - pure error is the error which is the raw diff bw prediction and goal. its the error which we haven’t squared it. This give us the direction and amount in which we are off by.
- 3 types of effects on pure error - 
    - Stopping -
        - If input is 0, then there is no point in adjusting the weight because prediction will always be 0. So multiplying pure error with input(0) we make the direction_and_amount by which weight is updated as 0.
    - Negative reversal -
        - When input is +ve, increasing weight increases prediction. But when input is -ve increasing weight decreases prediction. In order to address this bias, if we multiply input with pure error, it will reverse the sign of direction_and_amount if input is negative. This ensures that weight moves in correct direction even if input is negative. 
            - Note - I think I understood this but it’s little cloudy. Let’s clear it up when we go deeper
    - Scaling - 
        - Since direction_and_amount is a multiple of input, the weight grows big when input is big. This could cause weight to go out of control if input is very big. Later we’ll learn and use alpha to address this. This alpha helps to control how fast the network learns and avoid overshooting.

Error is directly proportional weight. Thus, weight can be directly adjusted to get 0 error. 

Derivative - Understanding how one variable changes when you move another variable is called derivative. 
- If there are two rods a and b sticking out of a box, you move rod a 1 inch inside and if rod b moves 2 inch inside then here the derivative is,
			rod_b = rod_a * 2
- In gradient descent weight_delta is our derivative

Alpha - 
Without alpha the network overreacts when input is high because we multiply delta with input for weight_delta. Thus in each iteration error will be high as the network overreacts more and more. This phenomenon is called divergence.

Usually alpha is a single real-valued number between 0 and 1. But how do we find the correct alpha for our neural network? It’s often done by guessing.
- If divergence is still high even after introducing alpha, then in shows alpha is not enough - increase the alpha
- If learning of neural network is very slow, then it shows weight updates are very less - decrease the alpha

Prediction / pred - input * weight
Error - (pred - goal) ** 2
Delta / pure_error - pred - goal
weight_delta / direction_and_amount / derivative - (pred - goal) * input

Thus weight update would be - weight = weight - (weight_delta * alpha)
