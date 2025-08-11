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

Prediction - input * weight
Error - ((input * weight) - goal) ** 2
Delta(pure error) - (input * weight) - goal
weight_delta(direction_and_amount) - ((input * weight) - goal) * input

This shows that error is directly proportional weight. Thus, weight can be directly adjusted to get 0 error. 
