Chapter 9 - 

Activation function - An activation function is a function applied to neurons in a layer. It takes a value and return another value. This makes us think that all the functions are activation functions but there are some constraints which make a function an activation function. Relu is an activation function.

Constraint 1 - There should be output number for any input number. There shouldn’t bet be a case where there’ll be no output for a particular input.

Constraint 2 - A good activation function should be monotonic and never change direction. It should either have always increasing output or always decreasing output. 
Example - y = x * x here if x is the input, then y will decrease when x moves from negative to 0 and increase after x crosses 0 - this is not a good activation function
Y = x is a good activation function.

This is not a strict requirement but this can cause multiple input to have same output. For example, in y = x * x -> y will be same for x = -2 and x = 2 which will be a problem in neural networks where we search for specific output. It’s like multiple correct answers. 

Constraint 3 - A good activation function is non-linear ie., there don’t give output which can be plotted in a straight line. They squiggle or turn.
Ex - y = (2 * x) + 5 -> this produces output in a straight line -> not a good activation function
Y = relu(x) -> negative x values will become 0 and other values will be same. So its not straight line -> good activation function

Constraint 4 - Should be easy to compute as it’ll be called multiple times in a network.

Some of the standard activation functions in use right now,

Sigmoid - It transforms/squishes any input number to a number with 0 and 1. This lets you interpret the output of any individual neuron as a probability. This is used in both hidden layers and output layers.

tanh - sigmoid will always give output between 0 and 1 (positive correlation) but tanh gives output between -1 and 1. This means that it can give both positive correlation and negative correlation. Thus it won’t be much useful in output layers (useful only if your output prediction is required to be between -1 and 1), it’ll always outperform sigmoid in hidden layers as the negative correlation is powerful for hidden layers.

Usage of activation functions on output layers depends on what you are trying to predict in the output layer. Some of the different needs are,

No activation function - Some network output layers might not need any activation functions like predicting temperature of a city with temperatures of surrounding cities. Sigmoid or tanh would predict between 0 and 1 or -1 and 1, but we want to predict the temperature.

Predicting yes/no probabilities - If you are predicting yes/no probabilities (one or many outputs) sigmoid would be best fit as it models individual probabilities separately for each output node.

Predicting which-one probability - In MNIST digit classifier, the final output layer neurons ranging from 0 to 9 each having different prediction probability. Consider the input is an image of 9, and if the network accurately predicts with 100 value in node 9 and 0 in all other nodes. 

Now if apply sigmoid in output layers, the values would change from 0 to 0.5 in all the nodes except 9 where it’ll change to 0.99. This is because 0.0 is the middle value in sigmoid and thus it’ll give out 0.5. Now the network thinks, “The image mostly depicts an image of 9 but there is a chance that it could be other numbers also”. 
This becomes even worse in back propagation as the error in all nodes would be 0.25 (0.5 * 0.5) while in node 9 it would be 0.0 (0.01 * 0.01). We can see that the error all the nodes other than 9 is greater than 0 even when the network predicted correctly because sigmoid changed the output value.

Thus for these cases we use Softmax, which in this case will make the predictions in all the node except 9 as 0.0 while node 9 will get the value of 1.0. Now the network believes, “this is definitely 9, no doubt” and thus the error will be zero in back propagation. 

Softmax doesn’t say that there’s a 50% probability that this image is 1, 50% probability that this image 2 and so on. The sum of all probabilities across all nodes in softmax always comes to 100 ie., it would say that there’s a 0% probability that this image is 1, 0% probability that this image is 1 ….. 100% probability that this image is 9 or 10% probability that this image is 1, 20% probability that this image is 1 ….. 70% probability that this image is 9.

How is softmax computed? - raise each value exponentially ie., if x is the value predicted in the output layer, raise it exponentially -> e ^ x. Here e is a special number ~2.71828. As per our previous example all 0 values in the nodes except 9 (where the value is 100), will become 1.0 and node 9 becomes something big (2.68 * 10 ^ 43).
 
Now sum all the values -> 1.0 + 1.0 + 1.0 …. + (2.68 * 10 ^ 43) = ~ (2.68 * 10 ^ 43)

Now divide each value with this sum -> 1.0 / (2.68 * 10 ^ 43) = ~0.000… => 0.0

Thus all the values except node 9 will become 0.0 and node 9 will become 1.0 because the computation would be like ~(2.68 * 10 ^ 43) / ~(2.68 * 10 ^ 43) = ~1.0

Thus softmax will increase higher values more aggressively while lower values will be increased by very very less. Hence higher prediction will be a clear winner and there’ll be no confusion in the network. This increasing is called sharpness of attenuation. You can have less aggressive softmax function (lower attenuation) by decreasing the e value or more aggressive softmax function (higher attenuation) by increasing the e value. But this e value (~2.71828) is the standard.

