Chapter 6 - 

Input dataset for neural network - 
- This is a matrix where each column depicts the state of a particular thing in all the iterations. Each row represents the state of all the things in a single iteration. Example, below steers light dataset shows state of each light in each iteration, here stop is represented by 0 and walk by 1,
			Input(What you know)			    Output(whether we can walk or not)
												(what you want to know)
        - 1           0          1                                                        ->     0
        - 0           1          1                                                        ->     1
        - 0           0          1                                                        ->     0
        - 1           1          1                                                         ->     1
        - 0           1          1                                                        ->     1
        - 1           0          1                                                        ->     0

- The relation between what you know (input dataset) and what you want to know (goal dataset) is called pattern. This pattern is a property of matrix.


- If we build a NN with whatever we learned till now for this street light dataset, we would take first row of input dataset and train NN to reduce error and get the prediction as close as first row of walk_stop matrix which is the goal matrix. We can loop through entire data set and do the same. Take a row, train NN to reduce error on prediction(adjust weights with weight_delta) and move on to next row. This will be done for multiple iterations. This type of gradient descent learning is called Stochastic gradient descent.
- Average/Full gradient descent - Here we don’t update the weight for each training example(each row of streetlight matrix). Instead we go through all rows, calculate the average of weight_delta throughout the rows and finally update the weights. This will be done for multiple iterations.
- Batch gradient descent - This is in the middle of stochastic and average gradient descent. Here instead of updating weights for each row or updating after processing all the rows, we update weights after a fixed batch size of rows. More on this in later chapters.

Imagine this example of streetlights - [1, 0, 1] whereas weights are [0.5, 1, -0.5] and goal is 0
	Now the pred would be 1*0.5 + 0*1 + 1 * -0.5 = 1.5 - 1.5 = 0 == goal
Here the network predicts prefect output without learning anything but just because the prediction was accurate accidentally. This network would fail in real world prediction because there was no learning. This is called overfitting.

We can overcome this failure by not training the network in a smaller subset of data but by training it on larger dataset. Even if the network finds perfect correlation between input and output accidentally for some inputs, the other inputs can compensate for it and make the network actually learn. Thus we make network to generalize instead of memorize

In the street light example, we can see that the far right light is always on, so it doesn’t provide anything in predicting the output. Though in each prediction calculation it causes the weight to move upward and downward without any use. This unnecessary process just creates noise and without this network can learn faster. This is were Regularizarion comes in where the inputs which causes equal upward and downward pressure on the weights are silenced. More on this later.

This can also happen in all the inputs. If the entire input dataset has no correlation with the output dataset then you introduce and intermediate dataset in the middle which has some correlation with the output dataset. Thus between layer 0(input) and layer 2(output) introduce a new layer 1. This is called stacking of neural network.

But how do we update weights while predicting between layer 0 and layer 1? We usually calculate delta by getting diff bw pred and goal. But here what will be the goal? Here comes Back Propogation.

If there are 3 layers say layer 0, 1 and 2 where 1 is the intermediate layer, 0 is the input layer and 2 is the output layer. In order to find the delta at layer 1, we need to find delta at layer 2 and multiply it with the respective weights of each node in layer 1. This gives the delta at layer 1 which can be used to adjust the weights at layer 0 -> layer 1. This is called Back Propogation.

But why does this work??

A node at layer 1 contributes to the prediction and error based on its weight. If weight is higher, it contributes more and if it is lower it contributes less. So, if we want to increase/decrease the prediction by x amount(delta) we need the layer 1 delta to be same amount multiplied by the weight because the weight decided the contribution of each node to the error / delta. I feel like I understood this but at the same time it feels hazy. Let’s go forward and see if it gets more clear.

Now what is the use if this middle layer?

It’s just a single line math. You multiply input with weight and get middle node value, later multiply it with another weight and get prediction like below, 1 * 10 * 5 ‎ = 50
But this can be done with two layers itself like below,
5 x 10 ‎ = 50
Thus any three layer prediction can be done with two layer network. Then what is the use of middle layer? Initially we needed this layer for regularization where when the input has no correlation with the output. We introduced a middle layer which can create a new correlation with output layer when input layer has no correlation. But how does this create new correlation if it’s just single line math? Because it’s not a single line math. The single middle layer actually subscribes to the correlation of all the input nodes or only one of the input node or few of the input nodes. It’s mixing up the correlation of the input nodes to create a new correlation with the output node. 

Also the middle node we can specify some condition and based on which it can subscribe to correlation of an input node or completely shut off an input node. This is called conditional correlation or sometimes correlation.

This condition creates a non-linearity in the network. There are different non linearity conditions/logic but one of the best and most used is “if the node would be negative, set it to 0 or turn if off”. This kind of non-linearity logic is called Relu. Without this non-linearity logic, the network would be linear and lose the use of using multi layered neural network as described before.
