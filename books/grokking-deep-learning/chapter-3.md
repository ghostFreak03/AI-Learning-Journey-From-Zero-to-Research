**What is a neural network?**

- It’s an interface which accepts inputdata(real world/easily knowable data) as _information_ and a weight (in previous chapters mentioned as knobs) as _knowledge_ and outputs a _prediction_. 
- It adjusts the **knowledge/weight** based on the accuracy of its previous predictions. So this is a supervised trial and error learning ie., **Supervised Parameterized Learning**

In our example, our NN predicts win % of a cricket team based on number of sixes hit by them. But there are more to cricket in order to predict. So we can increase the number of parameters and make the prediction more accurate

**WeightedSum/dotProduct** - Sum of each input multiplied by its corresponding **weight/sum** of predictions of each input. This dotProduct gives a _notion of similarity_ between two data arrays/**vectors**(in python)

**Final prediction of the model strongly depends on how input is relative to its corresponding weight**. 

 - If weight is low then input needs to be high in order to increase the output(prediction). 

 - If weight[0.5, 0, 0, 1] => if input[0] is BIG OR if input[3] is 1, then output is high

 - If weight is negative then input also should be negative.

 - If weight[-1, 0, 1, 0] => if NOT input[0] OR if input[2] is 1 then output is high

 - If weight[1, 0, 0, 0] => if input[0] is 1, then output is high

 - If weight is all 0 then there’ll be no high output	

This is because we are multiplying input[i] with output[i] and then adding this prediction with other position predictions. **Thus the relation between input and weight is the key to get high output value.**

We can use input data(**vectors/array**) and have multiple weight array(**matrix**) so that we can make predictions on different things with same input data. This includes the process of **vector-matrix multiplication** where input(vector) is multiplied with weights(matrix)

We can pass in an output vector from a neural network as input to next/same neural network with different weights. This type of multiple neural networks feed will be mainly used in **Image processing**.

**Dot product mechanism** -

 - Number of col of matrix1 should be equal to number of row of matrix2 in order to do dot product and output with shape of `[#rows in m1, #col in m2]`

```
	m1 = np.zeros((1, 4))
	m2 = np.zeros((4, 3))
	output = m1.dot(m2)
	output shape = [1, 3]
```

 - Thus dot product thumb rule - **when two matrices which are being dot product put side by side, the neighboring values of each matrix should be same. Or else it’ll be error**
 (1, 4).(4, 3) -> (1, 3) - here 4,4 is the neighboring values and they are same
 (1, 4).(2, 3) -> Error - here 4,2 is the neighboring values and they are not same
