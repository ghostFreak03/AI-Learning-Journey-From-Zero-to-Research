Chapter 10 - 

Again to the problem of overfitting. Regularization is an approach to avoid overfitting but there is a more popular and better method to counter overfitting. It’s the convolution layer. 


Convolution layer - 

Overfitting is caused when we have more input parameters with weights and less dataset to compare the predictions with. If there 100 parameters and only limited true dataset, then the model will adjust the weights for each parameters and memorize the patterns instead of learning it. Thus we have to reduce the input parameters to have only the generalized nodes but who do we know which to remove and which to keep. If we remove more nodes, then the model loses the ability to learn. Thus we need to find an optimal structure in which there are less number of nodes (generalized nodes) and yet the model is more expressive (can still learn and predict). One of the most popularly used structure is called convolution. When it’s used in a layer, it’s called convolutional layer.

Core idea in convolutional layer is instead of having a large linear layer (large number of nodes in a single layer) connecting to all the output nodes, we split the input into smaller linear layers with one output node for each smaller group. This output node will later get forward propagated to next layer.

If there are 1 2 3 …. 100 input nodes, 10 output nodes, then each 100 input node would be connected to each output node. Thus there’ll be 100*10=1000 weights. Now if we split the input layer into smaller groups of 25 nodes each with one output node for each group, there’ll be (25*1) + (25*1) + (25*1) + (25*1) = 100 weights. Now these four nodes can be used to predict the 10 output nodes. Thus it’ll be 100 + (4*10) ‎ = 140 weights. 

This each smaller group is called convolutional kernel. 

In an example of an image of 2, consider that there are 8x8 matrix of pixels. Consider a kernel of size 3x3 matrix. This kernel will act like a sliding window and move one pixel to the right after each prediction at the current location. At the end, it would have processed all 3x3 positions in the 8x8 matrix and predicted a 6x6 matrix of values. Now 8x8 matrix has been reduced to 6x6 matrix. What if there are multiple kernels say., 4 used at the same time individually  on the same image of 2. Imagine that each kernel is trained to predict a certain pattern in the image. Now each kernel would produce a 6x6 matrix. If we combine all 4 of the 6x6 matrices, we would get a generalized input of image 2 instead of the fine grained version which was the original input. We can use this to train the entire model.

Notice that this technique allows each kernel to learn a particular pattern and then search for the existence of that pattern somewhere in the image.

Thus this reduces the ration of weights to datapoints and thus drastically reduces the ability to overfit to training data and increasing its ability to generalize.
