Chapter 7 - 

In general, a neural network tries to find or create correlation between the input layer and output layer by adjusting the weights. Different configurations of weights and layers between input and output layer play a major role in finding the correlation. These configurations are called architectures. 

There are multiple architectures and each has their own pros and cons. Different architecture channel signal between layers in different ways to make correlation easier to discover. Will discuss on different architectures in rest of the chapters.


Chapter 8 - 

Overfitting & Regularization - 

Overfitting - When a network trains more time on the training data say images, it gets fine tuned to more granular level of the image and thus predicts poorly if the test data image is little bit different even though its the image of same item. This is called Overfitting. When a network relies on the noises instead of just the signals from the training data, overfitting happens. What is noise and signals? See below example,

Consider two pictures of dogs where one is a dog laying on a pillow with a background (img1) and other is just a silhouette of a dog with black filled inside dog shape and background is white (img2). 
If the network is trained very well on img1, it would pickup all the noises like pillow and background instead of just the signals like dog shape, ears and other dog specifications. This would cause the network to overfit and predict a picture as dog only if it has these features/noises. Thus the network will fail with test image img2 which is just a dog silhouette.

Basically overfitting is like tailoring a dress which perfectly fits a group of people and thus it won’t fit at all when people of that group wear it. Over training causes overfitting.

Thus we need some way to make the network ignore the fine grained details and capture only the general information from the training data. This brings us to Regularization. Regularization is set of methods to make a network generalize on data points instead of just memorizing the training data.

Regularization method - Early stopping - 

Early stopping is a simplest method of regularization. But when to stop training? We need to run the model on the data that isn’t in training set and stop if prediction success rate start decreasing. This second set of data is called “validation set”.

Thus we have 3 sets of data, 
- training set, 
- testing set(data used to test accuracy after training. Don’t use it as validation set because model could overfit with test data), 
- validation set

Regularization method - Dropout - 

As the name sounds, we turn off nodes (set to 0) randomly during training. This causes network to train using random subsections of the neural network. This actually works very well and used in many networks because of the simplicity. But why it works?

Theory - Dropout makes big network small by randomly training little subsections of the network at a time and little networks don’t overfit.

Imagine a mold of fine grained sand used to take the the shape of a fork. It’ll capture each and every fine details of the fork as small grains can move and adjust to the exact shape of the fork. But what is its coin sized stones, it’ll capture only the rough shape of the fork instead of the fine details of the fork. Here smaller networks acts as the big stones because there are less number of nodes in it and big networks has multiple nodes which acts like sand particles.

One more interesting property is, even if smaller networks overfit a little bit no two smaller network would overfit for the same noise. But why?

Because each smaller network start at different random weights, make different mistakes and update accordingly. They stop once they learned enough noise and signals to predict accurately. Thus if there are 100 smaller neural networks(all initialized randomly) trained on same data, each will get latched onto different noise but similar broader signal. Thus when they mistake due to overfitting, all will make different mistakes due to different noises. Their mistakes due to noises will get cancelled out between them giving out only what they all learned in common - the signal

when half of layer_1 node are dropped(made 0), layer_2 will get only half weighted sum from layer_1. but layer_2 requires full value 
        # because without it, it will cause inconsistency while predicting using test_data where we don't use dropout method.
        # so we multiply the layer_1 value by 2 after dropout
