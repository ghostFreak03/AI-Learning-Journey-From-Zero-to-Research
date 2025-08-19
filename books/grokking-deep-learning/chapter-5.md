Chapter 5 - 

Same concepts as in chapter 4 but with multiple inputs with multiple weights for corresponding inputs and single/mulitple outputs. This gives us one/multiple delta which can be used to calculate weight_delta for each input and adjust the weight for each input.  A important property of neural network - 

Consider three inputs a, b and c. I we never change weight of a in each iteration and only adjust weights of b and c and finally if network learns to predict required goal without a, then it will never be able to incorporate a input into its predictions.

If weight_delta of a input is always 0 - there will be no weight update for a
But still the network will learn with the help of b and c. At the end the network is able to predict the goal. Now if we add a back into the learning , the network will never be correctly incorporate an into its predictions. Thus the network can predict with only the inputs which it used for learning from the beginning
