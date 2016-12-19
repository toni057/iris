# Iris and tensorflow

Here we build a simple softmax classifier for the [iris](https://en.wikipedia.org/wiki/Iris_flower_data_set) dataset using tensorflow.  


Model fitted is of the form:  
+ y = W * x + b

where x are the input features, W weights vector and b the intercept term.

To obtain class probability estimates we minimize the softmax cross entropy criterion.
