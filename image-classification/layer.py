import tensorflow as tf
import math

#%%

def create_variable_on_cpu(name, shape, dtype, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var
        
        
class Layer():
    
    def get(self):
        return self.layer
    

    def __init__(self, name, x):
        self.name = name
        self.x = x

    
    def add_layer(self, **kwargs):
        # sort out in and out dimensions
        if len(kwargs) == 1:
            w_shape = kwargs['kernel']
            b_shape = w_shape[3]
        elif len(kwargs) == 2:
            w_shape = [kwargs['in_dim'], kwargs['out_dim']]
            b_shape = kwargs['out_dim']

        self.w = create_variable_on_cpu(name='weights', 
                                 shape=w_shape,
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(0.1)))
        
        self.b = create_variable_on_cpu(name='bias',
                                 shape=b_shape,
                                 dtype=tf.float32,
                                 initializer = tf.constant_initializer(0.0))
        return self

    
    def conv(self, kernel, stride=1):
        self.type = 'conv'
        self.add_layer(kernel=kernel)
        self.layer = tf.nn.conv2d(self.x, self.w, strides=[1, stride, stride, 1], padding='VALID') + self.b
        return self
        
    
    def linear(self, in_dim, out_dim):
        self.type = 'linear'
        self.add_layer(in_dim=in_dim, out_dim=out_dim)
        self.layer = tf.matmul(self.x, self.w) + self.b
        return self

        
    def batch_norm(self):
        axes = [0, 1, 2] if self.type == 'conv' else [0]
        
        mean, variance = tf.nn.moments(self.layer, axes, name='moment')
        self.layer=tf.nn.batch_normalization(self.layer, mean, variance, self.b, None, 1e-5, name='batch_norm')
        return self
        
    
    def activation(self, activation='relu'):
        if activation == 'relu':
            self.layer = tf.nn.relu(self.layer)
        elif activation == 'sigmoid':
            self.layer = tf.nn.sigmoid(self.layer)
        elif activation == 'softmax':
            self.layer = tf.nn.softmax(self.layer)
        elif activation == 'linear':
            self.layer = self.layer        
        return self
        
    
    def dropout(self, keep_prob=0.75):
        self.layer = tf.nn.dropout(self.layer, keep_prob)
        return self
