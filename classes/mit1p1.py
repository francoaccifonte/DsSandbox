import os
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from pdb import set_trace as st


class Mit1():
    def main(self):
        sport = tf.constant("Tennis", tf.string)
        number = tf.constant(1.41421356237, tf.float64)

        print("`sport` is a {}-d Tensor".format(tf.rank(sport).numpy()))
        print("`number` is a {}-d Tensor".format(tf.rank(number).numpy()))

        sports = tf.constant(["Tennis", "basket"], tf.string)
        numbers = tf.constant([1.41421356237, 2], tf.float64)

        print("`sports` is a {}-d Tensor".format(tf.rank(sports).numpy()))
        print("`numbers` is a {}-d Tensor".format(tf.rank(numbers).numpy()))

    def tasks(self):
        '''TODO: Define a 2-d Tensor.'''
        matrix = tf.constant([[1, 2], [3, 4]], tf.float64)
        assert isinstance(matrix, tf.Tensor), "matrix must be a tf Tensor object"
        assert tf.rank(matrix).numpy() == 2

        '''TODO: Define a 4-d Tensor.'''
        # Use tf.zeros to initialize a 4-d Tensor of zeros with size 10 x 256 x 256 x 3.
        #   You can think of this as 10 images where each image is RGB 256 x 256.
        images = tf.constant(
            tf.zeros([10, 256, 256, 3], dtype=tf.float64), tf.float64
        )  # TODO

        assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
        assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
        assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"

    def computations(self):
        # Create the nodes in the graph, and initialize values
        a = tf.constant(15)
        b = tf.constant(61)

        # Add them!
        c1 = tf.add(a, b)
        c2 = a + b  # TensorFlow overrides the "+" operation so that it is able to act on Tensors
        print(c1)
        print(c2)

    def computations2(self):
        def func(a, b):
            '''TODO: Define the operation for c, d, e (use tf.add, tf.subtract, tf.multiply).'''
            a = tf.constant(a, dtype=tf.float64)
            b = tf.constant(b, dtype=tf.float64)
            c = tf.add(a, b)
            one = tf.constant(1, dtype=tf.float64)
            d = tf.subtract(b, one)
            e = tf.multiply(c, d)
            return e
        a, b = 1.5, 2.5
        e_out = func(a, b)
        print(e_out)


class OurDenseLayer(tf.keras.layers.Layer):
    # if __name__ == '__main__':
    #     tf.random.set_seed(1)
    #     layer = OurDenseLayer(3)
    #     layer.build((1, 2))
    #     x_input = tf.constant([[1, 2.]], shape=(1, 2))
    #     y = layer.call(x_input)

    #     print(y.numpy())
    #     mdl.lab1.test_custom_dense_layer_output(y)

    def __init__(self, n_output_nodes):
        super(OurDenseLayer, self).__init__()
        self.n_output_nodes = n_output_nodes

    def build(self, input_shape):
        d = int(input_shape[-1])
        # st()

        self.W = self.add_weight("weight", shape=[d, self.n_output_nodes])
        self.b = self.add_weight("bias", shape=[1, self.n_output_nodes])

    def call(self, x):
        z = tf.matmul(x, self.W)
        y = tf.sigmoid(z + self.b)
        return y


def test_model():
    # Define the number of outputs
    n_output_nodes = 3

    # First define the model
    model = Sequential()
    '''TODO: Define a dense (fully connected) layer to compute z'''
    # Remember: dense layers are defined by the parameters W and b!
    # You can read more about the initialization of W and b in the TF documentation :) 
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense?version=stable
    dense_layer = Dense(n_output_nodes, activation='sigmoid')

    # Add the dense layer to the model
    model.add(dense_layer)

    x_input = tf.constant([[1,2.]], shape=(1,2))
    st()
    model_output = model(x_input).numpy()


def subclassing():
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense

    class SubclassModel(tf.keras.Model):
        def __init__(self, n_output_nodes):
            super(SubclassModel, self).__init__()
            self.dense_layer = Dense(n_output_nodes, activation='sigmoid')

        def call(self, inputs):
            return self.dense_layer(inputs)

    n_output_nodes = 3
    model = SubclassModel(n_output_nodes)

    x_input = tf.constant([[1, 2.]], shape=(1, 2))

    print(model.call(x_input))


def tape():
    # Function minimization with automatic differentiation and SGD ###

    # Initialize a random value for our initial x
    x = tf.Variable([tf.random.normal([1])])
    print("Initializing x={}".format(x.numpy()))

    learning_rate = 1e-2  # learning rate for SGD
    history = []
    # Define the target value
    x_f = 4

    # We will run SGD for a number of iterations. At each iteration, we compute the loss, 
    #   compute the derivative of the loss with respect to x, and perform the SGD update.
    for i in range(500):
        with tf.GradientTape() as tape:
            loss = (x - x_f) ** 2  # "forward pass": record the current loss on the tape

        # loss minimization using gradient tape
        grad = tape.gradient(loss, x)  # compute the derivative of the loss with respect to x
        new_x = x - learning_rate * grad  # sgd update
        x.assign(new_x)  # update the value of x
        history.append(x.numpy()[0])

    # Plot the evolution of x as we optimize towards x_f!
    plt.plot(history)
    plt.plot([0, 500], [x_f, x_f])
    plt.legend(('Predicted', 'True'))
    plt.xlabel('Iteration')
    plt.ylabel('x value')
    plt.show()


if __name__ == '__main__':
    tape()
