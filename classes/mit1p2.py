import os
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import Tensorflow 2.0
# %tensorflow_version 2.x
import tensorflow as tf 

# Download and import the MIT 6.S191 package
# !pip install mitdeeplearning
import mitdeeplearning as mdl

# Import all remaining packages
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
from pdb import set_trace as st
# !apt-get install abcmidi timidity > /dev/null 2>&1

# Check that we are using a GPU, if not switch runtimes
#   using Runtime > Change Runtime Type > GPU
# assert len(tf.config.list_physical_devices('GPU')) > 0l


def generate_song_file(song):
    mdl.lab1.play_song(song)


def dataset():
    return mdl.lab1.load_training_data()


def vectorize_string(string):
    vectorized_output = np.array([char2idx[char] for char in string])
    return vectorized_output


# Download the dataset
songs = dataset()
songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

vectorized_songs = vectorize_string(songs_joined)

seq_length = 5
batch_size = 1


def get_batch(vectorized_songs, seq_length, batch_size):
    # the length of the vectorized songs string
    n = vectorized_songs.shape[0] - 1
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n - seq_length, batch_size)

    input_batch = [vectorized_songs[i: i+seq_length] for i in idx]
    output_batch = [vectorized_songs[i+1: i+seq_length+1] for i in idx]

    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])

    return x_batch, y_batch


x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)


def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        LSTM(rnn_units),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss


# Hyperparameter setting and optimization ###

# Optimization parameters:
num_training_iterations = 2000  # Increase this to train longer
batch_size = 4  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters:
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024  # Experiment between 1 and 2048

# Checkpoint location:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")


# Define optimizer and training operation ###
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
optimizer = tf.keras.optimizers.Adam(learning_rate)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = compute_loss(y, y_hat)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

##################
# Begin training!#
##################


history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
if hasattr(tqdm, '_instances'):
    tqdm._instances.clear()  # clear if it exists

do_train = False
if do_train:
    for iter in tqdm(range(num_training_iterations)):
        # Grab a batch and propagate it through the network
        x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
        loss = train_step(x_batch, y_batch)

        # Update the progress bar
        # history.append(loss.numpy().mean())
        # plotter.plot(history)
        print("loss function value: {}".format(loss.numpy().mean()))

        # Update the model with the changed weights!
        if iter % 10 == 0:
            model.save_weights(checkpoint_prefix)

    # Save the trained model and the weights
    model.save_weights(checkpoint_prefix)

def 