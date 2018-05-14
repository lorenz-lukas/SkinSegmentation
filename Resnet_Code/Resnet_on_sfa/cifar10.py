# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 120,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 15.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.3  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01     # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype,seed = 1))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data,
                                          data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inference(images, is_train=False, should_summarize=True):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU is_training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #

    common_stdev = 5e-2
    common_weight_decay = 5e-3

    initial_conv_feature_maps = 64
    initial_conv_type = 5
    initial_stdev = common_stdev
    initial_weight_decay = common_weight_decay

    with tf.variable_scope('initial_conv') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[
                                                initial_conv_type, initial_conv_type, 3, initial_conv_feature_maps],
                                            stddev=initial_stdev,
                                            wd=initial_weight_decay)
        conv=tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        initial_conv_relu=tf.nn.relu(conv, name=scope.name)
        if should_summarize:
            _activation_summary(initial_conv_relu)

    # Consider pooling


    # conv layers weights
    conv_weights=[64, 128, 256, 512, 512]
    conv_stdevs=[common_stdev] * 5
    conv_weight_decays=[common_weight_decay] * 5
    conv_types= [3]*5
    dropout_rates=[0.9, 0.9, 0.8, 0.7, 0.7]
    pooling_layers=[False, True, True, True, False]
    input_tensors=[initial_conv_relu]

    # conv_weights=[64, 128, 256, 256, 256]
    # conv_stdevs=[common_stdev] * 5
    # conv_weight_decays=[common_weight_decay] * 5
    # conv_types= [3]*5
    # dropout_rates=[0.9, 0.9, 0.8, 0.8, 0.8]
    # pooling_layers=[False, True, True, False, False]
    # input_tensors=[initial_conv_relu]    

    for i in xrange(len(conv_weights)):
        conv_type=conv_types[i]
        input_maps=conv_weights[i - \
            1] if (i - 1) >= 0 else initial_conv_feature_maps
        output_maps=conv_weights[i]
        shortcut=input_tensors[i]


        with tf.variable_scope('res_conv' + str(i)) as scope:

            if pooling_layers[i]:
                kernel_fix_dim=_variable_with_weight_decay('dim_matching_weights',
                                                    shape=[1, 1,
                                                        input_maps, output_maps],
                                                    stddev=conv_stdevs[i],
                                                    wd=conv_weight_decays[i])

                shortcut=tf.nn.conv2d(shortcut, kernel_fix_dim, [
                                1, 2, 2, 1], padding='SAME')

            bn1=tf.contrib.layers.batch_norm(shortcut,
                                            center=True, scale=True,
                                            is_training=is_train,
                                            scope='bn1')

            bn_relu_1=tf.nn.relu(bn1, name=scope.name + "1")

            kernel_1=_variable_with_weight_decay('weights1',
                                                shape=[conv_type, conv_type,
                                                    output_maps, output_maps],
                                                stddev=conv_stdevs[i],
                                                wd=conv_weight_decays[i])

            conv1=tf.nn.conv2d(bn_relu_1, kernel_1, [
                              1, 1, 1, 1], padding='SAME')

            bn2=tf.contrib.layers.batch_norm(conv1,
                                            center=True, scale=True,
                                            is_training=is_train,
                                            scope='bn2')

            bn_relu_2=tf.nn.relu(bn2, name=scope.name + "2")

            kernel_2=_variable_with_weight_decay('weights2', shape=[conv_type, conv_type,output_maps, output_maps], stddev=conv_stdevs[i], wd=conv_weight_decays[i])

            conv2=tf.nn.conv2d(bn_relu_2, kernel_2, [1, 1, 1, 1], padding='SAME')

            addded_activation=conv2 + shortcut

            if is_train:
                addded_activation = tf.nn.dropout(addded_activation,dropout_rates[i])

            if should_summarize:
                _activation_summary(addded_activation)


        input_tensors.append(addded_activation)

    # global_avg_pooling
    with tf.variable_scope('global_avg_pooling') as scope:

        last_relu = tf.nn.relu(input_tensors[-1])

        l_width=last_relu.get_shape()[1].value
        l_height=last_relu.get_shape()[2].value

        pool=tf.nn.avg_pool(last_relu, ksize=[1, l_width, l_height, 1],
                    strides=[1, l_width, l_height, 1], padding='SAME', name='global_pool')

        reshape=tf.reshape(pool, [FLAGS.batch_size, -1])
        dim=conv_weights[-1]


    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights=_variable_with_weight_decay('weights', [dim, NUM_CLASSES],
                                              stddev=1 / dim, wd=0.0)
        biases=_variable_on_cpu('biases', [],
                                  tf.constant_initializer(0.0))
        softmax_linear=tf.add(
            tf.matmul(reshape, weights), biases, name=scope.name)
        if should_summarize:
            _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels=tf.cast(labels, tf.int64)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean=tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages=tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses=tf.get_collection('losses')
    loss_averages_op=loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps=int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr=tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op=_add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt=tf.train.GradientDescentOptimizer(lr)
        grads=opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op=opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages=tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op=variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op=tf.no_op(name='train')

    return train_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory=FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename=DATA_URL.split('/')[-1]
    filepath=os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _=urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo=os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path=os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
