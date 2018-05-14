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

"""A binary to train CIFAR-10 using multiple GPUs with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

from PIL import Image

import math

from tensorflow.python import debug as tf_debug

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './32_by_32_faces_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('input_filename', 'SFA_pixel_regions_training_set.tfrecords',
                           """TFRecords training set filename""")

tf.app.flags.DEFINE_string('eval_filename', 'SFA_pixel_regions_test_set.tfrecords',
                           """TFRecords eval set filename""")

# tf.app.flags.DEFINE_string('input_filename', 'cifar10_train.tfrecords',
#                            """TFRecords training set filename""")

tf.app.flags.DEFINE_integer('num_examples', cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
                            """Number of examples to run.""")

tf.app.flags.DEFINE_integer('eval_num_examples', cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
                            """Number of eval examples to run.""")

tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_string('best_dir', './best_dir',
                            """Best eval dir.""")                            

calculate_rates = True

# INITIAL_IMAGE_SIZE = 150
INITIAL_IMAGE_SIZE = cifar10.IMAGE_SIZE
IMAGE_SIZE = cifar10.IMAGE_SIZE


def _parse_function_no_distortion(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image_decoded = tf.reshape(tf.decode_raw(parsed_features["image"], tf.uint8), [
                                INITIAL_IMAGE_SIZE, INITIAL_IMAGE_SIZE, 3])


    image_decoded = tf.cast(image_decoded, tf.float32)

    final_image = tf.image.per_image_standardization(image_decoded)
    return final_image, tf.cast(parsed_features["label"], tf.int32)


def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image_decoded = tf.reshape(tf.decode_raw(parsed_features["image"], tf.uint8), [
                                INITIAL_IMAGE_SIZE , INITIAL_IMAGE_SIZE , 3])


    image_decoded = tf.cast(image_decoded, tf.float32)

    brightness_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    contrast_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    hue_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    saturation_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    rotation_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    zoom_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    skew_x_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    skew_y_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    translate_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)

    image_decoded = tf.image.random_flip_left_right(image_decoded)

    # angle = tf.random_uniform(
    #     [1], minval=(-1 * (math.pi / 4)), maxval=math.pi / 4, dtype=tf.float32)
    # image_rotated = tf.contrib.image.rotate(
    #     image_decoded, angle, interpolation='BILINEAR')
    # image_decoded = tf.cond(rotation_percentage < 0.4,
    #                         lambda: image_rotated, lambda: image_decoded)

    image_brightness = tf.image.random_brightness(image_decoded, max_delta=0.8)
    image_decoded = tf.cond(brightness_percentage < 0.3,
                            lambda: image_brightness, lambda: image_decoded)

    image_contrast = tf.image.random_contrast(
        image_decoded, lower=0.7, upper=1.5)
    image_decoded = tf.cond(contrast_percentage < 0.3,
                            lambda: image_contrast, lambda: image_decoded)

    image_hue = tf.image.random_hue(image_decoded, max_delta=0.2)
    image_decoded = tf.cond(hue_percentage < 0.5,
                            lambda: image_hue, lambda: image_decoded)

    image_saturation = tf.image.random_saturation(
        image_decoded, lower=0.5, upper=1.5)
    image_decoded = tf.cond(saturation_percentage < 0.3,
                            lambda: image_saturation, lambda: image_decoded)

    zoom_scale = tf.random_uniform(
        [], minval=1.0, maxval=1.1, dtype=tf.float32)
    new_size = tf.constant(
        [INITIAL_IMAGE_SIZE, INITIAL_IMAGE_SIZE], dtype=tf.float32) * zoom_scale
    new_size = tf.cast(new_size, tf.int32)
    image_zoom = tf.image.resize_images(image_decoded, new_size)
    image_zoom = tf.image.resize_image_with_crop_or_pad(
        image_zoom, INITIAL_IMAGE_SIZE, INITIAL_IMAGE_SIZE)
    image_decoded = tf.cond(zoom_percentage < 0.1,
                            lambda: image_zoom, lambda: image_decoded)

    # skew_x_angle = tf.random_uniform(
    #     [1], minval=(-1 * (math.pi / 12)), maxval=math.pi / 12, dtype=tf.float32)
    # skew_x_tan = tf.tan(skew_x_angle)
    # skew_x_vector_1 = tf.constant([1], dtype=tf.float32)
    # skew_x_vector_2 = tf.constant([0, 0, 1, 0, 0, 0], dtype=tf.float32)
    # skew_x_vector = tf.concat([skew_x_vector_1,skew_x_tan, skew_x_vector_2],0)
    # skewed_x_image = tf.contrib.image.transform(image_decoded, skew_x_vector, interpolation='BILINEAR')
    # image_decoded = tf.cond(skew_x_percentage < 0.1,
    #                         lambda: skewed_x_image, lambda: image_decoded)

    # skew_y_angle = tf.random_uniform(
    #     [1], minval=(-1 * (math.pi / 12)), maxval=math.pi / 6, dtype=tf.float32)
    # skew_y_tan = tf.tan(skew_y_angle)
    # skew_y_vector_1 = tf.constant([1, 0, 0], dtype=tf.float32)
    # skew_y_vector_2 = tf.constant([1, 0, 0, 0], dtype=tf.float32)
    # skew_y_vector = tf.concat([skew_y_vector_1,skew_y_tan, skew_y_vector_2],0)
    # skewed_y_image = tf.contrib.image.transform(image_decoded, skew_y_vector, interpolation='BILINEAR')
    # image_decoded = tf.cond(skew_y_percentage < 0.1,
    #                         lambda: skewed_y_image, lambda: image_decoded)

    # translate_y = tf.random_uniform(
    #     [1], minval=(-1 * (INITIAL_IMAGE_SIZE / 5)), maxval=INITIAL_IMAGE_SIZE / 6, dtype=tf.float32)
    # translate_x = tf.random_uniform(
    #     [1], minval=(-1 * (INITIAL_IMAGE_SIZE / 5)), maxval=INITIAL_IMAGE_SIZE / 6, dtype=tf.float32)
    # translate_vector_1 = tf.constant([1, 0], dtype=tf.float32)
    # translate_vector_2 = tf.constant([0, 1], dtype=tf.float32)
    # translate_vector_3 = tf.constant([0, 0], dtype=tf.float32)
    # translate_vector = tf.concat(
    #     [translate_vector_1, translate_x, translate_vector_2, translate_y, translate_vector_3], 0)
    # translated_image = tf.contrib.image.transform(image_decoded, translate_vector, interpolation='BILINEAR')
    # image_decoded = tf.cond(translate_percentage < 0.1,
    #                         lambda: translated_image, lambda: image_decoded)    

    final_image = tf.image.per_image_standardization(image_decoded)
    return final_image, tf.cast(parsed_features["label"], tf.int32)


def tower_loss(scope, images, labels):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
      images: Images. 4D tensor of shape [batch_size, height, width, 3].
      labels: Labels. 1D tensor of shape [batch_size].

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    logits = cifar10.inference(images, is_train=True)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = cifar10.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def eval_once(sess, summary_writer, top_k_op, summary_op, acc_iterator, global_step, eval, num_examples, current_lr, lr):
    """Run Eval once.

    Args:
        saver: Saver.
        summary_writer: Summary writer.
        top_k_op: Top K op.
        summary_op: Summary op.
    """

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))

        num_iter = int(math.ceil(num_examples / (2*FLAGS.batch_size) ))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * FLAGS.batch_size * 2
        step = 0
        while step < num_iter and not coord.should_stop():
            # predictions = sess.run([top_k_op], {lr: current_lr})
            predictions = sess.run([top_k_op])
            true_count += np.sum(predictions)
            step += 1

        # Compute precision @ 1.
        precision = true_count / total_sample_count

        summary = tf.Summary()
        # summary.ParseFromString(
        #     sess.run(summary_op, {lr: current_lr}) )
        summary.ParseFromString(
            sess.run(summary_op) )        
        if eval:
            summary.value.add(tag='Eval Set Precision @ 1',
                              simple_value=precision)
        else:
            summary.value.add(tag='Training Precision @ 1',
                              simple_value=precision)
        summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return precision


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        num_steps_per_epoch = (
            cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) / (FLAGS.batch_size * FLAGS.num_gpus)
        decay_steps = int(num_steps_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        # lr = tf.placeholder( dtype = tf.float32)
        lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        cifar10.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)        

        # Create an optimizer that performs gradient descent.
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

        # Get images and labels for CIFAR-10.

        dataset = tf.data.TFRecordDataset(FLAGS.input_filename)
        # Parse the record into tensors.
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(buffer_size=cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
        dataset = dataset.repeat()  # Repeat the input indefinitely.
        dataset = dataset.prefetch(FLAGS.batch_size * 7)
        dataset = dataset.batch(FLAGS.batch_size)
        iterator = dataset.make_initializable_iterator()

        # GET TRAINING ACCURACY
        acc_dataset = tf.data.TFRecordDataset(FLAGS.input_filename)
        # Parse the record into tensors.
        acc_dataset = acc_dataset.map(_parse_function_no_distortion)
        acc_dataset = acc_dataset.repeat()  # Repeat the input indefinitely.
        acc_dataset = acc_dataset.batch(FLAGS.batch_size)
        acc_iterator = acc_dataset.make_initializable_iterator()

        eval_acc_dataset = tf.data.TFRecordDataset(FLAGS.eval_filename)
        # Parse the record into tensors.
        eval_acc_dataset = eval_acc_dataset.map(_parse_function_no_distortion)
        # Repeat the input indefinitely.
        eval_acc_dataset = eval_acc_dataset.repeat()
        eval_acc_dataset = eval_acc_dataset.batch(FLAGS.batch_size)
        eval_acc_iterator = eval_acc_dataset.make_initializable_iterator()

        # Calculate the gradients for each model tower.
        tower_grads = []
        acc_top_k_op_list = []
        eval_acc_top_k_op_list = []        
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
                        # Dequeues one batch for the GPU

                        image_batch, label_batch = iterator.get_next()

                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        loss = tower_loss(scope, image_batch, label_batch)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        acc_images, acc_labels = acc_iterator.get_next()
                        acc_logits = cifar10.inference(
                            acc_images, should_summarize=False)
                        acc_top_k_op_list.append(tf.nn.in_top_k(
                            acc_logits, acc_labels, 1))
                        tf.get_variable_scope().reuse_variables()


                        eval_acc_images, eval_acc_labels = eval_acc_iterator.get_next()
                        eval_acc_logits = cifar10.inference(
                            eval_acc_images, should_summarize=False)
                        eval_acc_top_k_op_list.append(tf.nn.in_top_k(
                            eval_acc_logits, eval_acc_labels, 1))
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(
                            tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        acc_top_k_op = acc_top_k_op_list[0]
        eval_acc_top_k_op = eval_acc_top_k_op_list[0]
        for i in xrange(1,FLAGS.num_gpus):
            acc_top_k_op = tf.concat([acc_top_k_op,acc_top_k_op_list[i]],0)
            eval_acc_top_k_op = tf.concat([eval_acc_top_k_op,eval_acc_top_k_op_list[i]],0)                        

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(
                    var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_op = opt.apply_gradients(
                grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root='./tmp_dump')
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init)

        coord = tf.train.Coordinator()

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(acc_iterator.initializer)
        sess.run(eval_acc_iterator.initializer)
        sess.run(iterator.initializer)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        acc_last_precision = 0.0
        eval_acc_last_precision = 0.0

        # draw(sess, image_batch, label_batch)
        # exit()


        best_precision = 0.0

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            current_epoch = float((step * FLAGS.batch_size * FLAGS.num_gpus) //
                                  (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN))

            if current_epoch == 0:
                current_lr = cifar10.INITIAL_LEARNING_RATE
            else:
                current_lr = (cifar10.INITIAL_LEARNING_RATE) / \
                    math.sqrt(current_epoch)
                if current_lr < 1e-6:
                    current_lr = 1e-6


            # _, loss_value = sess.run(
            #     [train_op, loss], {lr: current_lr})

            _, loss_value = sess.run(
                [train_op, loss])            
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                epoch = int(current_epoch)

                format_str = ('%s: step: %d; epoch: %d; loss = %.6f, last_precision = %.2f, last_eval_precision =  %.2f, best_eval_precision =  %.2f  (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, epoch, loss_value, acc_last_precision, eval_acc_last_precision, best_precision,
                                     examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                # summary_str = sess.run(summary_op, {lr: current_lr})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if step % 15000 == 0 and calculate_rates and step > 0:

                acc_last_precision = eval_once(
                    sess, summary_writer, acc_top_k_op, summary_op, acc_iterator, step, False, FLAGS.num_examples, current_lr, lr)

                eval_acc_last_precision = eval_once(
                    sess, summary_writer, eval_acc_top_k_op, summary_op, eval_acc_iterator, step, True, FLAGS.eval_num_examples, current_lr, lr)

                if eval_acc_last_precision > best_precision:
                    best_precision = eval_acc_last_precision
                    best_checkpoint_path = os.path.join(FLAGS.best_dir, 'model.ckpt')
                    saver.save(sess, best_checkpoint_path, global_step=step)                    







def draw(sess, image_batch, label_batch):

    im, label = sess.run([image_batch, label_batch])
    shape = im.shape
    for i in xrange(shape[0]):

        if label[i] == 0:
            continue

        imagem = im[i, :, :, :]

        im_min = np.amin(imagem)
        im_max = np.amax(imagem)

        imagem_f = (((imagem - im_min) / (im_max - im_min))
                    * 255).astype(np.uint8)

        imagem_f_rgb = np.zeros(imagem_f.shape,dtype = np.uint8)

        imagem_f_rgb[:,:,0] = imagem_f[:,:,2]
        imagem_f_rgb[:,:,1] = imagem_f[:,:,1]
        imagem_f_rgb[:,:,2] = imagem_f[:,:,0]

        pImg = Image.fromarray(imagem_f_rgb, "RGB")
        pImg = pImg.resize((INITIAL_IMAGE_SIZE, INITIAL_IMAGE_SIZE), Image.LANCZOS)
        pImg.show()


        raw_input()


def main(argv=None):  # pylint: disable=unused-argument
    # cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    if tf.gfile.Exists(FLAGS.best_dir):
      tf.gfile.DeleteRecursively(FLAGS.best_dir)
    tf.gfile.MakeDirs(FLAGS.best_dir)    

    train()


if __name__ == '__main__':
    tf.app.run()
