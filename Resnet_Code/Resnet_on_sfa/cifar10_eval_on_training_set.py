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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import cv2

import numpy as np
import tensorflow as tf

import cifar10
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_filename', 'SFA_pixel_regions_test_set.tfrecords',
                           """TFRecords training set filename""")

# tf.app.flags.DEFINE_string('input_filename', 'cifar10_eval.tfrecords',
#                            """TFRecords training set filename""")

tf.app.flags.DEFINE_string('eval_dir', './new_eval_on_training',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './32_by_32_faces_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 786432,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


ori_basefolder = '../../SkinDataset/ORI'
test_filenames = ['243.jpg','278.jpg']
full_ori_test_filenames = [os.path.join(ori_basefolder,te) for te in test_filenames]

im = cv2.imread(full_ori_test_filenames[0],cv2.IMREAD_GRAYSCALE)
im_rows = im.shape[0]
im_cols = im.shape[1]

bool_vec = [False]*FLAGS.num_examples

def eval_once(saver, summary_writer, top_k_op, summary_op, iterator):
  global bool_vec
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:

    sess.run(iterator.initializer)

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint

      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / (FLAGS.batch_size*2) ))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * 2 * FLAGS.batch_size
      step = 0
      initial_bool_vec = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        bool_vec[initial_bool_vec:(initial_bool_vec+predictions.shape[0])] = predictions[:]
        initial_bool_vec = initial_bool_vec+predictions.shape[0]
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


##############################################

INITIAL_IMAGE_SIZE = cifar10.IMAGE_SIZE
IMAGE_SIZE = cifar10.IMAGE_SIZE

def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    vector_decoded = tf.reshape(tf.decode_raw(parsed_features["image"],tf.uint8) , [INITIAL_IMAGE_SIZE*INITIAL_IMAGE_SIZE*3] )
    red_decoded = tf.slice(vector_decoded,[0],[INITIAL_IMAGE_SIZE*INITIAL_IMAGE_SIZE])
    green_decoded = tf.slice(vector_decoded,[INITIAL_IMAGE_SIZE*INITIAL_IMAGE_SIZE],[INITIAL_IMAGE_SIZE*INITIAL_IMAGE_SIZE])
    blue_decoded = tf.slice(vector_decoded,[2*INITIAL_IMAGE_SIZE*INITIAL_IMAGE_SIZE],[INITIAL_IMAGE_SIZE*INITIAL_IMAGE_SIZE])

    red_decoded = tf.reshape(red_decoded,[INITIAL_IMAGE_SIZE,INITIAL_IMAGE_SIZE,1] )
    green_decoded = tf.reshape(green_decoded,[INITIAL_IMAGE_SIZE,INITIAL_IMAGE_SIZE,1] )
    blue_decoded = tf.reshape(blue_decoded,[INITIAL_IMAGE_SIZE,INITIAL_IMAGE_SIZE,1] )

    image_decoded = tf.concat([red_decoded, green_decoded, blue_decoded],2)
    image_decoded = tf.cast(image_decoded, tf.float32)
    
    final_image = tf.image.per_image_standardization(image_decoded)
    return final_image, tf.cast(parsed_features["label"], tf.int32)


##############################################

def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'



    dataset = tf.data.TFRecordDataset(FLAGS.input_filename)
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    dataset = dataset.repeat()  # Repeat the input indefinitely. 
    dataset = dataset.batch(FLAGS.batch_size)               
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    top_k_op_list = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:

                  images, labels = next_element

                  logits = cifar10.inference(
                      images, should_summarize=False)
                  top_k_op_list.append(tf.nn.in_top_k(
                      logits, labels, 1))
                  tf.get_variable_scope().reuse_variables()                  


    top_k_op = top_k_op_list[0]
    for i in xrange(1,FLAGS.num_gpus):
        top_k_op = tf.concat([top_k_op,top_k_op_list[i]],0)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op, iterator)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

    draw_result()


def draw_result():
  for k in xrange(len(test_filenames)):
    bin_im = np.zeros([im_rows,im_cols],dtype = np.uint8)
    for i in xrange(len(im_rows)):
      for j in xrange(len(im_cols)):
        if bool_vec[k*(im_rows*im_cols) + i*(im_cols) + j]:
          bin_im[i,j] = 255
    cv2.imwrite(str(i)+'.jpg',bin_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def main(argv=None):  # pylint: disable=unused-argument
  # cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
