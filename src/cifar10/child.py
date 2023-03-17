import sys

import numpy as np
import src.framework as fw

from src.utils import count_model_params
from src.utils import get_train_ops
from src.utils import DEFINE_boolean, DEFINE_float, DEFINE_integer, DEFINE_string
from src.utils import LearningRate, ClipMode, Optimizer, LayeredModel
from src.cifar10.image_ops import batch_norm

DEFINE_integer("batch_size", 32, "")
DEFINE_integer("child_cutout_size", None, "CutOut size")
DEFINE_integer("child_num_layers", 5, "")
DEFINE_string("child_fixed_arc", None, "")
DEFINE_float("child_grad_bound", 5.0, "Gradient clipping")
DEFINE_float("child_keep_prob", 1.0, "")
DEFINE_float("child_l2_reg", 1e-4, "")
DEFINE_float("child_lr", 0.1, "")
DEFINE_integer("child_lr_dec_every", 100, "")
DEFINE_boolean("child_lr_cosine", False, "Use cosine lr schedule")
DEFINE_float("child_lr_dec_rate", 0.1, "")
DEFINE_float("child_lr_max", None, "for lr schedule")
DEFINE_float("child_lr_min", None, "for lr schedule")
DEFINE_integer("child_lr_T_0", None, "for lr schedule")
DEFINE_integer("child_lr_T_mul", None, "for lr schedule")
DEFINE_integer("child_num_aggregate", None, "")
DEFINE_integer("child_num_replicas", 1, "")
DEFINE_integer("child_out_filters", 24, "")
DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")
DEFINE_string("data_format", "NHWC", "'NHWC' or 'NCWH'")


class DataFormat(object):
  class NCHW(object):
    name = 'NCHW'
    actual = "channels_first"

    @staticmethod
    def child_init(input): return fw.transpose(input, [0, 3, 1, 2])

    @staticmethod
    def child_init_preprocess(input): return fw.transpose(input, [2, 0, 1])

    @staticmethod
    def get_strides(stride): return [1, 1, stride, stride]

    @staticmethod
    def factorized_reduction(input):
      return fw.pad(input, [[0, 0], [0, 0], [0, 1], [0, 1]])[:, :, 1:, 1:]

    @staticmethod
    def concat_axis():
        return 1

    @staticmethod
    def get_N(input): return input.get_shape()[0]

    @staticmethod
    def get_C(input): return input.get_shape()[1]

    @staticmethod
    def get_HW(input): return input.get_shape()[2], input.get_shape()[3]

    @staticmethod
    def set_shape(tensor, h, w, filters) -> None:
      tensor.set_shape([None, filters, h, w])

    @staticmethod
    def enas_layer(inp, branches):
      return fw.reshape(
        fw.concat(branches, axis=1),
        [fw.shape(inp)[0], -1, inp.get_shape()[2], inp.get_shape()[3]])

    @staticmethod
    def fixed_layer(branches): return fw.concat(branches, axis=1)

    @staticmethod
    def pool_branch(input, start, count): return input[:, :, :, start : start + count]

    @staticmethod
    def global_avg_pool(input): return fw.reduce_mean(input, [2, 3])

    @staticmethod
    def micro_enas(input, inp, outputs, filters):
      n = fw.shape(inp)[0]
      h = fw.shape(inp)[2]
      w = fw.shape(inp)[3]
      return fw.reshape(fw.transpose(input, [1, 0, 2, 3, 4]), [n, outputs * filters, h, w])


  class NHWC(object):
    name = 'NHWC'
    actual = "channels_last"

    @staticmethod
    def child_init_preprocess(input): return input

    @staticmethod
    def child_init(input): return input

    @staticmethod
    def get_strides(stride): return [1, stride, stride, 1]

    @staticmethod
    def factorized_reduction(input):
      return fw.pad(input, [[0, 0], [0, 1], [0, 1], [0, 0]])[:, 1:, 1:, :]

    @staticmethod
    def concat_axis():
        return 3

    @staticmethod
    def get_N(input): return input.get_shape()[0]

    @staticmethod
    def get_C(input): return input.get_shape()[3]

    @staticmethod
    def get_HW(input): return input.get_shape()[1], input.get_shape()[2]

    @staticmethod
    def set_shape(tensor, h, w, filters):
      tensor.set_shape([None, h, w, filters])

    @staticmethod
    def enas_layer(_, branches):
      return fw.concat(branches, axis=3)

    @staticmethod
    def fixed_layer(branches):
      return fw.concat(branches, axis=3)

    @staticmethod
    def pool_branch(input, start, count): return input[:, start : start + count, :, :]

    @staticmethod
    def global_avg_pool(input): return fw.reduce_mean(input, [1, 2])

    @staticmethod
    def micro_enas(input, inp, outputs, filters):
      n = fw.shape(inp)[0]
      h = fw.shape(inp)[1]
      w = fw.shape(inp)[2]
      return fw.reshape(fw.transpose(input, [1, 2, 3, 0, 4]), [n, h, w, outputs * filters])


  def __init__(self):
    raise AttributeError("Use factory method `new` instead.")

  @staticmethod
  def new(format_str):
    return {
      "NCHW": DataFormat.NCHW(),
      "NHWC": DataFormat.NHWC()
    }[format_str]


class Child(object):
  def __init__(self,
               images,
               labels,
               eval_batch_size=100,
               clip_mode=None,
               lr_dec_start=0,
               optim_algo=None,
               name="generic_model",
               seed=None,
              ):
    """
    Args:
      lr_dec_every: number of epochs to decay
    """
    FLAGS = fw.FLAGS
    print("-" * 80)
    print("Build model {}".format(name))

    self.weights = fw.WeightRegistry()
    self.num_layers = FLAGS.child_num_layers
    self.cutout_size = FLAGS.child_cutout_size
    self.fixed_arc = FLAGS.child_fixed_arc
    self.out_filters = FLAGS.child_out_filters
    self.batch_size = FLAGS.batch_size

    self.eval_batch_size = eval_batch_size
    self.clip_mode = ClipMode.new(clip_mode, FLAGS.child_grad_bound)
    self.l2_reg = FLAGS.child_l2_reg
    self.keep_prob = FLAGS.child_keep_prob
    self.optim_algo = Optimizer.new(optim_algo, FLAGS.child_sync_replicas, FLAGS.child_num_aggregate, FLAGS.child_num_replicas)
    self.data_format = DataFormat.new(FLAGS.data_format)
    self.name = name
    self.seed = seed

    self.global_step = None
    self.valid_acc = None
    self.test_acc = None
    print("Build data ops")
    with fw.device("/gpu:0"):
      # training data
      self.num_train_examples = np.shape(images["train"])[0]
      self.num_train_batches = (
        self.num_train_examples + self.batch_size - 1) // self.batch_size
      x_train, y_train = fw.shuffle_batch(
        [images["train"], labels["train"]],
        self.batch_size,
        self.seed,
        50000
      )

      self.learning_rate = LearningRate.new(
        FLAGS.child_lr_cosine,
        FLAGS.child_lr,
        lr_dec_start,
        FLAGS.child_lr_dec_every * self.num_train_batches,
        FLAGS.child_lr_dec_rate,
        None,
        FLAGS.child_lr_max,
        FLAGS.child_lr_min,
        FLAGS.child_lr_T_0,
        FLAGS.child_lr_T_mul)

      def _pre_process(x):
        x = fw.random_flip_left_right(fw.random_crop(fw.pad(x, [[4, 4], [4, 4], [0, 0]]), [32, 32, 3], seed=self.seed), seed=self.seed)
        if self.cutout_size is not None:
          start = fw.random_uniform([2], minval=0, maxval=32, dtype=fw.int32)
          x = fw.where(
            fw.equal(
              fw.tile(
                fw.reshape(
                  fw.pad(
                    fw.ones([self.cutout_size, self.cutout_size], dtype=fw.int32),
                    [[self.cutout_size + start[0], 32 - start[0]],
                    [self.cutout_size + start[1], 32 - start[1]]])[self.cutout_size: self.cutout_size + 32, self.cutout_size: self.cutout_size + 32],
                  [32, 32, 1]),
                [1, 1, 3]),
              0),
              x=x,
              y=fw.zeros_like(x))
        x = self.data_format.child_init_preprocess(x)

        return x
      self.x_train = fw.map_fn(_pre_process, x_train, back_prop=False)
      self.y_train = y_train

      # valid data
      self.x_valid, self.y_valid = None, None
      if images["valid"] is not None:
        images["valid_original"] = np.copy(images["valid"])
        labels["valid_original"] = np.copy(labels["valid"])
        images["valid"] = self.data_format.child_init(images["valid"])
        self.num_valid_examples = np.shape(images["valid"])[0]
        self.num_valid_batches = (
          (self.num_valid_examples + self.eval_batch_size - 1)
          // self.eval_batch_size)
        self.x_valid, self.y_valid = fw.batch(
          [images["valid"], labels["valid"]],
          batch_size=self.eval_batch_size,
          capacity=5000)

      # test data
      images["test"] = self.data_format.child_init(images["test"])
      self.num_test_examples = np.shape(images["test"])[0]
      self.num_test_batches = (
        (self.num_test_examples + self.eval_batch_size - 1)
        // self.eval_batch_size)
      self.x_test, self.y_test = fw.batch(
        [images["test"], labels["test"]],
        batch_size=self.eval_batch_size,
        capacity=10000)

    # cache images and labels
    self.images = images
    self.labels = labels

  def eval_once(self, sess, eval_set, feed_dict=None, verbose=False):
    """Expects self.acc and self.global_step to be defined.

    Args:
      sess: fw.Session() or one of its wrap arounds.
      feed_dict: can be used to give more information to sess.run().
      eval_set: "valid" or "test"
    """

    assert self.global_step is not None
    global_step = sess.run(self.global_step)
    print("Eval at {}".format(global_step))

    if eval_set == "valid":
      assert self.x_valid is not None
      assert self.valid_acc is not None
      num_examples = self.num_valid_examples
      num_batches = self.num_valid_batches
      acc_op = self.valid_acc
    elif eval_set == "test":
      assert self.test_acc is not None
      num_examples = self.num_test_examples
      num_batches = self.num_test_batches
      acc_op = self.test_acc
    else:
      raise NotImplementedError("Unknown eval_set '{}'".format(eval_set))

    total_acc = 0
    total_exp = 0
    for batch_id in range(num_batches):
      acc = sess.run(acc_op, feed_dict=feed_dict)
      total_acc += acc
      total_exp += self.eval_batch_size
      if verbose:
        sys.stdout.write("\r{:<5d}/{:>5d}".format(total_acc, total_exp))
    if verbose:
      print("")
    print("{}_accuracy: {:<6.4f}".format(
      eval_set, float(total_acc) / total_exp))

  def _build_train(self):
    print("Build train graph")
    logits = self._model(self.x_train, True)
    log_probs = fw.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_train)
    self.loss = fw.reduce_mean(log_probs)

    self.train_acc = fw.reduce_sum(fw.to_int32(fw.equal(fw.to_int32(fw.argmax(logits, axis=1)), self.y_train)))

    tf_variables = [var
        for var in fw.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print("-" * 80)
    for var in tf_variables:
      print(var)

    self.global_step = fw.Variable(0, dtype=fw.int32, name="global_step")
    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.global_step,
      self.learning_rate,
      clip_mode=self.clip_mode,
      l2_reg=self.l2_reg,
      optim_algo=self.optim_algo)

  def _build_valid(self, model, weights, x, y):
    """ always overridden """
    if x is not None:
      print("-" * 80)
      print("Build valid graph")
      logits = model(weights, x, False, reuse=True)
      predictions = fw.to_int32(fw.argmax(logits, axis=1))
      retval = (
        predictions,
        fw.reduce_sum(fw.to_int32(fw.equal(predictions, y))))
    else:
      retval = (None, None)
    return retval

  def _build_test(self):
    """ always overridden """
    print("-" * 80)
    print("Build test graph")
    logits = self._model(self.x_test, False, reuse=True)
    self.test_preds = fw.to_int32(fw.argmax(logits, axis=1))
    self.test_acc = fw.reduce_sum(fw.to_int32(fw.equal(self.test_preds, self.y_test)))

  def build_valid_rl(self, shuffle=False):
    print("-" * 80)
    print("Build valid graph on shuffled data")
    with fw.device("/gpu:0"):
      # shuffled valid data: for choosing validation model
      if not shuffle:
        self.images["valid_original"] = self.data_format.child_init(self.images["valid_original"])
      x_valid_shuffle, y_valid_shuffle = fw.shuffle_batch(
        [self.images["valid_original"], self.labels["valid_original"]],
        batch_size=self.batch_size,
        capacity=25000,
        enqueue_many=True,
        min_after_dequeue=0,
        num_threads=16,
        seed=self.seed,
        allow_smaller_final_batch=True,
      )

      def _pre_process(x):
        x = fw.pad(x, [[4, 4], [4, 4], [0, 0]])
        x = fw.image.random_crop(x, [32, 32, 3], seed=self.seed)
        x = fw.image.random_flip_left_right(x, seed=self.seed)
        x = self.data_format.child_init_preprocess(x)

        return x

      if shuffle:
        x_valid_shuffle = fw.map_fn(_pre_process, x_valid_shuffle,
                                    back_prop=False)

    logits = self._model(x_valid_shuffle, False, reuse=True)
    valid_shuffle_preds = fw.argmax(logits, axis=1)
    valid_shuffle_preds = fw.to_int32(valid_shuffle_preds)
    self.valid_shuffle_acc = fw.equal(valid_shuffle_preds, y_valid_shuffle)
    self.valid_shuffle_acc = fw.to_int32(self.valid_shuffle_acc)
    self.valid_shuffle_acc = fw.reduce_sum(self.valid_shuffle_acc)

  def _model(self, images, is_training, reuse=None):
    raise NotImplementedError("Abstract method")


  class PathConv(LayeredModel):
    def __init__(self, weights, reuse: bool, scope: str, out_filters: int, is_training, data_format):
      self.layers = [
        lambda x: fw.conv2d(
          x,
          weights.get(
            reuse,
            scope,
            "w",
            [1, 1, data_format.get_C(x), out_filters],
            None),
          [1, 1, 1, 1],
          "SAME",
          data_format=data_format.name),
        lambda x: batch_norm(x, is_training, data_format, weights)]


  class FactorizedReduction(LayeredModel):
    def __init__(self, is_training, data_format, weights):
      self.layers = [
        lambda x: fw.concat(values=x, axis=data_format.concat_axis()),
        lambda x: batch_norm(x, is_training, data_format, weights)]


  class Conv1x1(LayeredModel):
    def __init__(self, weights, reuse: bool, scope: str, inp_c, out_filters, is_training: bool, data_format):
      self.layers = [
        fw.relu,
        lambda x: fw.conv2d(
          x,
          weights.get(reuse, scope, "w", [1, 1, inp_c, out_filters], None),
          [1, 1, 1, 1],
          "SAME",
          data_format=data_format.name),
        lambda x: batch_norm(x, is_training, data_format, weights)]


  class ConvNxN(LayeredModel):
    def __init__(self, weights, reuse: bool, scope: str, filter_size: int, out_filters, data_format, is_training: bool):
      self.layers = [
        fw.relu,
        lambda x: fw.conv2d(
          x,
          weights.get(reuse, scope, "w", [filter_size, filter_size, out_filters, out_filters], None),
          [1, 1, 1, 1],
          "SAME",
          data_format=data_format.name),
        lambda x: batch_norm(x, is_training, data_format, weights)]


  class StemConv(LayeredModel):
    def __init__(self, weights, reuse, scope, out_filters, is_training, data_format):
      w = weights.get(reuse, scope, "w", [3, 3, 3, out_filters], None)
      self.layers = [
        lambda x: fw.conv2d(
          x,
          w,
          [1, 1, 1, 1],
          "SAME",
          data_format=data_format.name),
        lambda x: batch_norm(x, is_training, data_format, weights)]


  class InputConv(LayeredModel):
    # TODO Eliminate duplication
    def __init__(self, weights, reuse, scope, hw, out_filters, is_training: bool, data_format):
      self.layers = [
        lambda x: fw.conv2d(
          x,
          weights.get(
            reuse,
            scope,
            "w",
            [hw, hw, data_format.get_C(x), out_filters],
            None),
          [1, 1, 1, 1],
          'SAME',
          data_format=data_format.name),
        lambda x: batch_norm(x, is_training, data_format, weights),
        fw.relu]
