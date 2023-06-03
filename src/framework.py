import sys
import tensorflow as tf

from collections import defaultdict
from functools import partial


class NameScope(object):
  singleton = None

  @staticmethod
  def name_scope(name=None):
    if NameScope.singleton is None:
      NameScope.singleton = NameScope()
    NameScope.singleton.push(name)
    return NameScope.singleton

  def __init__(self):
    self.stack = []

  def __enter__(self):
    return self.current()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.stack.pop()

  def current(self):
    if len(self.stack) > 0:
      return '/'.join(self.stack) + '/'
    else:
      return ''

  def push(self, name):
    if name:
      self.stack.append(name)


name_scope = NameScope.name_scope


class WeightRegistry(object):
  @staticmethod
  def factory():
    return defaultdict(WeightRegistry.factory)

  def __init__(self):
    self.weight_map = defaultdict(WeightRegistry.factory)
    self.default_initializer = tf.initializers.variance_scaling()

  def create_weight(self, scope, name, shape, initializer=None, trainable=True):
    if initializer is None:
      initializer = self.default_initializer
    return Variable(initializer(shape, tf.float32), name=name, trainable=trainable, import_scope=scope)

  def get(self, reuse: bool, scope, name, shape, initializer, trainable=True):
    if not reuse:
      self.weight_map[scope + name] = self.create_weight(scope, name, shape, initializer=initializer, trainable=trainable)
    return self.weight_map[scope + name]


# TensorFlow 2
Graph = tf.Graph
Hook = tf.estimator.CheckpointSaverHook
IndexedSlices = tf.IndexedSlices
TensorArray = tf.TensorArray

def Variable(initial, name="Variable", trainable=False, dtype=None, import_scope=None):
  if import_scope is None:
    import_scope = name_scope().current()
  retval = tf.Variable(initial, name=import_scope + name, trainable=trainable, dtype=dtype)
  return retval

add_n = tf.add_n
argmax = tf.argmax
avg_pool = tf.nn.avg_pool
avg_pool2d = tf.keras.layers.AveragePooling2D
bool = tf.bool
boolean_mask = tf.boolean_mask
case = tf.case
clip_by_norm = tf.clip_by_norm
clip_by_global_norm = tf.clip_by_global_norm

def concat(values, axis, name=None):
  if name is None:
    name = name_scope().current() + "concat"
  return tf.concat(values, axis, name)

cond = tf.cond
constant = tf.constant
constant_initializer = tf.constant_initializer
control_dependencies = tf.control_dependencies
conv2d = tf.nn.conv2d
cos = tf.cos

def Dataset(images, labels):
   return tf.data.Dataset.from_tensor_slices((images, labels))

device = tf.device
divide = tf.math.divide
dropout = tf.nn.dropout
embedding_lookup = tf.nn.embedding_lookup
equal = tf.equal
executing_eagerly = tf.compat.v1.executing_eagerly
exp = tf.exp
exp_decay = tf.keras.optimizers.schedules.ExponentialDecay
fill = tf.fill
float32 = tf.float32
floor = tf.floor
function = tf.function
fused_batch_norm = tf.raw_ops.FusedBatchNorm
gather = tf.gather

def get_variable(name, shape, initializer=None):
    assert not tf.compat.v1.get_variable_scope().reuse
    if initializer is None:
        return tf.Variable(shape=shape, name=name)
    else:
        return tf.Variable(initializer(shape), name=name)

global_norm = tf.linalg.global_norm
gradients = tf.gradients
greater_equal = tf.greater_equal
identity = tf.identity
int32 = tf.int32
int64 = tf.int64
less = tf.less
less_equal = tf.less_equal
log = tf.math.log
logical_and = tf.logical_and
logical_or = tf.logical_or

def map_fn(fn, elems):
   return tf.nest.map_structure(tf.stop_gradient, tf.map_fn(fn, elems))

matmul = tf.matmul
maximum = tf.maximum
max_pool2d = tf.keras.layers.MaxPooling2D
minimum = tf.minimum
multinomial = tf.random.categorical
one_hot = tf.one_hot
ones_init = tf.ones_initializer
pad = tf.pad
random_crop = tf.image.random_crop
random_flip_left_right = tf.image.random_flip_left_right
random_uniform = tf.random.uniform
random_uniform_initializer = tf.random_uniform_initializer
range = tf.range
reduce_mean = tf.reduce_mean
reduce_sum = tf.reduce_sum
relu = tf.nn.relu

def reshape(tensor, shape, name=None):
  if name is None:
    name = name_scope().current() + "Reshape"
  return tf.reshape(tensor, shape, name)

scatter_sub = tf.tensor_scatter_nd_sub
separable_conv2d = tf.nn.separable_conv2d
shape = tf.shape

def shuffle_batch(data, batch_size, seed, capacity=25000):
    return tf.data.Dataset.from_tensor_slices(data).shuffle(capacity, seed).batch(batch_size)

sigmoid = tf.sigmoid
size = tf.size
softmax = tf.nn.softmax
softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits
sparse_softmax_cross_entropy_with_logits = tf.nn.sparse_softmax_cross_entropy_with_logits
split = tf.split
sqrt = tf.sqrt
stack = tf.stack
stop_gradient = tf.stop_gradient
tanh = tf.tanh
to_float = lambda x: tf.cast(x, tf.float32)
to_int32 = lambda x: tf.cast(x, tf.int32)
transpose = tf.transpose
where = tf.where
while_loop = tf.while_loop
zeros = tf.zeros
zeros_init = tf.zeros_initializer
zeros_like = tf.zeros_like

# TensorFlow 1 Compatibility
ConfigProto = partial(tf.compat.v1.ConfigProto, allow_soft_placement=True)
Saver = partial(tf.compat.v1.train.Saver, max_to_keep=2)

class Optimizer(object):
    @staticmethod
    def Momentum(learning_rate):
        return tf.keras.optimizers.experimental.SGD(
            learning_rate,
            0.9,
            True)

    @staticmethod
    def SGD(learning_rate):
        return tf.keras.optimizers.experimental.SGD(learning_rate)

    @staticmethod
    def Adam(learning_rate):
        return tf.keras.optimizers.Adam(
            learning_rate,
            beta_1=0.0,
            epsilon=1e-3)

    @staticmethod
    def SyncReplicas(opt, num_aggregate, num_replicas):
        return tf.compat.v1.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=num_aggregate,
            total_num_replicas=num_replicas,
            use_locking=True)
