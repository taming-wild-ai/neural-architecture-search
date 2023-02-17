import tensorflow as tf
from functools import partial

# TensorFlow 2
Graph = tf.Graph
IndexedSlices = tf.IndexedSlices
TensorArray = tf.TensorArray
Variable = partial(tf.Variable, trainable=False)
add_n = tf.add_n
argmax = tf.argmax
avg_pool = tf.nn.avg_pool
bool = tf.bool
boolean_mask = tf.boolean_mask
case = tf.case
clip_by_norm = tf.clip_by_norm
clip_by_global_norm = tf.clip_by_global_norm
concat = tf.concat
cond = tf.cond
constant = tf.constant
control_dependencies = tf.control_dependencies
conv2d = tf.nn.conv2d
cos = tf.cos
device = tf.device
divide = tf.math.divide
dropout = tf.nn.dropout
embedding_lookup = tf.nn.embedding_lookup
equal = tf.equal
exp = tf.exp
fill = tf.fill
float32 = tf.float32
floor = tf.floor
gather = tf.gather
global_norm = tf.linalg.global_norm
gradients = tf.gradients
greater_equal = tf.greater_equal
identity = tf.identity
int32 = tf.int32
int64 = tf.int64
less_equal = tf.less_equal
log = tf.math.log
logical_and = tf.logical_and
logical_or = tf.logical_or
map_fn = tf.map_fn
matmul = tf.matmul
maximum = tf.maximum
minimum = tf.minimum
one_hot = tf.one_hot
pad = tf.pad
random_uniform = tf.random.uniform
random_uniform_initializer = tf.random_uniform_initializer
range = tf.range
reduce_mean = tf.reduce_mean
reduce_sum = tf.reduce_sum
relu = tf.nn.relu
reshape = tf.reshape
separable_conv2d = tf.nn.separable_conv2d
shape = tf.shape
sigmoid = tf.sigmoid
size = tf.size
sparse_softmax_cross_entropy_with_logits = tf.nn.sparse_softmax_cross_entropy_with_logits
split = tf.split
sqrt = tf.sqrt
stack = tf.stack
stop_gradient = tf.stop_gradient
tanh = tf.tanh
transpose = tf.transpose
where = tf.where
while_loop = tf.while_loop
zeros = tf.zeros
zeros_like = tf.zeros_like

# TensorFlow 1 Compatibility
ConfigProto = partial(tf.compat.v1.ConfigProto, allow_soft_placement=True)
Constant = partial(tf.compat.v1.keras.initializers.Constant, dtype=tf.float32)
DEFINE_boolean = tf.compat.v1.app.flags.DEFINE_boolean
DEFINE_float = tf.compat.v1.app.flags.DEFINE_float
DEFINE_integer = tf.compat.v1.app.flags.DEFINE_integer
DEFINE_string = tf.compat.v1.app.flags.DEFINE_string
FLAGS = tf.compat.v1.app.flags.FLAGS
Hook = tf.compat.v1.train.CheckpointSaverHook
Saver = partial(tf.compat.v1.train.Saver, max_to_keep=2)
Session = tf.compat.v1.train.SingularMonitoredSession
assign = partial(tf.compat.v1.assign, use_locking=True)
assign_sub = tf.compat.v1.assign_sub
avg_pool2d = tf.compat.v1.layers.average_pooling2d
batch = partial(
    tf.compat.v1.train.batch,
    enqueue_many=True,
    num_threads=1,
    allow_smaller_final_batch=True)
exp_decay = tf.compat.v1.train.exponential_decay
flags = tf.compat.v1.flags
fused_batch_norm = tf.compat.v1.nn.fused_batch_norm
get_or_create_global_step = tf.compat.v1.train.get_or_create_global_step
get_variable = tf.compat.v1.get_variable
create_weight = partial(
    tf.compat.v1.get_variable,
    initializer=tf.keras.initializers.he_normal(seed=None),
    trainable=True)
max_pool2d = tf.compat.v1.layers.max_pooling2d
multinomial = tf.compat.v1.multinomial
ones = partial(tf.compat.v1.keras.initializers.ones, dtype=tf.float32)
run = tf.compat.v1.app.run
scatter_sub = partial(tf.compat.v1.scatter_sub, use_locking=True)
zeros = partial(tf.compat.v1.keras.initializers.zeros, dtype=tf.float32)

def shuffle_batch(data, batch_size, seed, capacity=25000):
    return tf.compat.v1.train.shuffle_batch(
        data,
        batch_size=batch_size,
        capacity=capacity,
        enqueue_many=True,
        min_after_dequeue=0,
        num_threads=16,
        seed=seed,
        allow_smaller_final_batch=True)

to_float = tf.compat.v1.to_float
to_int32 = tf.compat.v1.to_int32
trainable_variables = tf.compat.v1.trainable_variables
variable_scope = tf.compat.v1.variable_scope


class Optimizer(object):
    @staticmethod
    def Momentum(learning_rate):
        return tf.compat.v1.train.MomentumOptimizer(
            learning_rate,
            0.9,
            use_locking=True,
            use_nesterov=True)

    @staticmethod
    def SGD(learning_rate):
        return tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate,
            use_locking=True)

    @staticmethod
    def Adam(learning_rate):
        return tf.compat.v1.train.AdamOptimizer(
            learning_rate,
            beta1=0.0,
            epsilon=1e-3,
            use_locking=True)

    @staticmethod
    def SyncReplicas(opt, num_aggregate, num_replicas):
        return tf.compat.v1.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=num_aggregate,
            total_num_replicas=num_replicas,
            use_locking=True)

    @staticmethod
    def MovingAverage(opt, moving_average):
        return tf.contrib.opt.MovingAverageOptimizer(
            opt,
            average_decay=moving_average)