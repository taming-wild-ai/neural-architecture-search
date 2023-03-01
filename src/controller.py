import src.framework as fw

from src.utils import DEFINE_boolean, DEFINE_float, DEFINE_integer, DEFINE_string, LearningRate

DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_entropy_weight", None, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_lr", 1e-3, "")
DEFINE_float("controller_tanh_constant", None, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_boolean("controller_use_critic", False, "")
DEFINE_string("search_for", "macro", "Must be [macro|micro]")
DEFINE_boolean("controller_sync_replicas", False, "To sync or not to sync.")
DEFINE_integer("controller_num_aggregate", 1, "")
DEFINE_integer("controller_num_replicas", 1, "")


class Controller(object):
  def __init__(self):
    FLAGS = fw.FLAGS
    self.search_for = FLAGS.search_for
    self.tanh_constant = FLAGS.controller_tanh_constant
    self.temperature = FLAGS.controller_temperature
    self.learning_rate = LearningRate.new(
      False,
      FLAGS.controller_lr,
      0,
      100,
      0.9,
      None,
      None,
      None,
      None,
      None)
    self.lr_init = FLAGS.controller_lr
    self.l2_reg = FLAGS.controller_l2_reg
    self.entropy_weight = FLAGS.controller_entropy_weight
    self.bl_dec = FLAGS.controller_bl_dec
    self.use_critic = FLAGS.controller_use_critic
    self.sync_replicas = FLAGS.controller_sync_replicas
    self.num_aggregate = FLAGS.controller_num_aggregate
    self.num_replicas = FLAGS.controller_num_replicas

  def _build_sample(self):
    raise NotImplementedError("Abstract method.")

  def _build_greedy(self):
    raise NotImplementedError("Abstract method.")

  def _build_trainer(self):
    raise NotImplementedError("Abstract method.")
