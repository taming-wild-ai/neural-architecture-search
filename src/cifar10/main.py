import os
import pickle as pickle
import shutil
import sys
import time

import numpy as np
from absl import flags, app
import src.framework as fw

from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import print_user_flags

from src.cifar10.data_utils import read_data
from src.cifar10.macro_controller import MacroController
from src.cifar10.macro_child import MacroChild

from src.cifar10.micro_controller import MicroController
from src.cifar10.micro_child import MicroChild

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")

DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_train_every", 2,
               "train the controller after this number of epochs")
DEFINE_boolean("controller_training", True, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

def get_ops(images, labels):
  """
  Args:
    images: dict with keys {"train", "valid", "test"}.
    labels: dict with keys {"train", "valid", "test"}.
  """
  FLAGS = flags.FLAGS
  assert FLAGS.search_for is not None, "Please specify --search_for"

  if FLAGS.search_for == "micro":
    ControllerClass = MicroController
    ChildClass = MicroChild
  else:
    ControllerClass = MacroController
    ChildClass = MacroChild

  child_model = ChildClass(
    images,
    labels,
    clip_mode="norm",
    optim_algo="momentum")

  if FLAGS.child_fixed_arc is None:
    controller_model = ControllerClass(
      lstm_size=64,
      lstm_num_layers=1,
      lstm_keep_prob=1.0,
      lr_dec_start=0,
      lr_dec_every=1000000,  # never decrease learning rate
      optim_algo="adam")

    child_train_op, child_lr, child_optimizer = child_model.connect_controller(controller_model)
    dataset_valid_shuffle = child_model.ValidationRLShuffle(
        child_model,
        False)(
            child_model.images['valid_original'],
            child_model.labels['valid_original'])
    controller_train_op, controller_lr, controller_optimizer = controller_model.build_trainer(
        child_model,
        child_model.ValidationRL())

    controller_ops = {
      "train_step": controller_model.train_step, # tf.Variable
      # MacroController.generate_sample_arc() -> sample_arc
      # MicroController.generate_sample_arc() -> normal_arc, reduce_arc
      "generate_sample_arc": controller_model.generate_sample_arc,
      'loss': controller_model.loss, # Controller.loss(child_logits, y_valid_shuffle) -> loss
      "train_op": controller_train_op, # Controller.train_op(loss, vars) -> iteration_num
      "lr": controller_lr, # learning_rate() -> learning_rate
      'trainable_variables': controller_model.trainable_variables(),
      "valid_acc": controller_model.valid_acc, # Controller.valid_acc(child_logits, y_valid_shuffle) -> valid_acc
      "optimizer": controller_optimizer, # framework.Optimizer
      "baseline": controller_model.baseline, # tf.Variable
      "entropy": lambda: controller_model.current_entropy,
    }
    child_valid_rl_model = child_model.valid_rl_model
  else:
    assert not FLAGS.controller_training, (
      "--child_fixed_arc is given, cannot train controller")
    child_train_op, child_lr, child_optimizer = child_model.connect_controller(None)
    dataset_valid_shuffle = None
    controller_ops = None
    child_valid_rl_model = None

  return {
    "child": {
      'generate_train_losses': child_model.generate_train_losses, # Child.generate_train_losses(images, labels) -> logits, loss, train_loss, train_acc
      'test_model': child_model.test_model, # Child.test_model(images) -> child_logits
      'validation_model': child_model.valid_model, # Child.valid_model(images) -> child_logits
      'validation_rl_model': child_valid_rl_model, # Child.valid_rl_model(images) -> child_valid_rl_logits
      'global_step': child_model.global_step, # tf.Variable
      'dataset': child_model.dataset,
      'dataset_test': child_model.dataset_test,
      'dataset_valid_shuffle': dataset_valid_shuffle, # tf.Dataset
      "loss": child_model.loss, # Child.loss(child_logits) -> loss
      # MacroChild.loss(child_logits) -> train_loss
      # MicroChild.loss(child_logits, child_aux_logits) -> train_loss
      "train_loss": child_model.train_loss,
      "train_op": child_train_op, # Child.train_op(train_loss, vars) -> iteration_num
      "lr": child_lr, # Child.learning_rate() -> learning_rate
      'trainable_variables': child_model.trainable_variables(),
      "optimizer": child_optimizer, # framework.Optimizer
      "num_train_batches": child_model.num_train_batches,
    },
    "controller": controller_ops,
    "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
    "eval_func": child_model.eval_once,
    "num_train_batches": child_model.num_train_batches,
  }


def train():
  FLAGS = flags.FLAGS
  if FLAGS.child_fixed_arc is None:
    images, labels = read_data(FLAGS.data_path)
  else:
    images, labels = read_data(FLAGS.data_path, num_valids=0)
  ops = get_ops(images, labels)
  if ops['controller']:
      saver = fw.Saver(var_list=ops['child']['trainable_variables'] + ops['controller']['trainable_variables'])
  else:
      saver = fw.Saver(var_list=ops['child']['trainable_variables'])
  hooks = [
      fw.Hook(
          FLAGS.output_dir,
          save_steps=ops["child"]["num_train_batches"],
          saver=saver)]
  if FLAGS.child_sync_replicas:
    sync_replicas_hook = ops["child"]["optimizer"].make_session_run_hook(True)
    hooks.append(sync_replicas_hook)
  # if FLAGS.controller_training and FLAGS.controller_sync_replicas:
  #   sync_replicas_hook = ops["controller"]["optimizer"].make_session_run_hook(True)
  #   hooks.append(sync_replicas_hook)
  batch_iterator = None

  print(("-" * 80))
  print("Starting session")
  config = fw.ConfigProto()
  start_time = time.time()

  @fw.function
  def child_train_op(images, labels):
      with fw.GradientTape() as tape:
          child_train_logits, child_loss, child_train_loss, child_train_acc = ops['child']['generate_train_losses'](images, labels)
          child_grad_norm, child_grad_norm_list, _ = ops['child']['train_op'](child_loss, ops['child']['trainable_variables'], tape)
      return child_train_logits, child_loss, child_train_acc, child_grad_norm

  @fw.function
  def controller_train_op(child_train_logits, labels):
      with fw.GradientTape() as tape:
          ops["controller"]["generate_sample_arc"]()
          controller_loss = ops['controller']['loss'](child_train_logits, labels)
          gn, gn_list, _ = ops['controller']['train_op'](controller_loss, ops['controller']['trainable_variables'], tape)
          return controller_loss, gn

  while True:
      if batch_iterator is None:
          batch_iterator = ops['child']['dataset'].as_numpy_iterator()
      try:
          images, labels = batch_iterator.__next__()
      except StopIteration:
          batch_iterator = ops['child']['dataset'].as_numpy_iterator()
          images, labels = batch_iterator.__next__()
      child_train_logits, child_loss, child_train_acc, child_grad_norm = child_train_op(images, labels)
      child_lr = ops['child']['lr']()
      global_step = ops["child"]["global_step"].value()

      if FLAGS.child_sync_replicas:
          actual_step = global_step * FLAGS.child_num_aggregate
      else:
          actual_step = global_step
      epoch = actual_step // ops["num_train_batches"]
      curr_time = time.time()
      if global_step % FLAGS.log_every == 0:
          log_string = ""
          log_string += "epoch={:<6d}".format(epoch)
          log_string += "ch_step={:<6d}".format(global_step)
          log_string += " loss={:<8.6f}".format(child_loss)
          log_string += " lr={:<8.4f}".format(child_lr)
          log_string += " |g|={:<8.4f}".format(child_grad_norm)
          log_string += " tr_acc={:<3d}/{:>3d}".format(
              child_train_acc, FLAGS.batch_size)
          log_string += " mins={:<10.2f}".format(
              float(curr_time - start_time) / 60)
          print(log_string)

      if actual_step % ops["eval_every"] == 0:
        test_images_batch, test_labels_batch = ops['child']['dataset_test'].as_numpy_iterator().__next__()
        child_test_logits = ops['child']['test_model'](test_images_batch)
        if (FLAGS.controller_training and
            epoch % FLAGS.controller_train_every == 0):
          print(("Epoch {}: Training controller".format(epoch)))
          images_batch, labels_batch = ops['child']['dataset_valid_shuffle'].as_numpy_iterator().__next__()
          child_valid_rl_logits = ops['child']['validation_rl_model'](images_batch)
          for ct_step in range(FLAGS.controller_train_steps *
                                FLAGS.controller_num_aggregate):
            controller_valid_acc = ops['controller']['valid_acc'](child_valid_rl_logits, labels_batch)
            controller_loss, gn = controller_train_op(child_train_logits, labels)
            controller_entropy = 0
            lr = ops["controller"]["lr"]()
            bl = ops["controller"]["baseline"]

            if ct_step % FLAGS.log_every == 0:
              curr_time = time.time()
              log_string = ""
              log_string += "ctrl_step={:<6d}".format(ops["controller"]["train_step"].value())
              log_string += " loss={:<7.3f}".format(controller_loss)
              log_string += " ent={:<5.2f}".format(controller_entropy)
              log_string += " lr={:<6.4f}".format(lr)
              log_string += " |g|={:<8.4f}".format(gn)
              log_string += " acc={:<6.4f}".format(controller_valid_acc)
              log_string += f' bl={bl.value()}'
              log_string += " mins={:<.2f}".format(
                  float(curr_time - start_time) / 60)
              print(log_string)

          print("Here are 10 architectures")
          for _ in range(10):
            arc = ops["controller"]["generate_sample_arc"]()
            child_valid_rl_logits = ops['child']['validation_rl_model'](images_batch)
            acc = ops["controller"]["valid_acc"](child_valid_rl_logits, labels_batch)
            if FLAGS.search_for == "micro":
              normal_arc, reduce_arc = arc
              print((np.reshape(normal_arc, [-1])))
              print((np.reshape(reduce_arc, [-1])))
            else:
              start = 0
              for layer_id in range(FLAGS.child_num_layers):
                if FLAGS.controller_search_whole_channels:
                  end = start + 1 + layer_id
                else:
                  end = start + 2 * FLAGS.child_num_branches + layer_id
                print((np.reshape(arc[start: end], [-1])))
                start = end
            print(("val_acc={:<6.4f}".format(acc)))
            print(("-" * 80))

        print(("Epoch {}: Eval".format(epoch)))
        if FLAGS.child_fixed_arc is None:
          ops["eval_func"]("valid", child_valid_rl_logits, labels_batch)
        ops["eval_func"]("test", child_test_logits, test_labels_batch)

      if epoch >= FLAGS.num_epochs:
        break


def main(_):
  FLAGS = flags.FLAGS
  print(("-" * 80))
  if not os.path.isdir(FLAGS.output_dir):
    print(("Path {} does not exist. Creating.".format(FLAGS.output_dir)))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print(("Path {} exists. Remove and remake.".format(FLAGS.output_dir)))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print(("-" * 80))
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print(("Logging to {}".format(log_file)))
  sys.stdout = Logger(log_file)

  print_user_flags()
  train()


if __name__ == "__main__":
    app.run(main)
