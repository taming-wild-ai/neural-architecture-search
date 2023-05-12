import os
import pickle as pickle
import shutil
import sys
import time

import numpy as np
import src.framework as fw

from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
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
  FLAGS = fw.FLAGS
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

    child_train_op, child_lr, child_grad_norm, child_optimizer = child_model.connect_controller(controller_model)
    shuffle = child_model.ValidationRLShuffle(child_model, False)
    vrl = child_model.ValidationRL()
    x_valid_shuffle, y_valid_shuffle = shuffle(child_model.images['valid_original'], child_model.labels['valid_original'])
    controller_train_op, controller_lr, controller_grad_norm, controller_optimizer = controller_model.build_trainer(child_model, vrl)

    controller_ops = {
      "train_step": controller_model.train_step,
      "loss": controller_model.loss,
      "train_op": controller_train_op,
      "lr": controller_lr,
      "grad_norm": controller_grad_norm,
      "valid_acc": controller_model.valid_acc,
      "optimizer": controller_optimizer,
      "baseline": controller_model.baseline,
      "entropy": controller_model.sample_entropy,
      "sample_arc": controller_model.sample_arc,
      "skip_rate": controller_model.skip_rate,
    }
  else:
    assert not FLAGS.controller_training, (
      "--child_fixed_arc is given, cannot train controller")
    child_train_op, child_lr, child_grad_norm, child_optimizer = child_model.connect_controller(None)
    y_valid_shuffle = None
    controller_ops = None

  return {
    "child": {
      "images": child_model.x_train,
      'y_valid_shuffle': y_valid_shuffle,
      "model": ChildClass.Model(child_model, True),
      "global_step": child_model.global_step,
      "loss": child_model.loss,
      "train_op": child_train_op,
      "lr": child_lr,
      "grad_norm": child_grad_norm,
      "train_acc": child_model.train_acc,
      "optimizer": child_optimizer,
      "num_train_batches": child_model.num_train_batches,
    },
    "controller": controller_ops,
    "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
    "eval_func": child_model.eval_once,
    "num_train_batches": child_model.num_train_batches,
  }


def train():
  FLAGS = fw.FLAGS
  if FLAGS.child_fixed_arc is None:
    images, labels = read_data(FLAGS.data_path)
  else:
    images, labels = read_data(FLAGS.data_path, num_valids=0)

  g = fw.Graph()
  with g.as_default():
    ops = get_ops(images, labels)
    hooks = [fw.Hook(FLAGS.output_dir, save_steps=ops["child"]["num_train_batches"], saver=fw.Saver())]
    if FLAGS.child_sync_replicas:
      sync_replicas_hook = ops["child"]["optimizer"].make_session_run_hook(True)
      hooks.append(sync_replicas_hook)
    if FLAGS.controller_training and FLAGS.controller_sync_replicas:
      sync_replicas_hook = ops["controller"]["optimizer"].make_session_run_hook(True)
      hooks.append(sync_replicas_hook)

    print(("-" * 80))
    print("Starting session")
    with fw.Session(
      config=fw.ConfigProto(),
      hooks=hooks,
      checkpoint_dir=FLAGS.output_dir) as sess:
        start_time = time.time()
        child_logits_graph =     ops['child']['model'](ops['child']['images'])
        child_loss_graph =       ops['child']['loss'](child_logits_graph)
        child_aux_logits =       ops['child']['model']
        child_train_loss_graph = ops['child']['model'].child.train_loss(child_logits_graph, child_aux_logits)
        child_lr =               ops['child']['lr']
        child_grad_norm_graph =  ops['child']['grad_norm'](child_loss_graph, ops['child']['model'].child.tf_variables())
        child_train_acc_graph =  ops['child']['train_acc'](child_logits_graph)
        child_train_op_graph =   ops['child']['train_op'](child_train_loss_graph, ops['child']['model'].child.tf_variables())
        controller_loss_graph =      ops["controller"]["loss"](child_logits_graph, ops['child']['y_valid_shuffle'])
        controller_entropy =         ops["controller"]["entropy"]
        controller_lr =              ops["controller"]["lr"]
        controller_grad_norm_graph = ops["controller"]["grad_norm"](controller_loss_graph, ops['controller']['model'].tf_variables())
        controller_valid_acc_graph = ops["controller"]["valid_acc"](child_logits_graph, ops['child']['y_valid_shuffle'])
        controller_baseline =        ops["controller"]["baseline"]
        controller_skip_rate =       ops["controller"]["skip_rate"]
        controller_train_op_graph =  ops["controller"]["train_op"](controller_loss_graph, ops['controller']['model'].tf_variables())
        while True:
          logits, loss, lr, gn, tr_acc, _ = sess.run([
            child_logits_graph,
            child_loss_graph,
            child_lr,
            child_grad_norm_graph,
            child_train_acc_graph,
            child_train_op_graph,
          ])
          global_step = sess.run(ops["child"]["global_step"])

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
            log_string += " loss={:<8.6f}".format(loss)
            log_string += " lr={:<8.4f}".format(lr)
            log_string += " |g|={:<8.4f}".format(gn)
            log_string += " tr_acc={:<3d}/{:>3d}".format(
                tr_acc, FLAGS.batch_size)
            log_string += " mins={:<10.2f}".format(
                float(curr_time - start_time) / 60)
            print(log_string)

          if actual_step % ops["eval_every"] == 0:
            if (FLAGS.controller_training and
                epoch % FLAGS.controller_train_every == 0):
              print(("Epoch {}: Training controller".format(epoch)))
              for ct_step in range(FLAGS.controller_train_steps *
                                    FLAGS.controller_num_aggregate):

                loss, entropy, lr, gn, val_acc, bl, skip, _ = sess.run([
                  controller_loss_graph,
                  controller_entropy,
                  controller_lr,
                  controller_grad_norm_graph,
                  controller_valid_acc_graph,
                  controller_baseline,
                  controller_skip_rate,
                  controller_train_op_graph,
                ])

                if ct_step % FLAGS.log_every == 0:
                  curr_time = time.time()
                  log_string = ""
                  log_string += "ctrl_step={:<6d}".format(sess.run(ops["controller"]["train_step"]))
                  log_string += " loss={:<7.3f}".format(loss)
                  log_string += " ent={:<5.2f}".format(entropy)
                  log_string += " lr={:<6.4f}".format(lr)
                  log_string += " |g|={:<8.4f}".format(gn)
                  log_string += " acc={:<6.4f}".format(val_acc)
                  log_string += " bl={:<5.2f}".format(bl)
                  log_string += " mins={:<.2f}".format(
                      float(curr_time - start_time) / 60)
                  print(log_string)

              print("Here are 10 architectures")
              for _ in range(10):
                arc, acc = sess.run([
                  ops["controller"]["sample_arc"],
                  ops["controller"]["valid_acc"],
                ])
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
              ops["eval_func"](sess, "valid")
            ops["eval_func"](sess, "test")

          if epoch >= FLAGS.num_epochs:
            break


def main(_):
  FLAGS = fw.FLAGS
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

  utils.print_user_flags()
  train()


if __name__ == "__main__":
  fw.run()
