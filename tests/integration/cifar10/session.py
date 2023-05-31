import sys
import unittest
from absl import flags
import tensorflow as tf

from src.cifar10.main import get_ops, read_data
import src.framework as fw


class RestoreFLAGS:
    flags.FLAGS(sys.argv) # Need to parse flags before accessing them

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.old_values = {}

    def __enter__(self):
        for key, value in self.kwargs.items():
            self.old_values[key] = flags.FLAGS.__getattr__(key)
            flags.FLAGS.__setattr__(key, value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.old_values.items():
            flags.FLAGS.__setattr__(key, value)


class TestSession(unittest.TestCase):
    def test_micro_search_loss(self):
        micro_search_flags = {
            "data_format": "NCHW",
            "search_for": "micro",
            "data_path": "data/cifar10",
            "output_dir": "outputs",
            "batch_size": 160,
            "num_epochs": 150,
            "log_every": 50,
            "eval_every_epochs": 1,
            "child_use_aux_heads": True,
            "child_num_layers": 6,
            "child_out_filters": 20,
            "child_l2_reg": 1e-4,
            "child_num_branches": 5,
            "child_num_cells": 5,
            "child_keep_prob": 0.90,
            "child_drop_path_keep_prob": 0.60,
            "child_lr_cosine": True,
            "child_lr_max": 0.05,
            "child_lr_min": 0.0005,
            "child_lr_T_0": 10,
            "child_lr_T_mul": 2,
            "controller_training": True,
            "controller_search_whole_channels": True,
            "controller_entropy_weight": 0.0001,
            "controller_train_every": 1,
            "controller_sync_replicas": True,
            "controller_num_aggregate": 10,
            "controller_train_steps": 50,
            "controller_lr": 0.0035,
            "controller_tanh_constant": 1.1,
            "controller_op_tanh_reduce": 2.5}
        with RestoreFLAGS(**micro_search_flags):
            images, labels = read_data(flags.FLAGS.data_path)
            with tf.Graph().as_default():
                ops = get_ops(images, labels)
                logits_graph =     ops['child']['model'](ops['child']['images'])
                loss_graph =       ops["child"]["loss"](logits_graph)
                aux_logits =       ops['child']['model'].child.aux_logits
                train_loss_graph = ops['child']['model'].child.train_loss(logits_graph, aux_logits)
                lr_graph =         ops["child"]["lr"]
                grad_norm_graph =  ops["child"]["grad_norm"](loss_graph, ops['child']['model'].child.tf_variables())
                train_acc_graph =  ops["child"]["train_acc"](logits_graph)
                train_op_graph =   ops["child"]["train_op"](train_loss_graph, ops['child']['model'].child.tf_variables())
                with fw.Session(config=fw.ConfigProto()) as sess:
                    logits, loss, lr, gn, tr_acc, _ = sess.run([
                        logits_graph,
                        loss_graph,
                        lr_graph,
                        grad_norm_graph,
                        train_acc_graph,
                        train_op_graph
                    ])
                    self.assertLess(loss, 4.0)
                    self.assertLess(lr, 0.06)
                    self.assertLess(tr_acc, 120)
                    self.assertLess(gn, 25.0)

    def test_micro_final_loss(self):
        micro_final_flags = {
            "data_format": "NCHW",
            "search_for": "micro",
            "data_path": "data/cifar10",
            "output_dir": "outputs",
            "batch_size": 144,
            "num_epochs": 630,
            "log_every": 50,
            "eval_every_epochs": 1,
            "child_fixed_arc": "0 2 0 0 0 4 0 1 0 4 1 1 1 0 0 1 0 2 1 1 1 0 1 0 0 3 0 2 1 1 3 1 1 0 0 4 0 3 1 1",
            "child_use_aux_heads": True,
            "child_num_layers": 15,
            "child_out_filters": 36,
            "child_l2_reg": 2e-4,
            "child_num_branches": 5,
            "child_num_cells": 5,
            "child_keep_prob": 0.80,
            "child_drop_path_keep_prob": 0.60,
            "child_lr_cosine": True,
            "child_lr_max": 0.05,
            "child_lr_min": 0.0001,
            "child_lr_T_0": 10,
            "child_lr_T_mul": 2,
            "controller_training": False,
            "controller_search_whole_channels": True,
            "controller_entropy_weight": 0.0001,
            "controller_train_every": 1,
            "controller_sync_replicas": True,
            "controller_num_aggregate": 10,
            "controller_train_steps": 50,
            "controller_lr": 0.001,
            "controller_tanh_constant": 1.5,
            "controller_op_tanh_reduce": 2.5}
        with RestoreFLAGS(**micro_final_flags):
            images, labels = read_data(flags.FLAGS.data_path)
            with tf.Graph().as_default():
                ops = get_ops(images, labels)
                logits_graph =     ops['child']['model'](ops['child']['images'])
                loss_graph =       ops["child"]["loss"](logits_graph)
                aux_logits =       ops['child']['model'].child.aux_logits
                train_loss_graph = ops['child']['model'].child.train_loss(logits_graph, aux_logits)
                lr_graph =         ops["child"]["lr"]
                grad_norm_graph =  ops["child"]["grad_norm"](loss_graph, ops['child']['model'].child.tf_variables())
                train_acc_graph =  ops["child"]["train_acc"](logits_graph)
                train_op_graph =   ops["child"]["train_op"](train_loss_graph, ops['child']['model'].child.tf_variables())
                with fw.Session(config=fw.ConfigProto()) as sess:
                    logits, loss, lr, gn, tr_acc, _ = sess.run([
                        logits_graph,
                        loss_graph,
                        lr_graph,
                        grad_norm_graph,
                        train_acc_graph,
                        train_op_graph
                    ])
                    self.assertLess(loss, 4.0)
                    self.assertLess(lr, 0.06)
                    self.assertLess(tr_acc, 120)
                    self.assertLess(gn, 1000.0)

    def test_macro_search_loss(self):
        macro_search_flags = {
            "data_format": "NCHW",
            "search_for": "macro",
            "data_path": "data/cifar10",
            "output_dir": "outputs",
            "batch_size": 128,
            "num_epochs": 310,
            "log_every": 50,
            "eval_every_epochs": 1,
            "child_use_aux_heads": True,
            "child_num_layers": 14,
            "child_out_filters": 36,
            "child_l2_reg": 0.00025,
            "child_num_branches": 6,
            "child_keep_prob": 0.90,
            "child_drop_path_keep_prob": 0.60,
            "child_lr_cosine": True,
            "child_lr_max": 0.05,
            "child_lr_min": 0.0005,
            "child_lr_T_0": 10,
            "child_lr_T_mul": 2,
            "controller_training": True,
            "controller_search_whole_channels": True,
            "controller_entropy_weight": 0.0001,
            "controller_train_every": 1,
            "controller_sync_replicas": True,
            "controller_num_aggregate": 20,
            "controller_train_steps": 50,
            "controller_lr": 0.001,
            "controller_tanh_constant": 1.5,
            "controller_op_tanh_reduce": 2.5,
            "controller_skip_target": 0.4,
            "controller_skip_weight": 0.8 }
        with RestoreFLAGS(**macro_search_flags):
            images, labels = read_data(flags.FLAGS.data_path)
            with tf.Graph().as_default():
                ops = get_ops(images, labels)
                logits_graph =     ops['child']['model'](ops['child']['images'])
                loss_graph =       ops["child"]["loss"](logits_graph)
                lr_graph =         ops["child"]["lr"]
                grad_norm_graph =  ops["child"]["grad_norm"](loss_graph, ops['child']['model'].child.tf_variables())
                train_acc_graph =  ops["child"]["train_acc"](logits_graph)
                train_op_graph =   ops["child"]["train_op"](loss_graph, ops['child']['model'].child.tf_variables())
                with fw.Session(config=fw.ConfigProto()) as sess:
                    logits, loss, lr, gn, tr_acc, _ = sess.run([
                        logits_graph,
                        loss_graph,
                        lr_graph,
                        grad_norm_graph,
                        train_acc_graph,
                        train_op_graph
                    ])
                    self.assertLess(loss, 4.0)
                    self.assertLess(lr, 0.06)
                    self.assertLess(tr_acc, 120)
                    self.assertLess(gn, 4.0)

    def test_macro_final_loss(self):
        fixed_arc = "0"
        fixed_arc += " 3 0"
        fixed_arc += " 0 1 0"
        fixed_arc += " 2 0 0 1"
        fixed_arc += " 2 0 0 0 0"
        fixed_arc += " 3 1 1 0 1 0"
        fixed_arc += " 2 0 0 0 0 0 1"
        fixed_arc += " 2 0 1 1 0 1 1 1"
        fixed_arc += " 1 0 1 1 1 0 1 0 1"
        fixed_arc += " 0 0 0 0 0 0 0 0 0 0"
        fixed_arc += " 2 0 0 0 0 0 1 0 0 0 0"
        fixed_arc += " 0 1 0 0 1 1 0 0 0 0 1 1"
        fixed_arc += " 2 0 1 0 0 0 0 0 1 0 1 1 0"
        fixed_arc += " 1 0 0 1 0 0 0 1 1 1 0 1 0 1"
        fixed_arc += " 0 1 1 0 1 0 1 0 0 0 0 0 1 0 0"
        fixed_arc += " 2 0 0 1 0 0 0 0 0 0 0 1 0 1 0 1"
        fixed_arc += " 2 0 1 0 0 0 1 0 0 1 1 1 1 0 0 1 0"
        fixed_arc += " 2 0 0 0 0 1 0 1 0 1 0 0 1 0 1 0 0 1"
        fixed_arc += " 3 0 1 1 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0"
        fixed_arc += " 3 0 1 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1"
        fixed_arc += " 0 1 0 0 1 0 1 1 0 0 0 1 0 0 0 0 0 1 1 0 0"
        fixed_arc += " 3 0 1 0 1 1 0 0 1 0 1 1 0 1 1 0 1 0 0 1 0 0"
        fixed_arc += " 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0"
        fixed_arc += " 0 1 1 0 0 0 1 1 1 0 1 0 0 0 1 0 1 0 0 1 1 0 0 0"
        macro_search_flags = {
            "data_format": "NCHW",
            "search_for": "macro",
            "data_path": "data/cifar10",
            "output_dir": "outputs",
            "batch_size": 100,
            "num_epochs": 310,
            "log_every": 50,
            "eval_every_epochs": 1,
            "child_fixed_arc": fixed_arc,
            "child_use_aux_heads": True,
            "child_num_layers": 24,
            "child_out_filters": 96,
            "child_l2_reg": 2e-4,
            "child_num_branches": 4,
            "child_keep_prob": 0.50,
            "child_lr_cosine": True,
            "child_lr_max": 0.05,
            "child_lr_min": 0.001,
            "child_lr_T_0": 10,
            "child_lr_T_mul": 2,
            "controller_training": False,
            "controller_search_whole_channels": True,
            "controller_entropy_weight": 0.0001,
            "controller_train_every": 1,
            "controller_sync_replicas": True,
            "controller_num_aggregate": 20,
            "controller_train_steps": 50,
            "controller_lr": 0.001,
            "controller_tanh_constant": 1.5,
            "controller_op_tanh_reduce": 2.5,
            "controller_skip_target": 0.4,
            "controller_skip_weight": 0.8 }
        with RestoreFLAGS(**macro_search_flags):
            images, labels = read_data(flags.FLAGS.data_path)
            with tf.Graph().as_default():
                ops = get_ops(images, labels)
                logits_graph =     ops['child']['model'](ops['child']['images'])
                loss_graph =       ops["child"]["loss"](logits_graph)
                lr_graph =         ops["child"]["lr"]
                grad_norm_graph =  ops["child"]["grad_norm"](
                    loss_graph,
                    ops['child']['model'].child.tf_variables())
                train_acc_graph =  ops["child"]["train_acc"](logits_graph)
                train_op_graph =   ops["child"]["train_op"](
                    loss_graph,
                    ops['child']['model'].child.tf_variables())
                with fw.Session(config=fw.ConfigProto()) as sess:
                    logits, loss, lr, gn, tr_acc, _ = sess.run([
                        logits_graph,
                        loss_graph,
                        lr_graph,
                        grad_norm_graph,
                        train_acc_graph,
                        train_op_graph
                    ])
                    self.assertLess(loss, 4.0)
                    self.assertLess(lr, 0.06)
                    self.assertLess(tr_acc, 120)
                    self.assertLess(gn, 15)
