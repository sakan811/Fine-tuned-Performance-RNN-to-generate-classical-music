# Copyright 2023 The Magenta Authors.
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

# This file has been modified from the original
# Added Early-Stop and new logging for evaluation

"""Train and evaluate an event sequence RNN model."""

import os
import tempfile

from magenta.models.shared import sequence_generator_bundle
import tensorflow.compat.v1 as tf
import tf_slim


# Early Stop
class EarlyStoppingHook(tf.train.SessionRunHook):
    """
    A function that requests a stop at a specified step that inherits from the tf.train.SessionRunHook,
    which is a TensorFlow class that provides methods to interact and alter training and evaluation operations.
    """

    def __init__(self, patience=100, min_delta=0.001):
        self.patience = patience # Number of steps to wait for improvement before stopping.
        self.min_delta = min_delta # Minimum change in loss to consider as an improvement.
        self.best_score = None # Best observed evaluation loss score.
        self.early_stop = False # Whether to stop the training.
        self.wait = 0 # Counter for steps without improvement.
     
    # This function is called before each session run and specifies the TensorFlow metrics to fetch.    
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            {'eval_loss': tf.get_collection('loss')[0],
            #  'eval_accuracy': tf.get_collection('metrics/accuracy')[0],
             'global_step': tf.train.get_global_step()})

    # This function is called after each session run and uses the fetched metrics to implement the Early-stop logic.
    def after_run(self, run_context, run_values):
        current_score = run_values.results['eval_loss']
        step = run_values.results['global_step']

        # Initialize the 'best_score' with 'current_score' if it is the first run.
        if self.best_score is None:
            self.best_score = current_score
            
        # Reset the counter if the score improves.
        elif current_score < self.best_score + self.min_delta:
            self.wait = 0
            self.best_score = current_score
        
        # Increase the counter if the score does not improve.
        else:
            self.wait += 1
            
            # Activate Early-stop if the counter exceeds the 'patience' value.
            if self.wait >= self.patience:
                self.early_stop = True
                request_stop_msg = 'Requesting early stopping at global step {}: no improvement in loss score for {} steps.'
                tf.logging.info(request_stop_msg.format(step, self.patience))  
                raise tf.errors.CancelledError(None, None, "Early stopping was requested")
            
            
def run_training(build_graph_fn, train_dir, num_training_steps=None,
                 summary_frequency=10, save_checkpoint_secs=60,
                 checkpoints_to_keep=10, keep_checkpoint_every_n_hours=1,
                 master='', task=0, num_ps_tasks=0, 
                 warm_start_bundle_file=None, early_stopping_params={}, 
                 use_early_stopping=False):  
    """Runs the training loop.

  Args:
    build_graph_fn: A function that builds the graph ops.
    train_dir: The path to the directory where checkpoints and summary events
        will be written to.
    num_training_steps: The number of steps to train for before exiting.
    summary_frequency: The number of steps between each summary. A summary is
        when graph values from the last step are logged to the console and
        written to disk.
    save_checkpoint_secs: The frequency at which to save checkpoints, in
        seconds.
    checkpoints_to_keep: The number of most recent checkpoints to keep in
       `train_dir`. Keeps all if set to 0.
    keep_checkpoint_every_n_hours: Keep a checkpoint every N hours, even if it
        results in more checkpoints than checkpoints_to_keep.
    master: URL of the Tensorflow master.
    task: Task number for this worker.
    num_ps_tasks: Number of parameter server tasks.
    warm_start_bundle_file: Path to a sequence generator bundle file that will
        be used to initialize the model weights for fine-tuning.
    early_stopping_params: Early stopping parameters.
    use_early_stopping: Control the Early Stop function. If True, early stop function is turned on.
    """
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(num_ps_tasks)):
            build_graph_fn()

        global_step = tf.train.get_or_create_global_step()
        loss = tf.get_collection('loss')[0]
        perplexity = tf.get_collection('metrics/perplexity')[0]
        accuracy = tf.get_collection('metrics/accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

        logging_dict = {
            'Global Step': global_step,
            'Loss': loss,
            'Perplexity': perplexity,
            'Accuracy': accuracy
        }
      
        if early_stopping_params:
            early_stopping = EarlyStoppingHook(patience=early_stopping_params['patience'], min_delta=0.001)
        else:
            early_stopping = None

        hooks = [
            tf.train.NanTensorHook(loss),
            tf.train.LoggingTensorHook(
                logging_dict, every_n_iter=summary_frequency),
            tf.train.StepCounterHook(
                output_dir=train_dir, every_n_steps=summary_frequency),
        ]
        
        if num_training_steps:
            hooks.append(tf.train.StopAtStepHook(num_training_steps))

        if use_early_stopping and early_stopping:  # Condition for early stopping
            hooks.append(early_stopping)
        
        with tempfile.TemporaryDirectory() as tempdir:
            if warm_start_bundle_file:
                # We are fine-tuning from a pretrained bundle. Unpack the bundle and
                # save its checkpoint to a temporary directory.
                warm_start_bundle_file = os.path.expanduser(warm_start_bundle_file)
                bundle = sequence_generator_bundle.read_bundle_file(
                    warm_start_bundle_file)
                checkpoint_filename = os.path.join(tempdir, 'model.ckpt')
                with tf.gfile.Open(checkpoint_filename, 'wb') as f:
                    # For now, we support only 1 checkpoint file.
                    f.write(bundle.checkpoint_file[0])
                variables_to_restore = tf_slim.get_variables_to_restore(
                    exclude=['global_step', '.*Adam.*', 'beta.*_power'])
                init_op, init_feed_dict = tf_slim.assign_from_checkpoint(
                    checkpoint_filename, variables_to_restore)
                init_fn = lambda scaffold, sess: sess.run(init_op, init_feed_dict)
            else:
                init_fn = None

            scaffold = tf.train.Scaffold(
                init_fn=init_fn,
                saver=tf.train.Saver(
                    max_to_keep=checkpoints_to_keep,
                    keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))
            
            tf.logging.info('Starting training loop...')
            tf_slim.training.train(
                train_op=train_op,
                logdir=train_dir,
                scaffold=scaffold,
                hooks=hooks,
                save_checkpoint_secs=save_checkpoint_secs,
                save_summaries_steps=summary_frequency,
                master=master,
                is_chief=task == 0)
        tf.logging.info('Training complete.')


# TODO(adarob): Limit to a single epoch each evaluation step.
def run_eval(build_graph_fn, train_dir, eval_dir, num_batches,
             timeout_secs=300):
  """Runs the training loop.

  Args:
    build_graph_fn: A function that builds the graph ops.
    train_dir: The path to the directory where checkpoints will be loaded
        from for evaluation.
    eval_dir: The path to the directory where the evaluation summary events
        will be written to.
    num_batches: The number of full batches to use for each evaluation step.
    timeout_secs: The number of seconds after which to stop waiting for a new
        checkpoint.
  Raises:
    ValueError: If `num_batches` is less than or equal to 0.
  """
  if num_batches <= 0:
    raise ValueError(
        '`num_batches` must be greater than 0. Check that the batch size is '
        'no larger than the number of records in the eval set.')
  with tf.Graph().as_default():
    build_graph_fn()

    global_step = tf.train.get_or_create_global_step()
    loss = tf.get_collection('loss')[0]
    perplexity = tf.get_collection('metrics/perplexity')[0]
    accuracy = tf.get_collection('metrics/accuracy')[0]
    eval_ops = tf.get_collection('eval_ops')
    
    logging_dict = {
        'Global Step': global_step,
        'Loss': loss,
        'Perplexity': perplexity,
        'Accuracy': accuracy
    }

    hooks = [
        EvalLoggingTensorHookNew(logging_dict, every_n_iter=1), 
        tf_slim.evaluation.StopAfterNEvalsHook(num_batches),
        tf_slim.evaluation.SummaryAtEndHook(eval_dir),
    ]

    tf_slim.evaluation.evaluate_repeatedly(
        train_dir,
        eval_ops=eval_ops,
        hooks=hooks,
        eval_interval_secs=60,
        timeout=timeout_secs)


class EvalLoggingTensorHookNew(tf.train.LoggingTensorHook):
  """Logs the values of tensors in evaluation mode."""

  def after_run(self, run_context, run_values):
    """Overrides the after_run method of the LoggingTensorHook class."""
    super(EvalLoggingTensorHookNew, self).after_run(run_context, run_values)

    # Extract the results from run_values
    results = run_values.results

    # Log the values of the tensors
    tf.logging.info('New eval log: global_step = %s, loss = %s, perplexity = %s, accuracy = %s',
                    results['Global Step'], results['Loss'], 
                    results['Perplexity'], results['Accuracy'])
