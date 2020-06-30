# MIT License
# Copyright 2020 Ryan Hausen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# ofthis software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import os
import time
from functools import partial
from itertools import starmap
from multiprocessing import Pool
from typing import Callable, Dict, Iterator, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from astropy.io import fits
from tqdm import tqdm

import comet_ml
import gin
import gin.tf
import src.models.comet_utils as comet_utils
import src.models.tf_checkpoint_utils as tf_checkpoint_utils

TensorLike = Union[np.ndarray, tf.Tensor]

# ==============================================================================
# Special functions that need to be inplemented in the files that get imported
# ==============================================================================
DatasetFunc = Callable[
    None,
    [
        Iterator[Union[TensorLike, List[TensorLike]]],  # Training data
        int,  # number of training batches
        Iterator[Union[TensorLike, List[TensorLike]]],  # Test data
        int,  # number of testing batches
    ],
]


LossFunc = Callable[
    [
        Union[TensorLike, List[TensorLike]],  # Data
        Union[TensorLike, List[TensorLike]],  # Generator out
        Union[TensorLike, List[TensorLike]],  # Discriminator out real
        Union[TensorLike, List[TensorLike]],  # Discriminator out fake
    ],
    Union[float, tf.float32],  # Loss to optimize
]

# The metric function takes the output from a training step and returns a
# dictionary of metrics where the key is the name of the metric and the
# the value is a Tuple, the first value is a string from one of three options
# ["float", "figure", "image"], the second value is the metric to be recorded
# which depending on the indicated type, will be recorded by comet.ml an
# example of a MetricFuncResult with a single entry could be the following:
# { "accuracy": ("float", 0.85)}
MetricFuncResult = Dict[str, Tuple[str, Union[float, plt.Figure, np.ndarray]]]

MetricFunc = Callable[
    [
        float,  # epoch completion [0, 1.0]
        Union[TensorLike, List[TensorLike]],  # Data
        Union[TensorLike, List[TensorLike]],  # Generator out
        Union[TensorLike, List[TensorLike]],  # Discriminator out real
        Union[TensorLike, List[TensorLike]],  # Discriminator out fake
    ],
    MetricFuncResult,
]
# ==============================================================================


@gin.configurable
def gan_training_func(
    dataset_func: Callable,
    generator: tf.keras.models.Model,
    discriminator: tf.keras.models.Model,
    generator_optimizer: tf.keras.optimizers.Optimizer,
    discriminator_optimizer: tf.keras.optimizers.Optimizer,
    generator_loss: LossFunc,
    discriminator_loss: LossFunc,
    train_metric_function: MetricFunc,
    test_metric_function: MetricFunc,
    checkpoint_dir: str,
    max_checkpoints_to_keep: int,
    epochs: int,
    log_metric_batch_idx: int,
    model_code_file: str,
    comet_project_name: str,
    comet_disabled: bool,
    comet_experiment_key: str,
) -> None:
    pass

    experiment = comet_utils.setup_experiment(
        comet_experiment_key,
        comet_project_name,
        config_str_to_dict(gin.config_str()),
        model_code_file,
        comet_disabled,
    )

    training_step = tf.Variable(0)
    checkpoint_manager = tf_checkpoint_utils.setup_checkpoint_and_restore(
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        training_step,
        checkpoint_dir,
        experiment.get_key(),
        max_checkpoints_to_keep,
    )

    (
        training_data,
        train_steps_per_epoch,
        testing_data,
        test_steps_per_epoch,
    ) = dataset_func()


    train_step_f = partial(
        _train_step_f,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        generator_loss,
        discriminator_loss
    )

    train_step = partial(
        _step,
        training_step,
        train_step_f,
        checkpoint_manager.save,
        lambda batch_idx: batch_idx % log_metric_batch_idx == 0,
        partial(
            comet_utils.get_async_metric_logging_f,
            experiment,
            experiment.train,
        ),
        train_metric_function,
        train_steps_per_epoch,
    )


    test_step_f = partial(
        _test_step_f,
        generator,
        discriminator,
    )

    test_step = partial(
        _step,
        training_step,
        test_step_f,
        lambda x: None,
        lambda batch_idx: True,
        partial(
            comet_utils.get_async_metric_logging_f,
            experiment,
            experiment.test,
        ),
        test_metric_function,
        test_steps_per_epoch,
    )

    epoch_f = partial(
        _epoch_f,
        training_data,
        train_steps_per_epoch,
        train_step_f,
        testing_data,
        test_steps_per_epoch,
        test_step_f
    )

    for _ in map(epoch_f, range(epochs)):
        pass


def _train_step_f(
    training_step: tf.Variable,
    generator: tf.keras.models.Model,
    discriminator: tf.keras.models.Model,
    generator_optimizer: tf.keras.optimizers.Optimizer,
    discriminator_optimizer: tf.keras.optimizers.Optimizer,
    generator_loss: LossFunc,
    discriminator_loss: LossFunc,
    data: Union[TensorLike, List[TensorLike]],
) -> Tuple[Union[TensorLike, List[TensorLike]], Union[TensorLike, List[TensorLike]]]:

    training_step.assign_add(1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dsc_tape:
        generator_out = generator([data], training=True)

        discriminator_out_real = discriminator([data], training=True)
        discriminator_out_fake = discriminator([generator_out], training=True)

        loss_g = generator_loss(
            data, generator_out, discriminator_out_real, discriminator_out_fake
        )
        loss_d = discriminator_loss(
            data, generator_out, discriminator_out_real, discriminator_out_fake
        )

    gen_grads = gen_tape.gradient(loss_g, generator.trainable_variables)
    dsc_grads = dsc_tape.gradient(loss_d, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(dsc_grads, discriminator.trainable_variables)
    )

    return generator_out, discriminator_out_real, discriminator_out_fake



def _test_step_f(
    generator: tf.keras.models.Model,
    discriminator: tf.keras.models.Model,
    data: Union[TensorLike, List[TensorLike]],
) -> Tuple[Union[TensorLike, List[TensorLike]], Union[TensorLike, List[TensorLike]]]:
    generator_out = generator([data])

    discriminator_out_real = discriminator([data])
    discriminator_out_fake = discriminator([generator_out])

    return generator_out, discriminator_out_real, discriminator_out_fake


def _step(
    training_step: tf.Variable,
    step_f: Callable[
        Union[TensorLike, List[TensorLike]],
        Tuple[Union[TensorLike, List[TensorLike], Union[TensorLike, List[TensorLike]]]],
    ],
    save_f: Callable[None, None],
    should_log_metric: Callable[int, bool],
    get_log_result_func: Callable[float, Callable[MetricFuncResult]],
    metric_f: MetricFunc,
    batches_in_dataset: int,
    batch_idx: int,
    data: Union[TensorLike, List[TensorLike]],
) -> int:

    generator_out, discriminator_out_real, discriminator_out_fake = step_f(data)

    if should_log_metric(batch_idx):
        data_progress = batch_idx / batches_in_dataset
        log_result_f = get_log_result_func(training_step.numpy())

        metric_params = [
            data_progress,
            data,
            generator_out,
            discriminator_out_real,
            discriminator_out_fake,
        ]

        with Pool(1) as p:
            p.apply_async(metric_f, metric_params, callback=log_result_f)

        save_f()

    return batch_idx


# TODO: Pick up here
def _epoch_f(
    training_data: tf.data.Dataset,
    train_steps_per_epoch: int,
    train_step_f: Callable,
    testing_data: tf.data.Dataset,
    test_steps_per_epoch: int,
    test_step_f: Callable,
    epoch_idx: int
) -> None:

    print("Epoch: ", epoch_idx)
    start_time = time.time()

    training_finished_predicate = lambda idx: idx == train_steps_per_epoch
    train_batch_idx = starmap(train_step_f, enumerate(training_data, start=1))
    any(
        filter(
            training_finished_predicate,
            tqdm(
                train_batch_idx,
                total=train_steps_per_epoch - 1,
                desc="Training",
                unit="Step",
            ),
        )
    )

    test_epoch_finished = lambda idx: idx == test_steps_per_epoch
    test_batch_idx = starmap(test_step_f, enumerate(testing_data, start=1))
    any(
        filter(
            test_epoch_finished,
            tqdm(
                test_batch_idx,
                total=test_steps_per_epoch - 1,
                desc="Testing",
                unit="Step",
            ),
        )
    )

    print(f"\nEpoch completed in {np.round(time.time()-start_time, 2)} seconds")


def config_str_to_dict(config_str: str) -> Dict[str, str]:
    """Converts a Gin.config_str() to a dict for logging with comet.ml"""

    predicate = lambda x: len(x) > 0 and not x.startswith("#")
    to_kv = lambda x: [l.strip() for l in x.split("=")]
    lines = config_str.splitlines()
    return {k: v for k, v in map(to_kv, filter(predicate, lines))}


def main(gin_config_file: str) -> None:
    gin.parse_config_file(gin_config_file)
    gan_training_func()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Trainer")
    parser.add_argument("config", help="Gin config file with model params.")

    args = parser.parse_args()
    main(args.config)
