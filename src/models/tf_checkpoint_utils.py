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

import os

import tensorflow as tf


def setup_checkpoint_and_restore(
    generator: tf.keras.models.Model,
    discriminator: tf.keras.models.Model,
    generator_optimizer: tf.keras.optimizers.Optimizer,
    discriminator_optimizer: tf.keras.optimizers.Optimizer,
    training_step: tf.Variable,
    checkpoint_dir: str,
    experiment_key: str,
    max_to_keep: int,
) -> tf.train.CheckpointManager:

    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        training_step=training_step,
    )

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=os.path.join(checkpoint_dir, experiment_key),
        max_to_keep=max_to_keep,
    )

    ckpt = checkpoint_manager.latest_checkpoint
    checkpoint.restore(ckpt)
    print(f"Training from {ckpt if ckpt else 'scratch'}")

    return checkpoint_manager
