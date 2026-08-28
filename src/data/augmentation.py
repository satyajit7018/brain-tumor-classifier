"""Augmentation pipeline. Applied specifically to address class imbalance,
not as a generic default, check src/data/dataset.py's class balance output
first and decide which classes need heavier augmentation.
"""

import tensorflow as tf

augmentation_layers = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="augmentation",
)


def apply_augmentation(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.map(
        lambda x, y: (augmentation_layers(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
