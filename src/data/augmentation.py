import tensorflow as tf

augmentation_layers = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="augmentation",
)


def apply_augmentation(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.map(
        lambda x, y: (augmentation_layers(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

