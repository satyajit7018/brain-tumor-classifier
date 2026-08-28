import tensorflow as tf


def build_resnet50(input_shape=(224, 224, 3), num_classes=4, fine_tune_at=140) -> tf.keras.Model:
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    # Scale input from [0, 1] to ImageNet preprocess
    x = tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)
    x = base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="resnet50_finetuned")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_efficientnet_b0(input_shape=(224, 224, 3), num_classes=4, fine_tune_at=200) -> tf.keras.Model:
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs * 255.0)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="efficientnet_b0_finetuned")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

