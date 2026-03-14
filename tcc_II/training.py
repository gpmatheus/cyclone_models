import tensorflow as tf
import numpy as np
from pathlib import Path

keras = tf.keras

try:
    from . import data
except ImportError:
    import data


def parse_example(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_image_tf(image)
    return image, label


def preprocess_image_tf(image):
    angle_rad = tf.random.uniform([], 0, 2 * np.pi)
    image_shape = tf.shape(image)[0:2]
    cx = tf.cast(image_shape[1] / 2, tf.float32)
    cy = tf.cast(image_shape[0] / 2, tf.float32)
    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)
    transform = tf.stack(
        [
            cos_a,
            -sin_a,
            (1 - cos_a) * cx + sin_a * cy,
            sin_a,
            cos_a,
            (1 - cos_a) * cy - sin_a * cx,
            0.0,
            0.0,
        ]
    )
    transform = tf.reshape(transform, [8])
    transform = tf.expand_dims(transform, 0)
    image = tf.expand_dims(image, 0)
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transform,
        output_shape=image_shape,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    rotated = tf.squeeze(rotated, 0)
    return tf.image.resize_with_crop_or_pad(rotated, 64, 64)


def build_dataset(data, batch):
    imgs, labels = data

    imgs = imgs.astype("float32")
    # labels = labels.astype("float32")

    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=len(imgs))
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, imgs.shape


def load_datasets(channels, generated_channels, img_w, batch):
    train, valid, _ = data.preprocess(channels, generated_channels, img_w, force=False)

    train_ds, valid_ds = build_dataset(train, batch), build_dataset(valid, batch)
    return train_ds, valid_ds


def build_model(input_shape, lr, strides=(2, 2)):
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    reg = keras.regularizers.L2(1e-5)
    model = keras.models.Sequential()
    model.add(keras.layers.Input(input_shape))
    model.add(
        keras.layers.Conv2D(
            16,
            (4, 4),
            strides=strides,
            activation="relu",
            kernel_initializer=initializer,
            kernel_regularizer=reg,
            bias_initializer=initializer,
            bias_regularizer=reg,
        )
    )
    model.add(
        keras.layers.Conv2D(
            32,
            (3, 3),
            strides=strides,
            activation="relu",
            kernel_initializer=initializer,
            kernel_regularizer=reg,
            bias_initializer=initializer,
            bias_regularizer=reg,
        )
    )
    model.add(
        keras.layers.Conv2D(
            64,
            (3, 3),
            strides=strides,
            activation="relu",
            kernel_initializer=initializer,
            kernel_regularizer=reg,
            bias_initializer=initializer,
            bias_regularizer=reg,
        )
    )
    model.add(
        keras.layers.Conv2D(
            128,
            (3, 3),
            strides=strides,
            activation="relu",
            kernel_initializer=initializer,
            kernel_regularizer=reg,
            bias_initializer=initializer,
            bias_regularizer=reg,
        )
    )

    model.add(keras.layers.Flatten())

    model.add(
        keras.layers.Dense(
            256,
            activation="relu",
            kernel_initializer=initializer,
            kernel_regularizer=reg,
            bias_initializer=initializer,
            bias_regularizer=reg,
        )
    )
    model.add(
        keras.layers.Dense(
            64,
            activation="relu",
            kernel_initializer=initializer,
            kernel_regularizer=reg,
            bias_initializer=initializer,
            bias_regularizer=reg,
        )
    )
    model.add(
        keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer=initializer,
            kernel_regularizer=reg,
            bias_initializer=initializer,
            bias_regularizer=reg,
        )
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mse"]
    )
    return model


def train_model(
    model,
    train_ds,
    valid_ds,
    epochs,
    batch,
):

    train_ds, train_shape = train_ds
    valid_ds, valid_shape = valid_ds

    with tf.device("/GPU:0"):
        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=epochs,
            steps_per_epoch=train_shape[0] // batch,
            validation_steps=valid_shape[0] // batch,
        )

    return model


def save_model(model, path):

    model.save(path)


def main(
    channels=[0, 3],
    generated_channels=[0],
    img_w=64,
    batch=8,
    learning_rate=5e-5,
    epochs=500,
):

    train_ds, valid_ds = load_datasets(channels, generated_channels, img_w, batch)

    model = build_model(
        (img_w, img_w, len(channels) + len(generated_channels)), learning_rate
    )
    model.summary()
    model = train_model(model, train_ds, valid_ds, epochs, batch)
    save_model(model, Path("model"))


if __name__ == "__main__":
    main()
