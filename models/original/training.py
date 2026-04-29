import tensorflow as tf
import numpy as np
from pathlib import Path
import pickle
import os
import random

keras = tf.keras

try:
    from ... import data
except ImportError:
    import data

result_path_folder = os.getenv("RESULT_PATH") or ""

is_remote = os.environ.get("KAGGLE_URL_BASE") is not None


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


def build_dataset(data, batch, seed=None, sample_pct=1.0):

    reproduc = seed is not None

    imgs, labels = data

    imgs = imgs.astype("float32")
    labels = labels["Vmax"].astype("float32")

    images_len = imgs.shape[0]
    print(f"Dataset len: {images_len}")

    images_sample_len = int(images_len * sample_pct)
    print(f"Dataset new len: {images_sample_len}")

    idx = list(range(images_len))
    random.shuffle(idx)
    idx = idx[:images_sample_len]

    imgs = imgs[idx]
    labels = labels.iloc[idx].reset_index(drop=True)

    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=len(imgs), seed=seed)
    dataset = dataset.map(parse_example, num_parallel_calls=(1 if reproduc else tf.data.AUTOTUNE))
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(buffer_size=(1 if reproduc else tf.data.AUTOTUNE))

    return dataset, imgs.shape


def load_datasets(channels, img_w, batch, sample_pct, seed=None, force=True):
    train, valid, _ = data.preprocess(channels, img_w, force=force)
    train_ds = build_dataset(train, batch, seed=seed, sample_pct=sample_pct)
    valid_ds = build_dataset(valid, batch, seed=seed)
    return train_ds, valid_ds


def build_model(input_shape, lr, l2_regularizer=1e-5, strides=(2, 2)):
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    reg = keras.regularizers.L2(l2_regularizer)
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
        optimizer=keras.optimizers.Adam(learning_rate=lr), 
        loss="mse", 
        metrics=["mse", "mae", keras.metrics.RootMeanSquaredError()],
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

    steps_per_epoch = train_shape[0] // batch
    validation_steps = valid_shape[0] // batch

    with tf.device("/GPU:0"):
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )

    return model, history


def save_model(model, path):

    model.export(Path(f"{path}/model"))
    model.save(f"{path}/model.keras")

    with open(f"{path}/model.pkl", "wb") as file:
        pickle.dump(model, file)

def save_history(history, path):
    with open(f"{path}/history.pkl", "wb") as file:
        pickle.dump(history.history, file)


def set_seed(seed):
    """Configure todas as seeds para reproducibilidade"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Força TensorFlow a ser determinístico (mais lento, mas reproduzível)
    tf.config.experimental.enable_op_determinism()
    
    # Desativa otimizações não-determinísticas
    tf.config.run_functions_eagerly(True)



def main(
    channels=[0, 3],
    img_w=64,
    batch=8,
    learning_rate=5e-5,
    epochs=500,
    sample_pct=1.0,
    seed=None,
    l2_regularizer=1e-5,
    force=True,
):
    if seed is not None: 
        set_seed(seed)

    train_ds, valid_ds = load_datasets(channels, img_w, batch, sample_pct, seed=seed, force=force)

    model = build_model(
        (img_w, img_w, len(channels)), learning_rate, l2_regularizer
    )
    model.summary()
    model, history = train_model(model, train_ds, valid_ds, epochs, batch)

    save_model(model, result_path_folder)
    save_history(history, result_path_folder)


if __name__ == "__main__":
    main()
