import dask.array as da
import numpy as np
import h5py
import tensorflow as tf
import pandas as pd
import datetime
import os

data_path = os.getenv("DATA_PATH") or "data"
preprocessed_path = os.getenv("PREPROCESSED_PATH") or "data/preprocessed"
print(f"folder_path: {preprocessed_path}")
preprocessed_train = f"{preprocessed_path}/train.h5"
preprocessed_valid = f"{preprocessed_path}/valid.h5"
preprocessed_test = f"{preprocessed_path}/test.h5"
preprocessed_files = [preprocessed_train, preprocessed_valid, preprocessed_test]


def split_data(images, info):
    years = [datetime.datetime.strptime(i, "%Y%m%d%H").year for i in list(info["time"])]
    years = np.array(years)
    train_values = (years >= 2003) & (years <= 2014)
    valid_values = (years >= 2015) & (years <= 2016)
    test_values = years == 2017
    train_idx = np.where(train_values)[0]
    valid_idx = np.where(valid_values)[0]
    test_idx = np.where(test_values)[0]

    print(f"{len(train_idx)}")
    print(f"{len(valid_idx)}")
    print(f"{len(test_idx)}")

    train_img, train_info = images[train_idx], info.iloc[train_idx]
    valid_img, valid_info = images[valid_idx], info.iloc[valid_idx]
    test_img, test_info = images[test_idx], info.iloc[test_idx]
    return (train_img, train_info), (valid_img, valid_info), (test_img, test_info)


def get_images_slice(images_shape, width):
    start = images_shape[1] // 2 - width // 2
    end = images_shape[1] // 2 + width // 2
    return slice(start, end)


def cut_images(images, width):
    slc = get_images_slice(images.shape, width)
    return images[:, slc, slc, :]


def compute(img, img_1, img_2):
    return np.abs(img - (img_1 * 2) + img_2)


def load_normalized_data(channels, img_w):
    print("Loading data...")

    dsfiles = ["TCIR-ALL_2017.h5", "TCIR-ATLN_EPAC_WPAC.h5", "TCIR-CPAC_IO_SH.h5"]
    dspaths = [f"{data_path}/{file}" for file in dsfiles]

    files = [h5py.File(file, mode="r") for file in dspaths]
    data = [da.from_array(file["matrix"]) for file in files]
    info = [pd.read_hdf(file, key="info", mode="r") for file in dspaths]
    info = pd.concat(info)

    data = da.concatenate(data, axis=0)

    print(f"Loaded dataset: Images {data.shape} - Labels {info.shape}")
    data = da.nan_to_num(data)
    data[data > 1000] = 0

    print("Calculating means...")
    means = [da.nanmean(data[:, :, :, i]).compute() for i in channels]

    print(f"Means values: {means}")

    print("Calcularing standard deviations...")
    std = [da.std(data[:, :, :, i]).compute() for i in channels]

    print(f"Standard deviation values: {std}")

    rotation_width = int(np.ceil(np.sqrt((img_w**2) * 2)))
    if rotation_width % 2 != 0:
        rotation_width += 1

    print("Cropping images...")
    data = cut_images(data, rotation_width)

    print("Normalizing images...")
    for chan in range(len(channels)):
        data[:, :, :, chan] -= means[chan]
        data[:, :, :, chan] /= std[chan]

    data = np.array(data[:, :, :, channels])

    for file in files:
        file.close()

    return data, info


def preprocess(channels, generated_channels, img_w, force=True):

    print("Forcing preprocess...")
    if not force:
        print("Checking if files exist")
        exists = all([os.path.isfile(file) for file in preprocessed_files])

        if exists:
            print("Files exist. Loading files...")
            with h5py.File(preprocessed_train, mode="r") as train:
                train_imgs = train["matrix"][:]
                train_labels = train["info"][:]
                train = (train_imgs, train_labels)

            with h5py.File(preprocessed_valid, mode="r") as valid:
                valid_imgs = valid["matrix"][:]
                valid_labels = valid["info"][:]
                valid = (valid_imgs, valid_labels)

            with h5py.File(preprocessed_test, mode="r") as test:
                test_imgs = test["matrix"][:]
                test_labels = test["info"][:]
                test = (test_imgs, test_labels)

            print(f"\nTrain images shape: {train_imgs.shape}")
            print(f"Train labels shape: {train_labels.shape}")
            print(f"\nValidation images shape: {valid_imgs.shape}")
            print(f"Validation labels shape: {valid_labels.shape}")
            print(f"\nTest images shape: {test_imgs.shape}")
            print(f"Test labels shape: {test_labels.shape}")
            return train, valid, test

    data, info = load_normalized_data(channels, img_w)  # images, labels
    print(f"Data shape: {data.shape}")
    print(f"Info shape: {info.shape}")

    print("Preprocessing data...")
    ids = info["ID"].unique()  # unique ids

    """
    CREATE NEW CHANNELS
    single_cyclone_indexes is a list of lists that contains
    the indexes for each image of a single cyclone
    """
    single_cyclone_indexes = []

    """
    fills the single_cyclone_indexes with the sorted indexes of cyclones
    """
    print("Filling ids...")
    for id in ids:

        print(f"\nFinding images of: {id}")
        sub_info = info[info["ID"] == id]
        
        sub_info = sub_info.sort_values("time")

        sorted_idx = sub_info.index

        print(f"{len(sorted_idx)} found.")
        single_cyclone_indexes.append((id, sorted_idx))

    """
    create all new channels
    """
    print("Creating new channels...")
    cyclone_new_channels = []

    for id, indexes in single_cyclone_indexes:

        images = data[indexes]
        labels = info.iloc[indexes]

        # sort values
        labels = labels.sort_values("time").reset_index(drop=True)
        images = images[labels.index]

        # add sorted labels to sorted_labels list
        # labels = labels[2:]

        print(f"Generating channels for cyclone {id} of {len(indexes)} images")
        for idx in range(2, len(indexes)):

            new_imgs = None
            for gen_ch in generated_channels:

                current_img = images[idx, :, :, gen_ch]
                previous_img = images[idx - 1, :, :, gen_ch]
                previous_previous_img = images[idx - 2, :, :, gen_ch]
                new_img = compute(current_img, previous_img, previous_previous_img)

                if not new_imgs:
                    new_imgs = np.expand_dims(new_img, axis=-1)
                else:
                    new_imgs = np.concatenate((new_imgs, new_img), axis=-1)

            cyclone_new_channels.append(new_imgs)

    cyclone_new_channels = np.array(cyclone_new_channels)
    if len(cyclone_new_channels.shape) < 4:
        cyclone_new_channels = np.expand_dims(cyclone_new_channels, axis=-1)


    """
    select only the cyclones to add a new channel
    by eliminating the first two images of each cyclone
    """
    # single_cyclone_indexes = [i[2:] for _, i in single_cyclone_indexes]


    """
    load cyclones
    """
    print("\nLoading processed images\n")
    # images = None
    # labels = None
    # for i, (id, cyc_idx) in enumerate(single_cyclone_indexes):
    #     print(f"{i}/{len(single_cyclone_indexes)} - Loading from cyclone {id}")
    #     cyc_idx = cyc_idx[2:]
    #     imgs = data[cyc_idx]
    #     lbls = info.iloc[cyc_idx]
    #     if images is not None and labels is not None:
    #         images = np.concatenate((images, imgs), axis=0)
    #         labels = np.concatenate((labels, lbls), axis=0)
    #     else:
    #         images = imgs
    #         labels = lbls

    images = ()
    labels = []
    for id, cyc_idx in single_cyclone_indexes:
        print(cyc_idx)
        cyc_idx = cyc_idx[2:]
        imgs = data[cyc_idx]
        lbls = info.iloc[cyc_idx]
        images += (imgs,)
        labels.append(lbls)
    
    print("Concatenating images...")
    images = np.concatenate(images, axis=0)

    print("Concatenating labels...")
    labels = pd.concat(labels, axis=0)


    print(f"\nData shape: {images.shape}")

    images = np.concatenate((images, cyclone_new_channels), axis=-1)
    print(f"Data shape: {images.shape}")

    train, valid, test = split_data(images, labels)
    return train, valid, test


def save_preprocessed(channels=[0, 3], generated_channels=[0], img_w=64, data=None):

    if not data:
        (
            (train_imgs, train_labels),
            (valid_imgs, valid_labels),
            (test_imgs, test_labels),
        ) = preprocess(channels, generated_channels, img_w, force=True)
    else:
        (
            (train_imgs, train_labels),
            (valid_imgs, valid_labels),
            (test_imgs, test_labels),
        ) = data

    print("\nTrain dataset:")
    print(f"Images shape: {train_imgs.shape}")
    print(f"Info shape: {train_labels.shape}\n")

    print("\nValidation dataset:")
    print(f"Images shape: {valid_imgs.shape}")
    print(f"Info shape: {valid_labels.shape}\n")

    print("\nTest dataset:")
    print(f"Images shape: {test_imgs.shape}")
    print(f"Info shape: {test_labels.shape}\n")

    img_new_shape = train_imgs.shape[1:]
    labels_new_shape = train_labels.shape[1:]

    img_max_shape = (None,) + img_new_shape
    labels_max_shape = (None,) + labels_new_shape

    img_new_shape = (0,) + img_new_shape
    labels_new_shape = (0,) + labels_new_shape

    os.makedirs(data_path, exist_ok=True)

    print("Writing traininig file...")
    print(train_imgs.shape)
    print(train_labels.shape)
    with h5py.File(preprocessed_train, mode="w") as train:

        if "matrix" not in train:
            train.create_dataset("matrix", shape=img_new_shape, maxshape=img_max_shape)
        train["matrix"].resize(train["matrix"].shape[0] + train_imgs.shape[0], axis=0)
        train["matrix"][-train_imgs.shape[0] :] = train_imgs

    train_labels.to_hdf(preprocessed_train, key="info", mode="a")
        

    print("Writing validation file...")
    print(valid_imgs.shape)
    print(valid_labels.shape)
    with h5py.File(preprocessed_valid, mode="w") as valid:

        if "matrix" not in valid:
            valid.create_dataset("matrix", shape=img_new_shape, maxshape=img_max_shape)
        valid["matrix"].resize(valid["matrix"].shape[0] + valid_imgs.shape[0], axis=0)
        valid["matrix"][-valid_imgs.shape[0] :] = valid_imgs

    valid_labels.to_hdf(preprocessed_valid, key="info", mode="a")

    print("Writing test file...")
    print(test_imgs.shape)
    print(test_labels.shape)
    with h5py.File(preprocessed_test, mode="w") as test:

        if "matrix" not in test:
            test.create_dataset("matrix", shape=img_new_shape, maxshape=img_max_shape)
        test["matrix"].resize(test["matrix"].shape[0] + test_imgs.shape[0], axis=0)
        test["matrix"][-test_imgs.shape[0] :] = test_imgs

    test_labels.to_hdf(preprocessed_test, key="info", mode="a")


def main():
    save_preprocessed()


if __name__ == "__main__":
    main()
