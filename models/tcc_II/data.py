import dask.array as da
import numpy as np
import h5py
import pandas as pd
import datetime
import os
from typing import Tuple, List

data_path = os.getenv("DATA_PATH") or "data"
preprocessed_path = os.getenv("PREPROCESSED_PATH") or "data/preprocessed"
print(f"folder_path: {preprocessed_path}")
preprocessed_train = f"{preprocessed_path}/train.h5"
preprocessed_valid = f"{preprocessed_path}/valid.h5"
preprocessed_test = f"{preprocessed_path}/test.h5"
preprocessed_files = [preprocessed_train, preprocessed_valid, preprocessed_test]

def get_images_slice(images_shape, width):
    # Calcula o slice para cortar a imagem centralizada com a largura especificada
    start = images_shape[1] // 2 - width // 2
    end = images_shape[1] // 2 + width // 2
    return slice(start, end)

def cut_images(images, width):
    # Corta as imagens para o tamanho especificado, centralizando o corte
    slc = get_images_slice(images.shape, width)
    return images[:, slc, slc, :]

def calculate_img_ration_w(img_w):
    rotation_width = int(np.ceil(np.sqrt((img_w**2) * 2)))
    if rotation_width % 2 != 0:
        rotation_width += 1
    return rotation_width

def clear_images(images):
    images = da.nan_to_num(images)
    images[images > 1000] = 0
    return images

def get_train_data(images: da.Array | np.ndarray, info: pd.DataFrame) -> Tuple[da.Array, pd.DataFrame]:
    # Ciclones de 2003-2014
    years = [datetime.datetime.strptime(i, "%Y%m%d%H").year for i in list(info["time"])]
    years = np.array(years)
    train_values = (years >= 2003) & (years <= 2014)
    train_idx = np.where(train_values)[0]
    train_img, train_info = images[train_idx], info.iloc[train_idx]
    train_info = train_info.reset_index(drop=True)
    return train_img, train_info

def get_valid_data(images: da.Array | np.ndarray, info: pd.DataFrame) -> Tuple[da.Array, pd.DataFrame]:
    # Ciclones de 2015-2016
    years = [datetime.datetime.strptime(i, "%Y%m%d%H").year for i in list(info["time"])]
    years = np.array(years)
    valid_values = (years >= 2015) & (years <= 2016)
    valid_idx = np.where(valid_values)[0]
    valid_img, valid_info = images[valid_idx], info.iloc[valid_idx]
    valid_info = valid_info.reset_index(drop=True)
    return valid_img, valid_info

def get_test_data(images: da.Array | np.ndarray, info: pd.DataFrame) -> Tuple[da.Array, pd.DataFrame]:
    # Ciclones de 2017
    years = [datetime.datetime.strptime(i, "%Y%m%d%H").year for i in list(info["time"])]
    years = np.array(years)
    test_values = years == 2017
    test_idx = np.where(test_values)[0]
    test_img, test_info = images[test_idx], info.iloc[test_idx]
    test_info = test_info.reset_index(drop=True)
    return test_img, test_info

def split_data(images, info):
    train_img, train_info = get_train_data(images, info)
    valid_img, valid_info = get_valid_data(images, info)
    test_img, test_info = get_test_data(images, info)
    return (train_img, train_info), (valid_img, valid_info), (test_img, test_info)

def compute(img_t, img_t1, img_t2):
    # Calcula a diferença absoluta entre imagens consecutivas para detecção de movimento
    # Funciona com tanto numpy arrays quanto dask arrays
    return da.abs(img_t - (img_t1 * 2) + img_t2) if isinstance(img_t, da.Array) else np.abs(img_t - (img_t1 * 2) + img_t2)

def create_new_channels(
    cyclone_indexes: Tuple[str, np.ndarray], 
    dsimages: da.Array | np.ndarray, 
    dsinfo: pd.DataFrame,
    generated_channels: List[int]=[0],
):
    """
    Cria novos canais calculando diferenças entre frames consecutivas.
    Usa dask arrays para evitar carregar tudo na memória.
    """
    cyclone_new_channels = []
    generated_channels = list(range(len(generated_channels)))

    for _, indexes in cyclone_indexes:

        if len(indexes) < 3: 
            continue

        images = dsimages[indexes]
        labels = dsinfo.iloc[indexes]

        # sort values
        labels = labels.sort_values("time").reset_index(drop=True)
        images = images[labels.index]

        img_t   = images[2:, :, :, generated_channels]
        img_t1  = images[1:-1, :, :, generated_channels]
        img_t2  = images[:-2, :, :, generated_channels]

        new_channels = compute(img_t, img_t1, img_t2)
            
        cyclone_new_channels.append(new_channels)
    
    # Usar dask concatenate se dados são dask arrays
    if len(cyclone_new_channels) > 0 and isinstance(cyclone_new_channels[0], da.Array):
        cyclone_new_channels = da.concatenate(cyclone_new_channels, axis=0)
    else:
        cyclone_new_channels = np.concatenate(cyclone_new_channels, axis=0)

    print(cyclone_new_channels.shape)

    return cyclone_new_channels

def add_channels(generated_channels, data):
    """
    Adiciona novos canais gerados aos dados originais.
    Mantém dados em dask arrays durante todo o processamento.
    """
    (trainds, traininfo), (validds, validinfo), (testds, testinfo) = data
    
    ids = pd.concat([traininfo, validinfo, testinfo])["ID"].unique()

    train_single_cyclone_indexes = []
    valid_single_cyclone_indexes = []
    test_single_cyclone_indexes = []

    print("Filling ids...")
    for id in ids:

        train_sub_info = traininfo[traininfo["ID"] == id]
        train_sub_info = train_sub_info.sort_values("time")
        train_sorted_idx = train_sub_info.index
        train_single_cyclone_indexes.append((id, train_sorted_idx))

        valid_sub_info = validinfo[validinfo["ID"] == id]
        valid_sub_info = valid_sub_info.sort_values("time")
        valid_sorted_idx = valid_sub_info.index
        valid_single_cyclone_indexes.append((id, valid_sorted_idx))

        test_sub_info = testinfo[testinfo["ID"] == id]
        test_sub_info = test_sub_info.sort_values("time")
        test_sorted_idx = test_sub_info.index
        test_single_cyclone_indexes.append((id, test_sorted_idx))
    
    print("Creating new channels...")
    trainds_new_channels = create_new_channels(
        train_single_cyclone_indexes, 
        trainds, 
        traininfo, 
        generated_channels=generated_channels,
    )

    validds_new_channels = create_new_channels(
        valid_single_cyclone_indexes, 
        validds, 
        validinfo, 
        generated_channels=generated_channels,
    )

    testds_new_channels = create_new_channels(
        test_single_cyclone_indexes, 
        testds, 
        testinfo, 
        generated_channels=generated_channels,
    )

    print("\nPreparing datasets (lazy)...\n")

    # Construir índices para remover as duas primeiras imagens e concatenar
    train_indices_to_keep = []
    valid_indices_to_keep = []
    test_indices_to_keep = []
    
    train_labels = []
    valid_labels = []
    test_labels = []

    for id, cyc_idx in train_single_cyclone_indexes:
        cyc_idx = cyc_idx[2:]  # Remove as duas primeiras imagens
        train_indices_to_keep.extend(cyc_idx)
        train_labels.append(traininfo.iloc[cyc_idx])

    for id, cyc_idx in valid_single_cyclone_indexes:
        cyc_idx = cyc_idx[2:]  # Remove as duas primeiras imagens
        valid_indices_to_keep.extend(cyc_idx)
        valid_labels.append(validinfo.iloc[cyc_idx])

    for id, cyc_idx in test_single_cyclone_indexes:
        cyc_idx = cyc_idx[2:]  # Remove as duas primeiras imagens
        test_indices_to_keep.extend(cyc_idx)
        test_labels.append(testinfo.iloc[cyc_idx])

    # Usar slicing direto do dask array (evita carregar tudo na memória)
    print("Slicing train dataset images...")
    train_indices_to_keep = np.array(sorted(set(train_indices_to_keep)))
    train_images = trainds[train_indices_to_keep]
    
    print("Slicing validation dataset images...")
    valid_indices_to_keep = np.array(sorted(set(valid_indices_to_keep)))
    valid_images = validds[valid_indices_to_keep]

    print("Slicing test dataset images...")
    test_indices_to_keep = np.array(sorted(set(test_indices_to_keep)))
    test_images = testds[test_indices_to_keep]

    print("Concatenating train dataset labels...")
    train_labels = pd.concat(train_labels, axis=0)
    train_labels = train_labels.reset_index(drop=True)

    print(f"Train dataset shape: {train_images.shape}")

    # Concatena os canais originais com os novos canais usando dask
    print("Concatenating channels for train dataset...")
    train_images = da.concatenate((train_images, trainds_new_channels), axis=-1)
    print(f"Train dataset new shape: {train_images.shape}\n")

    print("Concatenating validation dataset labels...")
    valid_labels = pd.concat(valid_labels, axis=0)
    valid_labels = valid_labels.reset_index(drop=True)

    print(f"Validation dataset shape: {valid_images.shape}")

    # Concatena os canais originais com os novos canais usando dask
    print("Concatenating channels for validation dataset...")
    valid_images = da.concatenate((valid_images, validds_new_channels), axis=-1)
    print(f"Validation dataset new shape: {valid_images.shape}\n")

    print("Concatenating test dataset labels...")
    test_labels = pd.concat(test_labels, axis=0)
    test_labels = test_labels.reset_index(drop=True)

    print(f"Test dataset shape: {test_images.shape}")

    # Concatena os canais originais com os novos canais usando dask
    print("Concatenating channels for test dataset...")
    test_images = da.concatenate((test_images, testds_new_channels), axis=-1)
    print(f"Test dataset new shape: {test_images.shape}\n")

    ds = (
        (train_images, train_labels),
        (valid_images, valid_labels),
        (test_images, test_labels)
    )
    return ds

def load_data():
    dsfiles = ["TCIR-ATLN_EPAC_WPAC.h5", "TCIR-ALL_2017.h5", "TCIR-CPAC_IO_SH.h5"]
    dspaths = [f"{data_path}/{file}" for file in dsfiles]
    files = [h5py.File(file, mode="r") for file in dspaths]
    data = [da.from_array(file["matrix"]) for file in files]
    info = [pd.read_hdf(file, key="info", mode="r") for file in dspaths]

    data = da.concatenate(data, axis=0) # Lazy loaded
    info = pd.concat(info).reset_index(drop=True)
    return data, info


def load_normalized_data(
    channels: List[int], 
    img_w: int,
) -> Tuple[
    Tuple[da.Array, pd.DataFrame], 
    Tuple[da.Array, pd.DataFrame], 
    Tuple[da.Array, pd.DataFrame]
]:
    """
    Carrega e normaliza dados mantendo dask arrays para economizar memória.
    Compute acontece apenas quando os dados são escritos no arquivo.
    """
    print("Loading data...")

    data, info = load_data()
    trainds, traininfo = get_train_data(data, info)

    rotation_width = calculate_img_ration_w(img_w)
    trainds = cut_images(trainds, rotation_width)

    print(f"Removing nan values and values larger than 1000 from train dataset")
    trainds = clear_images(trainds)

    print("Calculating means...")
    means = [da.mean(trainds[:, :, :, i]).compute() for i in channels]
    print(f"Means values: {means}")

    print("Calcularing standard deviations...")
    std = [da.std(trainds[:, :, :, i]).compute() for i in channels]
    print(f"Standard deviation values: {std}")

    validds, validinfo = get_valid_data(data, info)
    testds, testinfo = get_test_data(data, info)

    validds = cut_images(validds, rotation_width)
    testds = cut_images(testds, rotation_width)

    validds = clear_images(validds)
    testds = clear_images(testds)

    # Manter como dask arrays! Só selecionar os canais
    trainds = trainds[:, :, :, channels]
    validds = validds[:, :, :, channels]
    testds = testds[:, :, :, channels]

    print("Normalizing images (lazy)...")
    # Função auxiliar para normalizar um bloco
    def normalize_block(block, channel_means, channel_stds):
        """Normaliza um bloco de imagens"""
        normalized = np.copy(block)
        for i in range(normalized.shape[-1]):
            normalized[..., i] = (normalized[..., i] - channel_means[i]) / channel_stds[i]
        return normalized

    # Aplicar normalização lazy com map_blocks
    trainds = trainds.map_blocks(
        normalize_block,
        channel_means=np.array(means),
        channel_stds=np.array(std),
        dtype=trainds.dtype
    )
    
    validds = validds.map_blocks(
        normalize_block,
        channel_means=np.array(means),
        channel_stds=np.array(std),
        dtype=validds.dtype
    )
    
    testds = testds.map_blocks(
        normalize_block,
        channel_means=np.array(means),
        channel_stds=np.array(std),
        dtype=testds.dtype
    )

    return (trainds, traininfo), (validds, validinfo), (testds, testinfo)


def preprocess(channels, generated_channels, img_w, force=True):

    if not force:
        print("Checking if files exist")
        exists = all([os.path.isfile(file) for file in preprocessed_files])

        if exists:
            print("Files exist. Loading files...")
            with h5py.File(preprocessed_train, mode="r") as train:
                train_imgs = train["matrix"][:]
            
            train_labels = pd.read_hdf(preprocessed_train, key="info", mode="r")
            train = (train_imgs, train_labels)
            

            with h5py.File(preprocessed_valid, mode="r") as valid:
                valid_imgs = valid["matrix"][:]
            
            valid_labels = pd.read_hdf(preprocessed_valid, key="info", mode="r")
            valid = (valid_imgs, valid_labels)

            with h5py.File(preprocessed_test, mode="r") as test:
                test_imgs = test["matrix"][:]
            
            test_labels = pd.read_hdf(preprocessed_test, key="info", mode="r")
            test = (test_imgs, test_labels)

            print(f"\nTrain images shape: {train_imgs.shape}")
            print(f"Train labels shape: {train_labels.shape}")
            print(f"\nValidation images shape: {valid_imgs.shape}")
            print(f"Validation labels shape: {valid_labels.shape}")
            print(f"\nTest images shape: {test_imgs.shape}")
            print(f"Test labels shape: {test_labels.shape}")
            return train, valid, test

    normalized_data = load_normalized_data(channels, img_w)
    (trainds, traininfo), (validds, validinfo), (testds, testinfo) = normalized_data

    print(f"Train dataset shape: {trainds.shape}")
    print(f"Train info shape: {traininfo.shape}\n")
    print(f"Validation dataset shape: {validds.shape}")
    print(f"Validation info shape: {validinfo.shape}\n")
    print(f"Test dataset shape: {testds.shape}")
    print(f"Test info shape: {testinfo.shape}\n")

    normalized_data = add_channels(generated_channels, normalized_data)

    return normalized_data


def save_preprocessed(channels=[0, 3], generated_channels=[0], img_w=64, data=None):
    """
    Salva dados pré-processados em arquivos HDF5.
    Realiza compute apenas durante a escrita no arquivo para economizar memória.
    """
    if not data:
        # Se não fornecido, pré-processa os dados
        (
            (train_imgs, train_labels),
            (valid_imgs, valid_labels),
            (test_imgs, test_labels),
        ) = preprocess(channels, generated_channels, img_w, force=True)
    else:
        # Usa os dados fornecidos
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

    # Define as formas para os datasets HDF5
    img_new_shape = train_imgs.shape[1:]
    labels_new_shape = train_labels.shape[1:]

    img_max_shape = (None,) + img_new_shape

    img_new_shape = (0,) + img_new_shape
    labels_new_shape = (0,) + labels_new_shape

    os.makedirs(preprocessed_path, exist_ok=True)

    print("Writing training file...")
    print(train_imgs.shape)
    print(train_labels.shape)
    # Salva o conjunto de treino - compute apenas na escrita
    with h5py.File(preprocessed_train, mode="w") as train:
        if "matrix" not in train:
            train.create_dataset("matrix", shape=img_new_shape, maxshape=img_max_shape, chunks=True)
        
        # Compute e salva em chunks para economizar memória
        if isinstance(train_imgs, da.Array):
            train_imgs_computed = train_imgs.compute()
        else:
            train_imgs_computed = train_imgs
            
        train["matrix"].resize(train["matrix"].shape[0] + train_imgs_computed.shape[0], axis=0)
        train["matrix"][-train_imgs_computed.shape[0] :] = train_imgs_computed

    train_labels.to_hdf(preprocessed_train, key="info", mode="a")
        

    print("Writing validation file...")
    print(valid_imgs.shape)
    print(valid_labels.shape)
    # Salva o conjunto de validação
    with h5py.File(preprocessed_valid, mode="w") as valid:
        if "matrix" not in valid:
            valid.create_dataset("matrix", shape=img_new_shape, maxshape=img_max_shape, chunks=True)
        
        # Compute e salva em chunks para economizar memória
        if isinstance(valid_imgs, da.Array):
            valid_imgs_computed = valid_imgs.compute()
        else:
            valid_imgs_computed = valid_imgs
            
        valid["matrix"].resize(valid["matrix"].shape[0] + valid_imgs_computed.shape[0], axis=0)
        valid["matrix"][-valid_imgs_computed.shape[0] :] = valid_imgs_computed

    valid_labels.to_hdf(preprocessed_valid, key="info", mode="a")

    print("Writing test file...")
    print(test_imgs.shape)
    print(test_labels.shape)
    # Salva o conjunto de teste
    with h5py.File(preprocessed_test, mode="w") as test:
        if "matrix" not in test:
            test.create_dataset("matrix", shape=img_new_shape, maxshape=img_max_shape, chunks=True)
        
        # Compute e salva em chunks para economizar memória
        if isinstance(test_imgs, da.Array):
            test_imgs_computed = test_imgs.compute()
        else:
            test_imgs_computed = test_imgs
            
        test["matrix"].resize(test["matrix"].shape[0] + test_imgs_computed.shape[0], axis=0)
        test["matrix"][-test_imgs_computed.shape[0] :] = test_imgs_computed

    test_labels.to_hdf(preprocessed_test, key="info", mode="a")


def main():
    # Função principal para executar o pré-processamento e salvamento
    save_preprocessed()


if __name__ == "__main__":
    main()