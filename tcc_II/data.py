import dask.array as da
import numpy as np
import h5py
import pandas as pd
import datetime
import os

# Definição dos caminhos para os dados
# Usa variáveis de ambiente se definidas, senão usa caminhos padrão
data_path = os.getenv("DATA_PATH") or "data"
preprocessed_path = os.getenv("PREPROCESSED_PATH") or "data/preprocessed"
print(f"folder_path: {preprocessed_path}")
preprocessed_train = f"{preprocessed_path}/train.h5"
preprocessed_valid = f"{preprocessed_path}/valid.h5"
preprocessed_test = f"{preprocessed_path}/test.h5"
preprocessed_files = [preprocessed_train, preprocessed_valid, preprocessed_test]


def split_data(images, info):
    # Divide os dados em conjuntos de treino, validação e teste baseado nos anos
    # Treino: 2003-2014, Validação: 2015-2016, Teste: 2017
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
    # Calcula o slice para cortar a imagem centralizada com a largura especificada
    start = images_shape[1] // 2 - width // 2
    end = images_shape[1] // 2 + width // 2
    return slice(start, end)


def cut_images(images, width):
    # Corta as imagens para o tamanho especificado, centralizando o corte
    slc = get_images_slice(images.shape, width)
    return images[:, slc, slc, :]


def compute(img, img_1, img_2):
    # Calcula a diferença absoluta entre imagens consecutivas para detecção de movimento
    return np.abs(img - (img_1 * 2) + img_2)


def load_normalized_data(channels, img_w):
    # Carrega e normaliza os dados de imagens de ciclones
    # channels: lista de canais a serem usados
    # img_w: largura da imagem final
    print("Loading data...")

    dsfiles = ["TCIR-ALL_2017.h5", "TCIR-ATLN_EPAC_WPAC.h5", "TCIR-CPAC_IO_SH.h5"]
    dspaths = [f"{data_path}/{file}" for file in dsfiles]

    files = [h5py.File(file, mode="r") for file in dspaths]
    data = [da.from_array(file["matrix"]) for file in files]
    info = [pd.read_hdf(file, key="info", mode="r") for file in dspaths]
    info = pd.concat(info)

    data = da.concatenate(data, axis=0)
    
    # Reset index para evitar duplicatas quando concatenando múltiplos arquivos
    info = info.reset_index(drop=True)

    print(f"Loaded dataset: Images {data.shape} - Labels {info.shape}")
    data = da.nan_to_num(data)
    data[data > 1000] = 0

    print("Calculating means...")
    means = [da.nanmean(data[:, :, :, i]).compute() for i in channels]

    print(f"Means values: {means}")

    print("Calcularing standard deviations...")
    std = [da.std(data[:, :, :, i]).compute() for i in channels]

    print(f"Standard deviation values: {std}")

    # Calcula a largura para rotação (diagonal da imagem quadrada)
    rotation_width = int(np.ceil(np.sqrt((img_w**2) * 2)))
    if rotation_width % 2 != 0:
        rotation_width += 1

    print("Cropping images...")
    data = cut_images(data, rotation_width)

    print("Normalizing images...")
    # Normaliza cada canal subtraindo a média e dividindo pelo desvio padrão
    for chan in range(len(channels)):
        data[:, :, :, chan] -= means[chan]
        data[:, :, :, chan] /= std[chan]

    # Seleciona apenas os canais especificados
    data = np.array(data[:, :, :, channels])

    # Fecha os arquivos HDF5
    for file in files:
        file.close()

    return data, info


def preprocess(channels, generated_channels, img_w, force=True):
    # Pré-processa os dados, gerando novos canais e dividindo em treino/validação/teste
    # channels: canais originais a usar
    # generated_channels: canais para gerar novos (diferença absoluta)
    # img_w: largura da imagem
    # force: força o pré-processamento mesmo se arquivos existirem

    if not force:
        print("Checking if files exist")
        exists = all([os.path.isfile(file) for file in preprocessed_files])

        if exists:
            print("Files exist. Loading files...")
            # Carrega os dados pré-processados se já existirem
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
    # Lista para armazenar índices de imagens de cada ciclone
    single_cyclone_indexes = []

    """
    fills the single_cyclone_indexes with the sorted indexes of cyclones
    """
    print("Filling ids...")
    # Preenche a lista com índices ordenados por tempo para cada ciclone
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
    # Cria novos canais calculando diferenças absolutas entre imagens consecutivas
    cyclone_new_channels = []

    inittotal = 0
    endtotal = 0
    for id, indexes in single_cyclone_indexes:

        images = data[indexes]
        labels = info.iloc[indexes]

        # sort values
        labels = labels.sort_values("time").reset_index(drop=True)
        images = images[labels.index]

        # add sorted labels to sorted_labels list
        # labels = labels[2:]

        print(f"Generating channels for cyclone {id} of {len(indexes)} images", end="")
        sum = 0
        for idx in range(2, len(indexes)):
            # Para cada imagem a partir da terceira, calcula novos canais
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
            sum += 1
        print(f" - {len(indexes)} -> {sum}")
        inittotal += len(indexes)
        endtotal += sum
    
    print(f"Init total: {inittotal}")
    print(f"End total: {endtotal}")

    cyclone_new_channels = np.array(cyclone_new_channels)
    if len(cyclone_new_channels.shape) < 4:
        cyclone_new_channels = np.expand_dims(cyclone_new_channels, axis=-1)

    """
    load cyclones
    """
    print("\nLoading processed images\n")
    # Carrega as imagens originais (a partir da terceira de cada ciclone)
    images = ()
    labels = []
    for id, cyc_idx in single_cyclone_indexes:
        print(cyc_idx)
        cyc_idx = cyc_idx[2:]  # Remove as duas primeiras imagens
        imgs = data[cyc_idx]
        lbls = info.iloc[cyc_idx]
        images += (imgs,)
        labels.append(lbls)
    
    print("Concatenating images...")
    images = np.concatenate(images, axis=0)

    print("Concatenating labels...")
    labels = pd.concat(labels, axis=0)
    labels = labels.reset_index(drop=True)

    print(f"\nData shape: {images.shape}")

    # Concatena os canais originais com os novos canais gerados
    images = np.concatenate((images, cyclone_new_channels), axis=-1)
    print(f"Data shape: {images.shape}")

    # Divide em treino, validação e teste
    train, valid, test = split_data(images, labels)
    return train, valid, test


def save_preprocessed(channels=[0, 3], generated_channels=[0], img_w=64, data=None):
    # Salva os dados pré-processados em arquivos HDF5
    # channels: canais originais
    # generated_channels: canais gerados
    # img_w: largura da imagem
    # data: dados pré-processados (opcional)

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
    labels_max_shape = (None,) + labels_new_shape

    img_new_shape = (0,) + img_new_shape
    labels_new_shape = (0,) + labels_new_shape

    os.makedirs(data_path, exist_ok=True)

    print("Writing traininig file...")
    print(train_imgs.shape)
    print(train_labels.shape)
    # Salva o conjunto de treino
    with h5py.File(preprocessed_train, mode="w") as train:

        if "matrix" not in train:
            train.create_dataset("matrix", shape=img_new_shape, maxshape=img_max_shape)
        train["matrix"].resize(train["matrix"].shape[0] + train_imgs.shape[0], axis=0)
        train["matrix"][-train_imgs.shape[0] :] = train_imgs

    train_labels.to_hdf(preprocessed_train, key="info", mode="a")
        

    print("Writing validation file...")
    print(valid_imgs.shape)
    print(valid_labels.shape)
    # Salva o conjunto de validação
    with h5py.File(preprocessed_valid, mode="w") as valid:

        if "matrix" not in valid:
            valid.create_dataset("matrix", shape=img_new_shape, maxshape=img_max_shape)
        valid["matrix"].resize(valid["matrix"].shape[0] + valid_imgs.shape[0], axis=0)
        valid["matrix"][-valid_imgs.shape[0] :] = valid_imgs

    valid_labels.to_hdf(preprocessed_valid, key="info", mode="a")

    print("Writing test file...")
    print(test_imgs.shape)
    print(test_labels.shape)
    # Salva o conjunto de teste
    with h5py.File(preprocessed_test, mode="w") as test:

        if "matrix" not in test:
            test.create_dataset("matrix", shape=img_new_shape, maxshape=img_max_shape)
        test["matrix"].resize(test["matrix"].shape[0] + test_imgs.shape[0], axis=0)
        test["matrix"][-test_imgs.shape[0] :] = test_imgs

    test_labels.to_hdf(preprocessed_test, key="info", mode="a")


def main():
    # Função principal para executar o pré-processamento e salvamento
    save_preprocessed()


if __name__ == "__main__":
    main()
