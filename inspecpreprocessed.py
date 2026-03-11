import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import datetime

hdf5_file = "/Users/matheussonego/Documents/Unipampa/tcc/cyclone_models/data/preprocessed/train.h5"
dataset_key = "matrix"

generated_channels = [0]
channels = [0, 2]
show_generated_channels = False

images_file = h5py.File(hdf5_file, mode='r')['matrix']
labels_file = pd.read_hdf(hdf5_file, key='info', mode='r')[["ID", "Vmax", "time"]]

# ids of cyclones
ids = list(labels_file[:]['ID'].unique())

# state
index = {"idindex": 0, "index": 0}


subplots_channels = len(channels) + (len(generated_channels) if show_generated_channels else 0)
f, arr = plt.subplots(subplots_channels, 3)
plt.subplots_adjust(bottom=0.2)  # espaço pros botões

sub_info = labels_file[labels_file['ID'] == ids[index["idindex"]]]

sub_info = sub_info.sort_values("time")

print(sub_info)

indexes = sub_info.index

print(indexes)

images = images_file[indexes]
labels = labels_file.iloc[indexes]
print(f'Cyclone {ids[index["idindex"]]} with size {len(indexes)}')

# now sort
labels = labels.sort_values("time").reset_index(drop=True)
images = images[labels.index]

white_img = np.full(images.shape[1: -1], 1.0)

def repaint():

    for i, ch in enumerate(channels):
        
        arr[i][0].imshow(images[index["index"], :, :, ch], cmap="gray")
        if i == 0:
            arr[i][0].set_title(f'[{index["index"]}] - {labels.iloc[index["index"]]["time"]}')
        arr[i][0].axis('off')

        arr[i][1].imshow(images[index["index"] + 1, :, :, ch], cmap="gray")
        if i == 0:
            arr[i][1].set_title(f'[{index["index"] + 1}] - {labels.iloc[index["index"] + 1]["time"]}')
        arr[i][1].axis('off')

        arr[i][2].imshow(images[index["index"] + 2, :, :, ch], cmap="gray")
        if i == 0:
            arr[i][2].set_title(f'[{index["index"] + 2}] - {labels.iloc[index["index"] + 2]["time"]}')
        arr[i][2].axis('off')


    plt.suptitle(ids[index["idindex"]])

    f.canvas.draw_idle()

repaint()

def change_cyclone():
    global indexes
    global images
    global labels
    indexes = labels_file[labels_file['ID'] == ids[index["idindex"]]].index.tolist()
    indexes = list(set(indexes))
    indexes.sort()
    images = images_file[indexes]
    labels = labels_file.iloc[indexes]
    labels = labels.sort_values("time").reset_index(drop=True)
    images = images[labels.index]
    index["index"] = 0
    repaint()

def previous_cyclone():
    print("going to previous cyclone...")
    index["idindex"] = index["idindex"] - 1 if index["idindex"] > 0 else index["idindex"]
    change_cyclone()

def next_cyclone():
    print("going to next cyclone...")
    index["idindex"] = index["idindex"] + 1 if index["idindex"] + 1 < len(ids) else index["idindex"]
    change_cyclone()

def left_image():
    print("going to left image...")
    index["index"] = index["index"] - 1 if index["index"] > 0 else index["index"]
    repaint()

def right_image():
    print("going to next image...")
    index["index"] = index["index"] + 1 if index["index"] + 3 < images.shape[0] else index["index"]
    repaint()


# UI

button_w, button_h = 0.1, 0.075

# build cyclones hud
ch = plt.axes([0.0, .55, button_w, button_h])
previous = Button(ch, "Previous")
previous.on_clicked(lambda _: previous_cyclone())

ch = plt.axes([0.0, .45, button_w, button_h])
next = Button(ch, "Next")
next.on_clicked(lambda _: next_cyclone())

# build image hud
ch = plt.axes([0.0, .1, button_w, button_h])
left = Button(ch, "Left")
left.on_clicked(lambda _: left_image())

ch = plt.axes([.12, .1, button_w, button_h])
right = Button(ch, "Right")
right.on_clicked(lambda _: right_image())


plt.show()
