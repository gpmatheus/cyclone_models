import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

hdf5_file = "/Users/matheussonego/Documents/Unipampa/tcc/cyclone_models/data/TCIR-ATLN_EPAC_WPAC.h5"
dataset_key = "matrix"

images_file = h5py.File(hdf5_file, mode='r')['matrix']
labels_file = pd.read_hdf(hdf5_file, key='info', mode='r')[["ID", "Vmax", "time"]]

# ids of cyclones
ids = list(labels_file['ID'].unique())

# state
index = {"idindex": 0, "index": 0, "channel": 0}

f, arr = plt.subplots(1, 3)
plt.subplots_adjust(bottom=0.2)  # espaço pros botões

indexes = labels_file[labels_file['ID'] == ids[index["index"]]].index.tolist()
images = images_file[indexes]
labels = labels_file.iloc[indexes]

def repaint():
    arr[0].imshow(images[index["index"], :, :, index["channel"]], cmap="gray")
    arr[0].set_title(f'[{index["index"]}]')
    arr[0].axis('off')

    arr[1].imshow(images[index["index"] + 1, :, :, index["channel"]], cmap="gray")
    arr[1].set_title(f'[{index["index"] + 1}]')
    arr[1].axis('off')

    arr[2].imshow(images[index["index"] + 2, :, :, index["channel"]], cmap="gray")
    arr[2].set_title(f'[{index["index"] + 2}]')
    arr[2].axis('off')

    plt.suptitle(ids[index["idindex"]])

    f.canvas.draw_idle()

repaint()

def change_cyclone():
    print("repainting...")
    global indexes
    global images
    global labels
    indexes = labels_file[labels_file['ID'] == ids[index["idindex"]]].index.tolist()
    images = images_file[indexes]
    labels = labels_file.iloc[indexes]
    index["index"] = 0
    repaint()

def set_channel(channel):
    print("changing channel...")
    index["channel"] = channel
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

with h5py.File(hdf5_file, mode='r') as file:
    shape = file['matrix'].shape
nchannels = shape[-1]


# UI

button_w, button_h = 0.1, 0.075

# build channels hud
channel_buttons_hud_w = .5
starts = np.linspace(
    .5 - channel_buttons_hud_w / 2, 
    .5 + channel_buttons_hud_w / 2 - button_w, 
    nchannels
)
channel_buttons = []
for i, a in enumerate(starts):
    bt = Button(plt.axes([a, 0.0, button_w, button_h]), f"{i}")
    bt.on_clicked(lambda _, i=i: set_channel(i))
    channel_buttons.append(bt)


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

file.close()