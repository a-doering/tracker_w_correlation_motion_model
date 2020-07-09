from os import path as osp

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import h5py
from tracktor.config import cfg

def im_name_to_im_path(im_name):
    """Convert im_name from h5py dataset to im_path"""
    seq , image_number, id = im_name.split("_")
    im_path = osp.join(cfg.DATA_DIR, "MOT17Det", "train", seq, "img1", image_number + ".jpg")
    return im_path

def plot_boxes_one_pair(h5_file, seq, index, predictions=None):
    """Plot boxes on image"""
    boxes = h5_file[f"/{seq}/boxes"][index]
    boxes_enlarged = h5_file[f"/{seq}/boxes_enlarged"][index]
    boxes_next = h5_file[f"/{seq}/boxes_next"][index]
    im_name_0 = h5_file[f"/{seq}/names"][index]
    im_name_1 = h5_file[f"/{seq}/names_next"][index]

    print(im_name_0)
    # print(im_name_0)
    print(im_name_1)

    # print(boxes)
    # print(boxes_enlarged)
    # print(boxes_next)
    im_path_0 = im_name_to_im_path(im_name_0)
    print(im_path_0)
    id = im_name_0.split("_")[-1]
    im_output_0 = osp.join(cfg.ROOT_DIR, "output", "dataset_test", im_name_0 + ".jpg")

    im_0 = cv2.imread(im_path_0)
    im_0 = im_0[:, :, (2, 1, 0)]

    sizes = np.shape(im_0)
    height = float(sizes[0])
    width = float(sizes[1])
 
    fig = plt.figure()
    fig.set_size_inches(width / 100, height / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im_0)

    cmap = ['r', 'cyan']
    t = [boxes, boxes_enlarged]
    for i in range(len(t)):
        t_i = t[i]
        ax.add_patch(
            plt.Rectangle(
                (t_i[0], t_i[1]),
                t_i[2] - t_i[0],
                t_i[3] - t_i[1],
                fill=False,
                linewidth=2.0, color=cmap[i]
            ))

        ax.annotate(id, (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                    color=cmap[i], weight='bold', fontsize=18, ha='center', va='center')

    plt.axis('off')
    # plt.tight_layout()
    plt.draw()
    plt.savefig(im_output_0, dpi=100)
 #   plt.close()

    ##### Next image
    im_path_1 = im_name_to_im_path(im_name_1)
    print(im_path_1)

    id = im_name_1.split("_")[-1]
    # Needs image name zero to be sorted next to the previous one and not with the next frame
    im_output_1 = osp.join(cfg.ROOT_DIR, "output", "dataset_test", im_name_0 + '_next' + ".jpg")

    im_1 = cv2.imread(im_path_1)
    im_1 = im_1[:, :, (2, 1, 0)]

    sizes = np.shape(im_1)
    height = float(sizes[0])
    width = float(sizes[1])
    ax.imshow(im_1)

    cmap = ['w', 'cyan', 'm']
    t = [boxes_next, boxes_enlarged]
    if predictions is not None:
        t = [boxes_next, boxes_enlarged, predictions]
    for i in range(len(t)):
        t_i = t[i]
        ax.add_patch(
            plt.Rectangle(
                (t_i[0], t_i[1]),
                t_i[2] - t_i[0],
                t_i[3] - t_i[1],
                fill=False,
                linewidth=2.0, color=cmap[i]
            ))

        ax.annotate(id, (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                    color=cmap[i], weight='bold', fontsize=18, ha='center', va='center')

    plt.axis('off')
    # plt.tight_layout()
    plt.draw()
    plt.savefig(im_output_1, dpi=100)
    plt.close()

filename = 'correlation_dataset_1.50_0.50.hdf5'

h5_path = osp.join(cfg.DATA_DIR, 'correlation_dataset', 'dataset_more_info', filename)    
h5_file = h5py.File(h5_path, "r")
sequences=['MOT17-02']#, 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10','MOT17-11', 'MOT17-13']
lengths = []
for seq in sequences:
    sample = h5_file[f"/{seq}/names"]
    lengths.append(sample.shape[0])

lengths = np.array(lengths)
total_pairs = lengths.sum()

for pair in range(20):# max total_pairs
    index = pair
    for i, length in enumerate(lengths):
        if i > 0: index -= lengths[i-1]
        if index < length:
            seq = sequences[i]
            break
    plot_boxes_one_pair(h5_file, seq, index)