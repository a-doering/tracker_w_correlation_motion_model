from os import path as osp

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import h5py

def im_name_to_im_path(im_name):
    """Convert im_name from h5py dataset to im_path"""
    seq , image_number, id = im_name.split("_")
    im_path = osp.join("data", "MOT17Det", "train", seq, "img1", image_number + ".jpg")
    return im_path

def plot_boxes_one_pair(h5_file, seq, index):
    """Plot boxes on image"""
    boxes = h5_file[f"/{seq}/boxes"][index]
    boxes_enlarged = h5_file[f"/{seq}/boxes_enlarged"][index]
    im_name = h5_file[f"/{seq}/names"][index]
    im_path = im_name_to_im_path(im_name)

    im_output = osp.join("output", "dataset_test", im_name + ".jpg")

    im = cv2.imread(im_path)
    im = im[:, :, (2, 1, 0)]

    sizes = np.shape(im)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / 100, height / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im)

    cmap = ['r', 'b', 'g']
    t = [boxes, boxes_enlarged]
    for i in range(len(t)):
        t_i = t[i]
        ax.add_patch(
            plt.Rectangle(
                (t_i[0], t_i[1]),
                t_i[2] - t_i[0],
                t_i[3] - t_i[1],
                fill=False,
                linewidth=2.0, **cmap[i]
            ))

        ax.annotate(j, (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                    color=cmap[i]['ec'], weight='bold', fontsize=18, ha='center', va='center')


        plt.axis('off')
        # plt.tight_layout()
        plt.draw()
        plt.savefig(im_output, dpi=100)
        plt.close()

h5_path = osp.join(data, 'correlation_dataset', dataset_more_info/correlation_dataset_1.50_0.50.hdf5)
h5_file = h5py.File(h5_path, "r")
sequences=['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10','MOT17-11', 'MOT17-13']
lengths = []
for seq in sequences:
    sample = h5_file[f"/{seq}/name"]
    lengths.append(sample.shape[0])

lenghts = np.array(lengths)
total_pairs = lengths.sum()

for pair in range(20):# max total_pairs
    index = pair
    for i, length in enumerate(lengths):
        if i > 0: index-=self.lengths[i-1]
        if index < length:
            seq = sequences[i]
    plot_boxes_one_pair(h5_file, seq, index)