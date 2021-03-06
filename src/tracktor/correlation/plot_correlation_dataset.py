from os import path as osp
import os

import numpy as np
import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
import h5py
from tracktor.config import cfg

from torch.utils.data import DataLoader
from tracktor.datasets.dataloader_correlation import Dataset

def im_name_to_im_path(im_name):
    """Convert im_name from h5py dataset to im_path"""
    seq , image_number, id = im_name.split("_")

    if seq[3:5] == "17":
        im_path = osp.join(cfg.DATA_DIR, 'MOT17Det', 'train', seq, "img1", image_number + ".jpg")
    elif seq[3:5] == "20":
        im_path = osp.join(cfg.DATA_DIR, 'MOT20', 'train', seq, "img1", image_number + ".jpg")
    else: 
        raise Exception("No valid MOT challenge.")
    return im_path

def plot_boxes_one_pair(sample, step, predictions=None, save=False, output_dir=None):
    """Plot boxes on image"""
    _, _, boxes_gt, boxes, boxes_enlarged, im_name_prev, im_name_current, _, _ = sample
    
    # Unpacking batch (first element)
    boxes_gt = boxes_gt.squeeze()
    boxes = boxes.squeeze()
    boxes_enlarged = boxes_enlarged.squeeze()
    im_name_prev = im_name_prev[0]
    im_name_current = im_name_current[0]

    assert im_name_prev.split("_")[-1] == im_name_current.split("_")[-1]
    track_id = im_name_prev.split("_")[-1]

    if save:
        if output_dir is None:
            output_dir = osp.join(cfg.ROOT_DIR, "output", "validation_images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    #################################        Prev image plotting        #################################
    im_path_prev = im_name_to_im_path(im_name_prev)
    cmap = ['cyan', 'cyan']
    linestyle = ['solid', 'dashed']
    boxes_to_print = [boxes, boxes_enlarged]
    output_name = osp.join(output_dir, str(step) + "_" + im_name_current + "_prev.jpg") if output_dir else None

    prev_image = plot_image(im_path_prev, boxes_to_print, cmap, track_id, output_name, linestyle)
    if save: print(f"Saved image {output_name}")

    #################################       Current image plotting       #################################
    im_path_current = im_name_to_im_path(im_name_current)
    cmap = ['r', 'cyan', 'm']
    linestyle = ['solid', 'solid', 'solid']
    boxes_to_print = [boxes_gt, boxes_enlarged, predictions]
    if predictions is None:
        boxes_to_print = boxes_to_print[:-1]
        cmap = cmap[:-1]
        linestyle = linestyle[:-1]
    output_name = osp.join(output_dir, str(step) + "_" + im_name_current + ".jpg") if output_dir else None

    current_image = plot_image(im_path_current, boxes_to_print, cmap, track_id, output_name, linestyle)
    if save: print(f"Saved image {output_name}")

    return prev_image, current_image

def plot_tracktor_image(blob, tracks, ids, name):
    image = blob['img'].squeeze().permute(1, 2, 0).numpy()

    splited_path = blob['img_path'][0].split("/")
    seq_name = splited_path[-3]
    output_dir = osp.join(cfg.ROOT_DIR, "output", "debug_images", seq_name)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    img_name = splited_path[-1].split(".")
    file_name = osp.join(output_dir, img_name[0] + "_" + name + "." + img_name[-1])

    cmap = ['r'] * len(ids)

    plot_image(image, tracks, cmap, ids, file_name)

def plot_image(base_im, boxes_to_print, cmap, track_id, output_name=None, linestyle=None):
    image = base_im
    if isinstance(image, str):
        image = cv2.imread(image)
        image = image[:, :, (2, 1, 0)]

    height, width = image.shape[:2]

    fig = plt.figure()
    fig.set_size_inches(width / 100.0, height / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image)

    if linestyle is None:
        linestyle = ['solid'] * len(boxes_to_print)

    if isinstance(track_id, int):
        track_id = [track_id] + [None] * len(boxes_to_print) - 1

    for i, (box, c, style, t_id) in enumerate(zip(boxes_to_print, cmap, linestyle, track_id)):
        ax.add_patch(
            plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                linewidth=2.0, color=c,
                linestyle=style
            ))

        if t_id is not None:
            ax.annotate(t_id, (box[0] + (box[2] - box[0]) / 2.0, box[1] + (box[3] - box[1]) / 2.0),
                    color='k', weight='bold', fontsize=18, ha='center', va='center')

    plt.axis('off')
    # plt.tight_layout()
    plt.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape((3,) + fig.canvas.get_width_height()[::-1])
    image = torch.from_numpy(image)

    if output_name:
        plt.savefig(output_name, dpi=100)

    plt.close()

    return image

if __name__ == "__main__":
    filename = 'correlation_dataset_1.50_0.50.hdf5'

    h5_path = osp.join(cfg.DATA_DIR, 'correlation_dataset', 'dataset_more_info', filename)
    db_val = Dataset(h5_path, ['MOT17-10'])
    val_loader = DataLoader(db_val, batch_size=1)

    for i, batch in enumerate(val_loader):
        pre_img, current_img = plot_boxes_one_pair(batch, save=True)
        if i >= 10: break
