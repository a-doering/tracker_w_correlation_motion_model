import torch
import h5py
import numpy as np

class Dataset(torch.utils.data.Dataset):

    def __init__(self, h5_path, sequences=['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05','MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10','MOT17-11', 'MOT17-13']):

        self.train_folders = sequences
        self.file = h5py.File(h5_path, "r")
        
        self.lengths = []
        for seq in self.train_folders:
            sample = self.file[f"/{seq}/fmap_prev"]
            self.lengths.append(sample.shape[0])
            print(f"{seq} samples: [{self.lengths[-1]}]")

        self.lengths = np.array(self.lengths)
        self.total_samples = self.lengths.sum()
        print(f"total samples [{self.total_samples}]")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        for i, length in enumerate(self.lengths):
            if i > 0: index -= self.lengths[i-1]
            if index < length:
                seq = self.train_folders[i]
                break
        
        # print(f"seq [{seq}] new index [{index}]")
        fmap_prev = self.file[f"/{seq}/fmap_prev"][index]
        fmap_enlarged = self.file[f"/{seq}/fmap_enlarged"][index]
        gt_boxes = self.file[f"/{seq}/boxes_next"][index]

        boxes = self.file[f"/{seq}/boxes"][index]
        boxes_enlarged = self.file[f"/{seq}/boxes_enlarged"][index]
        im_name_prev = self.file[f"/{seq}/names"][index]
        im_name_current = self.file[f"/{seq}/names_next"][index]

        imWidth = self.file[f"/{seq}/imWidth"]
        imHeight = self.file[f"/{seq}/imHeight"]

        return fmap_prev, fmap_enlarged, gt_boxes, boxes, boxes_enlarged, im_name_prev, im_name_current, imWidth, imHeight