from ..config import cfg
import torch
from tracktor.frcnn_fpn import FRCNN_FPN
from torchvision.transforms import ToTensor

import configparser
import h5py 
import os.path as osp
from PIL import Image


# Load model
obj_detect = FRCNN_FPN(num_classes=2)
obj_detect.load_state_dict(torch.load("output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model",
                            map_location=lambda storage, loc: storage))
obj_detect.eval()
obj_detect.cuda()

# Hardcoded loader for MOT17
mot_dir = osp.join(cfg.DATA_DIR, 'MOT17Det', 'train')

sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
				         'MOT17-11', 'MOT17-13']

# Open hdf5 file and create arrays
h5_path = osp.join(cfg.DATA_DIR, 'correlation_dataset', 'correlation_dataset.hdf5')
h5 = h5py.File(h5_file, mode='w')

h5_dataset_1 = h5.create_dataset("fmap", maxshape=(None, 256, 7,7))
h5_dataset_2 = h5.create_dataset("fmap_enlarged", maxshape=(None, 256, 7,7))
h5_dataset_3 = h5.create_dataset("labels",maxshape=(None, 4))
h5_dataset_4 = h5.create_dataset("name", maxshape=(None,))


for seq in sequences:
    seq_im_dir = osp.join(mot_dir, seq, 'img1')
    gt_file = osp.join(mot_dir, seq, 'gt', 'gt.txt')
    config_file = osp.join(mot_dir, seq, 'seqinfo.ini')

    assert osp.exists(config_file), \
        'Config file does not exist: {}'.format(config_file)

    config = configparser.ConfigParser()
    config.read(config_file)
    seqLength = int(config['Sequence']['seqLength'])
    imDir = config['Sequence']['imDir']

    # Constructing tracks, one sample per frame. Access boxes like: boxes[frame][id]
    total = []
    boxes = {}

    for i in range(1, seqLength+1):
        boxes[i] = {}
        visibility[i] = {}

    no_gt = False
    if osp.exists(gt_file):
        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                # class person, certainity 1, visibility >= 0.25
                if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self._vis_threshold:
                    # Make pixel indexes 0-based, should already be 0-based (or not)
                    x1 = int(row[2]) - 1
                    y1 = int(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + int(row[4]) - 1
                    y2 = y1 + int(row[5]) - 1
                    bb = np.array([x1,y1,x2,y2], dtype=np.float32)
                    boxes[int(row[0])][int(row[1])] = bb
                    visibility[int(row[0])][int(row[1])] = float(row[8])
    else:
        no_gt = True

    id_in_frame = {}

    for i in range(1,seqLength+1):
        im_path = osp.join(imDir,"{:06d}.jpg".format(i))
        id_in_frame[i] = boxes[i].keys()
        sample = {'gt':boxes[i],
                    'im_path':im_path,
                    'vis':visibility[i]}

        total.append(sample)

    # Find all ids that are in this frame and next
    id_in_frame_and_next = {}
    for i in range(1, seqLength):
        id_in_frame_and_next[i] = set(id_in_frame[i]).intersection(id_in_frame[i+1])

    # Create feature maps
    for i in range(1, seqLength):
        boxes = [total[i][id] for id in id_in_frame_and_next[i]] # bounding boxes of the previous frame
        enlarged_boxes = [total[i+1][id] for id in id_in_frame_and_next[i]]# enlarged bounding boxes of the previous frame

        boxes = torch.tensor(boxes)
        enlarged_boxes = torch.tensor(enlarged_boxes)

        img = Image.open(total[i]['im_path']).convert("RGB")
        img = ToTensor(img)

        # Load_image computes the feature map of the given image
        # Every time you load an image, the last one is stored as previous
        obj_detect.load_image(img) # load previous frame
        img = Image.open(total[i+1]['im_path']).convert("RGB")
        img = ToTensor(img)
        obj_detect.load_image(img) # load current frame

        prev_7x7_features, current_7x7_features = obj_detect.get_feature_patches(boxes, enlarged_boxes)
    # _7x7_features.shape -> [#boxes, 256, 7, 7]

    # Store the outputs with ground truth
    # Save as:
    # Sequence Name _ frame (of first image) _ ID, e.g.
    # MOT17-02_000001_0055

