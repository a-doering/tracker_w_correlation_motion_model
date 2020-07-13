from tracktor.config import cfg
import torch
from tracktor.frcnn_fpn import FRCNN_FPN

import configparser
import h5py 
import os.path as osp
from PIL import Image
import numpy as np
import csv
import copy

def enlarge_boxes(bb, boxes_enlargement_factor):
    """Enlarges bounding box widht and height by some factor."""
    bb = bb.copy()
    if boxes_enlargement_factor > 1.0:
        delta = (boxes_enlargement_factor - 1) / 2

        width_delta = (bb[2] - bb[0]) * delta
        height_delta = (bb[3] - bb[1]) * delta

        bb[0] -= width_delta
        bb[1] -= height_delta
        bb[2] += width_delta
        bb[3] += height_delta

    return bb

def clip_boxes_to_image(bb, size):
    """Clips boxes to size"""
    bb = bb.copy()
    height, width = size
    bb[0] = np.clip(bb[0], 0, width)
    bb[1] = np.clip(bb[1], 0, height)
    bb[2] = np.clip(bb[2], 0, width)
    bb[3] = np.clip(bb[3], 0, height)
    return bb

def create_dataset(boxes_enlargement_factor, vis_threshold, sequences, append=False, verbose=False, truncate=False):
    """Create a dataset for the correlation layer"""
    # Open hdf5 file and create arrays
    print(100*'#')
    filename = 'correlation_dataset_{:.2f}_{:.2f}.hdf5'.format(boxes_enlargement_factor, vis_threshold)
    h5_file = osp.join(cfg.DATA_DIR, 'correlation_dataset', 'dataset_more_info', filename)

    if append:
        h5 = h5py.File(h5_file, mode='a')
    else:
        h5 = h5py.File(h5_file, mode='w')

    for seq in sequences:

        print(100*'#')
        print("Sequence: {}".format(seq))
        if seq[3:5] == "17":
            mot_dir = osp.join(cfg.DATA_DIR, 'MOT17Det', 'train')
        elif seq[3:5] == "20":
            mot_dir = osp.join(cfg.DATA_DIR, 'MOT20', 'train')
        else: 
            raise Exception("No valid MOT challenge.")

        # We want to overwrite all sequences listed
        # Mostly doing appending if the previous sequence has not finished during the creation
        if append:
            print("Keys in the sequence: {}".format(h5.keys()))
            if seq in h5.keys():
                del h5[seq]
                print("Keys in sequence after deletion: {}".format(h5.keys()))
        

        seq_im_dir = osp.join(mot_dir, seq)
        gt_file = osp.join(mot_dir, seq, 'gt', 'gt.txt')
        config_file = osp.join(mot_dir, seq, 'seqinfo.ini')

        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])
        imDir = config['Sequence']['imDir']
        imWidth = int(config['Sequence']['imWidth'])
        imHeight = int(config['Sequence']['imHeight'])

        # Constructing tracks, one sample per frame. Access boxes like: boxes[frame][id]
        total = {}
        boxes = {}
        visibility = {}

        for i in range(1, seqLength+1):
            boxes[i] = {}
            visibility[i] = {}

        no_gt = False
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    # class person, certainity 1, visibility >= 0.25
                    if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= vis_threshold:
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

            total[i] = sample

        # Truncate sequence length
        if truncate and seqLength > 1050:
            seqLength = seqLength % 1000
            print("Seqlength truncated to {} save memory.".format(seqLength))

        # Find all ids that are in this frame and next
        id_in_frame_and_next = {}
        for i in range(1, seqLength):
            id_in_frame_and_next[i] = set(id_in_frame[i]).intersection(id_in_frame[i+1])
            #print(id_in_frame_and_next[i])

        num_pairs = sum([len(x) for x in id_in_frame_and_next.values()])
        num_detections = sum([len(x) for x in id_in_frame.values()])
        print("Number of detections: {}".format(num_detections))
        print("Number of pairs: {}".format(num_pairs))
        # Create a group per sequence. 
        # This allows fixed size shape for the datasets of size sum()
        group = h5.create_group(seq)
        h5_dataset_1 = group.create_dataset("fmap_prev", (num_pairs, 256, 7,7), dtype=np.float32)
        h5_dataset_2 = group.create_dataset("fmap_enlarged", (num_pairs, 256, 7,7), dtype=np.float32)
        h5_dataset_3 = group.create_dataset("boxes_next",(num_pairs, 4), dtype=np.float32)
        h5_dataset_4 = group.create_dataset("boxes",(num_pairs, 4), dtype=np.float32)
        h5_dataset_5 = group.create_dataset("boxes_enlarged",(num_pairs, 4), dtype=np.float32)
        h5_dataset_6 = group.create_dataset("names",(num_pairs,),dtype=h5py.special_dtype(vlen=str))
        h5_dataset_7 = group.create_dataset("names_next",(num_pairs,),dtype=h5py.special_dtype(vlen=str))
        h5_dataset_8 = group.create_dataset("preprocessed_image_sizes", (num_pairs, 2), dtype=np.float32)
        h5_dataset_9 = group.create_dataset("original_image_sizes",(num_pairs, 2), dtype=np.float32)
        
        pairs_stored = 0
        # Create feature maps
        for i in range(1, seqLength):
            if i%100 == 0:
                print("{:04} frames done in this sequence".format(i))
            if verbose:
                print('####### Frame {:04} from sequence {} #######'.format(i, seq))
                print(id_in_frame_and_next[i])
            pairs_in_frame= len(id_in_frame_and_next[i])

            # Skip in these cases, prevent error
            if pairs_in_frame == 0:
                continue
            boxes = [total[i]['gt'][id] for id in id_in_frame_and_next[i]] # bounding boxes of the previous frame
            # for id in id_in_frame_and_next[i]:
            #     print(id)

            boxes_next = [total[i+1]['gt'][id] for id in id_in_frame_and_next[i]]
            enlarged_boxes = [clip_boxes_to_image(enlarge_boxes(total[i]['gt'][id], boxes_enlargement_factor), (imHeight, imWidth))  for id in id_in_frame_and_next[i]]# enlarged bounding boxes of the previous frame
            # print(boxes[0])
            # print(enlarged_boxes[0])
            # print(boxes_next[0])
            # print(40*'+')
            # Sequence Name _ frame (of first image) _ id
            # MOT17-02_000001_000001
            names = [seq + '_{:06}_'.format(i) + '{:06}'.format(id) for id in id_in_frame_and_next[i]]
            names_next = [seq + '_{:06}_'.format(i+1) + '{:06}'.format(id) for id in id_in_frame_and_next[i]]

            h5_dataset_3[pairs_stored:pairs_stored+pairs_in_frame] = boxes_next
            h5_dataset_4[pairs_stored:pairs_stored+pairs_in_frame] = boxes
            h5_dataset_5[pairs_stored:pairs_stored+pairs_in_frame] = enlarged_boxes
            h5_dataset_6[pairs_stored:pairs_stored+pairs_in_frame] = names
            h5_dataset_7[pairs_stored:pairs_stored+pairs_in_frame] = names_next           
            # print(names)
            # print(names_next)
            boxes = torch.tensor(boxes)
            enlarged_boxes = torch.tensor(enlarged_boxes)

            # Load and convert image
            pic = Image.open(osp.join(seq_im_dir, total[i]['im_path'])).convert("RGB")
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
            # Put it from HWC to CHW format
            img = img.permute((2, 0, 1)).contiguous()
            img = img.float().div(255)
            img.unsqueeze_(0)
            if verbose:
                print(img.shape, img.type, img.type())#1,3,1080,1920 ---> yes, torch object        
            obj_detect.load_image(img) # load previous frame
            obj_detect.load_image(img) # load current frame

            prev_7x7_features, current_7x7_features = obj_detect.get_feature_patches(boxes, enlarged_boxes)
            if verbose:
                print(prev_7x7_features.shape) # _7x7_features.shape -> [#boxes, 256, 7, 7]

            h5_dataset_1[pairs_stored:pairs_stored+pairs_in_frame] = prev_7x7_features.data.cpu().numpy()
            h5_dataset_2[pairs_stored:pairs_stored+pairs_in_frame] = current_7x7_features.data.cpu().numpy()
            h5_dataset_8[pairs_stored:pairs_stored+pairs_in_frame] = obj_detect.preprocessed_images.image_sizes
            h5_dataset_9[pairs_stored:pairs_stored+pairs_in_frame] = obj_detect.original_image_sizes

            if i%100 == 0:
                print("Image sizes: ")
                print(40*'-')
                print(obj_detect.preprocessed_images.image_sizes)
                print(40*'-')
                print(obj_detect.original_image_sizes)

            pairs_stored += pairs_in_frame
            if verbose:
                print(pairs_stored)

        # # For developing, only do the first couple.
        #     if i == 101:
        #         break
        # if seq == 'MOT17-02':
        #     break
    h5.close()
    print(100*'#')
    print("Finished creating dataset {}".format(filename))

# Load model
print('Loading model...')
obj_detect = FRCNN_FPN(num_classes=2, correlation_head=None)
obj_detect.load_state_dict(torch.load("output/faster_rcnn_fpn_training_mot_17/model_epoch_27_original.model",
                            map_location=lambda storage, loc: storage))
obj_detect.eval()
obj_detect.cuda()
print('Model loaded!')

# Hardcoded loader for MOT17
# sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05' ,'MOT17-02','MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
sequences = ['MOT20-01','MOT20-02', 'MOT20-03', 'MOT20-05','MOT17-02','MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']

# Hardcoded parameters
vis_threshold = [0.5]
boxes_enlargement_factor = [1.5]#, 2.0, 1.2]#, 1.5, 1.0, 1.05, 1.1,1.3,2.0]#[1.0,1.05, 1.1, 1.2,1.3,1.5,2.0]


for b in boxes_enlargement_factor:
    for v in vis_threshold:
        create_dataset(b,v, sequences, append=False, verbose=False, truncate=True)