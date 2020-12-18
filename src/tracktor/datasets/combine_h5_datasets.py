import torch
import h5py
import numpy as np
from tracktor.config import cfg
import os.path as osp


filenames = ['data/correlation_dataset/dataset_more_info/correlation_dataset_1.00_0.50.hdf5',
            'data/correlation_dataset/dataset_more_info/correlation_dataset_1.20_0.50.hdf5',
            'data/correlation_dataset/dataset_more_info/correlation_dataset_1.50_0.50.hdf5',
            'data/correlation_dataset/dataset_more_info/correlation_dataset_2.00_0.50.hdf5']

filename_comb = 'combined_dataset_1.0_1.2_1.5_2.0.hdf5'
h5_file = osp.join(cfg.DATA_DIR, 'correlation_dataset', 'dataset_more_info', filename_comb)

sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05' ,'MOT17-02','MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']

# Get lengths of 
lengths = []
curr_f = h5py.File(filenames[0], mode='r')
print("Lengths x4")
for seq in sequences:
    length = len(filenames) * curr_f[f'/{seq}/fmap_prev'].shape[0]
    print(length)
    lengths.append(length)
curr_f.close()

print(40*'#')
print("Creating dataset")
h5 = h5py.File(h5_file, mode='w')
for seq, num_pairs in zip(sequences, lengths):
    print(seq)
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
    h5_dataset_10 = group.create_dataset("enlargement_factor", (num_pairs,), dtype=np.float32)

    print(40*'#')
    print("Copying dataset")
    pairs_stored = 0
    num_pairs_one_seq = int(num_pairs / len(filenames))

    for f in filenames:
        curr_f = h5py.File(f, mode='r')
        print(f)
        enlargement_factor = np.float32(osp.basename(f).split("_")[2])
        print("Enlargement factor: {}".format(enlargement_factor))
        print("Pairs stored: {}".format(pairs_stored))

        h5_dataset_1[pairs_stored:pairs_stored+num_pairs_one_seq] = curr_f[f"/{seq}/fmap_prev"]
        h5_dataset_2[pairs_stored:pairs_stored+num_pairs_one_seq] = curr_f[f"/{seq}/fmap_enlarged"]
        h5_dataset_3[pairs_stored:pairs_stored+num_pairs_one_seq] = curr_f[f"/{seq}/boxes_next"]
        h5_dataset_4[pairs_stored:pairs_stored+num_pairs_one_seq] = curr_f[f"/{seq}/boxes"]
        h5_dataset_5[pairs_stored:pairs_stored+num_pairs_one_seq] = curr_f[f"/{seq}/boxes_enlarged"]
        h5_dataset_6[pairs_stored:pairs_stored+num_pairs_one_seq] = curr_f[f"/{seq}/names"]
        h5_dataset_7[pairs_stored:pairs_stored+num_pairs_one_seq] = curr_f[f"/{seq}/names_next"]
        h5_dataset_8[pairs_stored:pairs_stored+num_pairs_one_seq] = curr_f[f"/{seq}/preprocessed_image_sizes"]
        h5_dataset_9[pairs_stored:pairs_stored+num_pairs_one_seq] = curr_f[f"/{seq}/original_image_sizes"]

        enlargement_list = [enlargement_factor for i in range(num_pairs_one_seq)]
        h5_dataset_10[pairs_stored:pairs_stored+num_pairs_one_seq] = enlargement_list
        pairs_stored += num_pairs_one_seq
        curr_f.close()
h5.close()