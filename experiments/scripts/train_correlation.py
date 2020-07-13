from sacred import Experiment
import os.path as osp
import os
import numpy as np
import yaml
import cv2

import torch
from torch.utils.data import DataLoader
from tracktor.config import cfg

from tracktor.config import get_output_dir, get_tb_dir
from tracktor.correlation.solver import Solver
#from tracktor.datasets.factory import Datasets
from tracktor.datasets.dataloader_correlation import Dataset
from tracktor.correlation.correlation_head import CorrelationHead

ex = Experiment()
ex.add_config('experiments/cfgs/correlation.yaml')

Solver = ex.capture(Solver, prefix='correlation.solver')

@ex.automain
def my_main(_config, correlation):
    # set all seeds
    torch.manual_seed(correlation['seed'])
    torch.cuda.manual_seed(correlation['seed'])
    np.random.seed(correlation['seed'])
    torch.backends.cudnn.deterministic = True

    print(_config)

    output_dir = osp.join(get_output_dir(correlation['module_name']), correlation['name'])
    tb_dir = osp.join(get_tb_dir(correlation['module_name']), correlation['name'])

    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    #########################
    # Initialize dataloader #
    #########################
    print("[*] Initializing Dataloader")

    #db_train = Datasets(correlation['db_train'], correlation['dataloader'])
    h5_file = osp.join(cfg.DATA_DIR, 'correlation_dataset', correlation['db_train'])
    db_train = Dataset(h5_file, ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10','MOT17-11', 'MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'])
    db_train = DataLoader(db_train, batch_size=512, shuffle=True)

    if correlation['db_val']:
        h5_file_val = osp.join(cfg.DATA_DIR, 'correlation_dataset', correlation['db_val'])
        db_val = Dataset(h5_file_val, ['MOT17-13'])    boxes[int(row[0])][int(row[1])] = bb
        # Stick to batchsize = 1, plot images is not vectorized yet
        db_val = DataLoader(db_val, batch_size=1)
        #db_val = Datasets(correlation['db_val'])
    else:
        db_val = None

    ##########################
    # Initialize the modules #
    ##########################
    print("[*] Building Correlation Head")
    network = CorrelationHead()
    if correlation['load_from_rcnn']:
        network.load_from_rcnn(correlation['rcnn_weights'])
    network.train()
    network.cuda()

    ##################
    # Begin training #
    ##################
    print("[*] Solving ...")

    iters_per_epoch = len(db_train)
    max_epochs = 500
    start_decrease = 150
    # we want to keep lr until iter 15000 and from there to iter 25000 a exponential decay
    l = eval(f"lambda epoch: 1 if epoch < {start_decrease} else 0.001**((epoch - {start_decrease})/({max_epochs}-{start_decrease}))")
    solver = Solver(output_dir, tb_dir, lr_scheduler_lambda=l)
    solver.train(network, db_train, db_val, max_epochs, 50, model_args=correlation['model_args'])
