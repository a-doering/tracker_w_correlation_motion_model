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
from tracktor.correlation.correlation_head import CorrelationHead
from dataloader_correlation import Dataset

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
    h5_file = osp.join(cfg.DATA_DIR, 'correlation_dataset','correlation_dataset_1.50_0.50.hdf5')
    db_train = Dataset(h5_file)
    db_train = DataLoader(db_train, batch_size=64, shuffle=True)

    if correlation['db_val']:
        db_val = None
        #db_val = DataLoader(db_val, batch_size=1, shuffle=True)
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

    #TODO change scheduling of training and adapt to our patch based correlation approach
    lr = correlation['solver']['optim_args']['lr']
    iters_per_epoch = len(db_train)
    # we want to keep lr until iter 15000 and from there to iter 25000 a exponential decay
    l = eval("lambda epoch: 1 if epoch*{} < 15000 else 0.001**((epoch*{} - 15000)/(25000-15000))".format(
                                                                iters_per_epoch,  iters_per_epoch))
    #else:
    #   l = None
    max_epochs = 100
    solver = Solver(output_dir, tb_dir, lr_scheduler_lambda=l)
    solver.train(network, db_train, db_val, max_epochs, 100, model_args=correlation['model_args'])
