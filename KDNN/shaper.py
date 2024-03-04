import argparse
import os

import pandas as pd
import numpy as np
import torch

import KDNN.dataset as module_dataset
import KDNN.model.model as module_arch

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from KDNN.utils.setup import SetupRun, SetupLogger
from KDNN.utils.util import read_json
from KDNN.utils.visualization import TensorboardWriter


def explain_graphs(run_setup: SetupRun,
                   logger_setup: SetupLogger,
                   vizualizer_setup,
                   dataset_type,
                   device: str,
                   prob=0.7,
                   mc_samples=100):
    # setup logger
    logger = logger_setup("test")

    # setup dataset
    pretrained_model = run_setup.init_obj("pretrained_model", module_dataset)
    dataset = run_setup.init_obj(name=f'dataset_{dataset_type}',
                                 module=module_dataset,
                                 pretrained_model=pretrained_model)

    # setup data_loader instances
    test_loader = DataLoader(dataset=dataset, batch_size=1)

    # setup model architecture, then print to console
    model = run_setup.init_obj('arch', module_arch)
    logger.info(model)

    # loading the best checkpoint from training
    path_to_checkpoint = os.path.join("experiments", run_setup["outdir"], "checkpoint", "model_best.pth")
    model.load_state_dict(torch.load(path_to_checkpoint, map_location=torch.device(device))['state_dict'])
    model.to(device)

    model.eval()

    message = ["edge index", "i", "with edge", "without edge", "target"]

    logger.info(" - ".join(message))
    df_index2phi = pd.DataFrame()

    with (torch.no_grad()):
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            for edge_ind in range(data.edge_attr.shape[0]):
                tmp = {}
                phi = []

                for i in range(mc_samples):
                    indexes = np.random.binomial(1, prob, data.edge_attr.shape[0])

                    indexes[edge_ind] = 1
                    s_y = torch.from_numpy(indexes).nonzero().flatten()
                    indexes[edge_ind] = 0
                    s_n = torch.from_numpy(indexes).nonzero().flatten()

                    edge_index_y = data.edge_index[:, s_y]
                    edge_index_n = data.edge_index[:, s_n]

                    edge_attr_y = data.edge_attr[s_y]
                    edge_attr_n = data.edge_attr[s_n]

                    data_y = Data(x=data.x,
                                  edge_index=edge_index_y,
                                  edge_attr=edge_attr_y,
                                  )
                    data_n = Data(x=data.x,
                                  edge_index=edge_index_n,
                                  edge_attr=edge_attr_n,
                                  )

                    data_y = data_y.to(device)
                    data_n = data_n.to(device)
                    target = target.to(device)

                    output_y = model(data_y)
                    output_n = model(data_n)

                    vizualizer_setup.set_step(batch_idx, mode="test")

                    message = [batch_idx + 1, edge_ind, i, output_y.item(), output_n.item(), target.item()]

                    logger.info(" - ".join(str(x) for x in message))

                    phi.append((output_y - output_n).to("cpu"))

                phi = np.mean(phi)

                tmp["number"] = batch_idx + 1
                tmp["edge_ind"] = edge_ind
                tmp["length"] = data.edge_attr[edge_ind]
                tmp["phi"] = phi

                df_index2phi = pd.concat([df_index2phi, pd.DataFrame(tmp, index=[0])])
    return df_index2phi


def main(config,
         log_config,
         dataset_type
         ):
    # read configurations, hyperparameters for training and logging
    config = read_json(config)
    log_config = read_json(log_config)

    # set directories where trained model and log will be saved.
    checkpoint_dir = os.path.join("explanations", config['outdir'], dataset_type)
    log_dir = os.path.join("explanations", config['outdir'], dataset_type)

    run_setup = SetupRun(config=config,
                         checkpoint_dir=checkpoint_dir)

    log_setup = SetupLogger(config=log_config,
                            log_dir=log_dir)

    cfg_trainer = run_setup['trainer']['args']
    vizualizer_setup = TensorboardWriter(log_dir, log_setup, cfg_trainer['tensorboard'])

    # run training process
    df_index2phi = explain_graphs(run_setup,
                                  log_setup,
                                  vizualizer_setup=vizualizer_setup,
                                  device=config['device'],
                                  dataset_type=dataset_type)
    return df_index2phi
