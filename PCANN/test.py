import argparse
import os

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

import PCANN.dataset as module_dataset
import PCANN.dataset.transform as module_transform
import PCANN.utils.metric as module_metric
import PCANN.model.model as module_arch

from tqdm import tqdm
from torch_geometric.loader import DataLoader

from PCANN.utils.setup import SetupRun, SetupLogger
from PCANN.utils.util import read_json
from PCANN.utils.visualization import TensorboardWriter
from PCANN.utils.util import MetricTracker


def run_testing(run_setup: SetupRun,
                logger_setup: SetupLogger,
                vizualizer_setup,
                dataset_type,
                device: str):
    # setup logger
    logger = logger_setup("test")

    # setup dataset
    pretrained_model = run_setup.init_obj("pretrained_model", module_dataset)

    dataset_transforms = T.Compose([run_setup.init_obj(transform, module_transform)
                                    for transform in run_setup['dataset_transforms']])
    dataset = run_setup.init_obj(name=f'dataset_{dataset_type}',
                                 module=module_dataset,
                                 pretrained_model=pretrained_model,
                                 transform=dataset_transforms
                                 )
    # setup data_loader instances
    test_loader = DataLoader(dataset=dataset, batch_size=1)

    # setup model architecture, then print to console
    model = run_setup.init_obj('arch', module_arch)
    logger.info(model)

    # loading the best checkpoint from training
    path_to_checkpoint = os.path.join("experiments", run_setup["outdir"], "checkpoint", "model_best.pth")
    model.load_state_dict(torch.load(path_to_checkpoint, map_location=torch.device('cpu'))['state_dict'])

    # setup function handles of metrics
    criterion = run_setup.init_funct("loss", F)
    metrics = [run_setup.init_funct(metric, module_metric) for metric in run_setup['metrics']]

    # run testing process with saving metrics in logs
    model.eval()
    test_metrics = MetricTracker('loss', 'output', *[m.__name__ for m in metrics], writer=vizualizer_setup)

    message = ["number", "pdb_id", "predicted", "target", "loss"]
    for met in metrics:
        message.append(met.__name__)

    logger.info(" - ".join(message))

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):

            data, target = data.to(device), data.target.to(device)
            output = model(data)
            loss = criterion(output, target)

            vizualizer_setup.set_step(batch_idx, mode="test")
            test_metrics.update('loss', loss.item())
            test_metrics.update('output', output)

            message = [batch_idx + 1, data.label[0], output.item(), target.item(), loss.item()]

            for met in metrics:
                test_metrics.update(met.__name__, met(output, target))
                message.append(met(output, target))

            logger.info(" - ".join(str(x) for x in message))

    return test_metrics.result()


def main(config,
         log_config,
         dataset_type
         ):
    # read configurations, hyperparameters for training and logging
    config = read_json(config)
    log_config = read_json(log_config)

    # set directories where trained model and log will be saved.
    checkpoint_dir = os.path.join("tests", config['outdir'], dataset_type)
    log_dir = os.path.join("tests", config['outdir'], dataset_type)

    run_setup = SetupRun(config=config,
                         checkpoint_dir=checkpoint_dir)

    log_setup = SetupLogger(config=log_config,
                            log_dir=log_dir)

    cfg_trainer = run_setup['trainer']['args']
    vizualizer_setup = TensorboardWriter(log_dir, log_setup, cfg_trainer['tensorboard'])

    # run training process
    run_testing(run_setup, log_setup, vizualizer_setup=vizualizer_setup, device="cpu",
                dataset_type=dataset_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-l', '--log-config', default="logger_config.json", type=str,
                        help='log config file path (default: logger_config.json)')
    parser.add_argument('-d', '--device', default="cpu", type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--dataset_type', default="test", type=str, choices=["test", "valid", "train"])
    args = parser.parse_args()

    main(config=args.config,
         log_config=args.log_config,
         dataset_type="test"
         )
