import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from PCANN.model import KdModel_PoolEdges
from PCANN.dataset import KdDataset, EsmPretrainedModel, InteractionGraph


def load_models(path_to_checkpoint_dir,
                device="cpu"
                ):
    ## parameters used during training
    node_feature_dim = 1280
    node_embedding_dim = 64
    edge_feature_dim = 1
    num_layers = 3
    heads = 1
    add_self_loops = False
    concat = False
    bias = True
    batchnorm = True
    linear_layer_nodes = False
    linear_layer_edges = False

    ## load 25 PCANN models
    models = []
    for model_id in tqdm(range(1, 26), desc="load_models"):
        path_to_checkpoint = os.path.join(path_to_checkpoint_dir, f"model_{model_id:02d}.pth")
        model = KdModel_PoolEdges(node_feature_dim=node_feature_dim,
                                  node_embedding_dim=node_embedding_dim,
                                  edge_feature_dim=edge_feature_dim,
                                  num_layers=num_layers,
                                  linear_layer_nodes=linear_layer_nodes,
                                  linear_layer_edges=linear_layer_edges,
                                  batchnorm=batchnorm,
                                  heads=heads,
                                  add_self_loops=add_self_loops,
                                  concat=concat,
                                  bias=bias,
                                  )

        model.load_state_dict(torch.load(path_to_checkpoint, map_location=torch.device(device))['state_dict'])
        model.eval()
        models.append(model)

    return models


def load_data(path_to_target_csv="examples/create_graph_dataset/log_kd.csv",  # basename for csv with target data
              path_to_pdb_dir="examples/create_graph_dataset/pdb/",  # name of raw pdb directory
              pdb_fname_format="{}.pdb1",  # format string with pdb_id from path_to_target_csv to get file name
              model_name="esm2_t33_650M_UR50D",  # name of esm model
              path_to_esm_dir="trained_models/ESM-2/esm-main",  # path to downloaded esm repo,
              n_process=1
              ):
    esm_model = EsmPretrainedModel(path_to_esm_dir, model_name)
    dataset = KdDataset(root="input",
                        path_to_pdb_dir=path_to_pdb_dir,
                        path_to_target_csv=path_to_target_csv,
                        pdb_fname_format=pdb_fname_format,
                        pretrained_model=esm_model,
                        n_process=n_process,
                        transform=InteractionGraph()
                        )
    return dataset


def make_prediction(dataset,
                    modelsm,
                    device="cpu"
                    ):
    # define constants
    R = 1.987204258e-03  # kcal·mol−1·K−1
    T = 298  # K 

    # setup data_loader instances
    test_loader = DataLoader(dataset=dataset, batch_size=1)

    # make predictions
    df_output = pd.DataFrame()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):

            data = data.to(device)
            predictions = []
            pdb_id = data.label[0]
            for model in tqdm(models, desc="make_predictions"):
                output = model(data)
                predictions.append(output.item() * R * T)

            predictions = np.array(predictions).round(2)
            pdb_id, predictions, predictions.mean().round(2)
            df_batch_output = pd.DataFrame([pdb_id, predictions.mean().round(2), *predictions],
                                           index=["pdb_id", "avg", *[model_id for model_id in range(1, 26)]]).T
            df_output = pd.concat([df_output, df_batch_output])
    return df_output


def pdb_to_input_csv(path_to_pdb):
    pdb_id = os.path.basename(path_to_pdb).split(".")[0]
    return pd.DataFrame({"pdb_id": pdb_id,
                         "phenotype": "wt",
                         "mutation(s)": None,
                         "target": None
                         }, index=[0])


if __name__ == '__main__':
    ## ENTER PATH TO YOUR PDB
    path_to_dimer_pdb = "pdb/1b27.pdb1"

    ## ENTER PATH TO DOWNLOADED ESM REPOSITORY 
    # This examples use the EsmPretrainedModel.
    # Please, download code from github by using script: {YOUR_PATH}/PCANN/trained_models/ESM-2/download.sh
    path_to_esm_dir = "../../trained_models/ESM-2/esm-main"

    ## ENTER PATH TO PCANN CHECKOUT DIRECTORY
    path_to_checkpoint_dir = "../../trained_models/PCANN/"

    ## You don't need to change the rest part of the script
    ## prepare input.csv for dataloader
    df_input = pdb_to_input_csv(path_to_pdb=path_to_dimer_pdb)
    os.makedirs("input", exist_ok=True)
    df_input.to_csv("input/input.csv", index=False)

    ## load dataset
    dataset = load_data(path_to_target_csv="input/input.csv",  # basename for csv with target data
                        path_to_pdb_dir=os.path.dirname(path_to_dimer_pdb),  # name of raw pdb directory
                        pdb_fname_format="{}" + os.path.splitext(path_to_dimer_pdb)[1],
                        # format string with pdb_id from path_to_target_csv to get file name
                        model_name="esm2_t33_650M_UR50D",  # name of esm model
                        path_to_esm_dir=path_to_esm_dir,  # path to downloaded esm repo,
                        n_process=1
                        )

    ## load models
    models = load_models(path_to_checkpoint_dir=path_to_checkpoint_dir
                         )

    ## make predictions
    df_output = make_prediction(dataset, models)

    ## save output
    os.makedirs("output", exist_ok=True)
    df_output.to_csv("output/output.csv", index=False)

    print("see output directory (avg and 25 models)")
