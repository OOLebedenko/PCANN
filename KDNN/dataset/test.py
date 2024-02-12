import torch
import pandas as pd
import numpy as np
import os
import subprocess
from typing import Union, List, Tuple, Callable, Optional
import tempfile

from KDNN.dataset import KdDataset
from KDNN.dataset.prepare import DimerStructure, PretrainedModel,  EsmPretrainedModel
from scipy.spatial.distance import cdist

AnyPath = Union[str, bytes, os.PathLike]

class TestExtraction(KdDataset):

    def __init__(self,
                 path_to_pdb_dir: AnyPath,
                 path_to_target_csv: AnyPath,
                 pretrained_model: PretrainedModel,
                 pdb_fname_format="{}.pdb1.gz",
                 root: AnyPath = ".",
                 interface_cutoff: float = 5.0,
                 n_process: int = 1,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 ):
        super().__init__(path_to_pdb_dir,
                         path_to_target_csv,
                         pretrained_model,
                         pdb_fname_format,
                         root,
                         interface_cutoff,
                         n_process,
                         transform,
                         pre_transform,
                         pre_filter)

    def process_manual(self, raw_paths):
        print(self.raw_paths)
        for raw_path in self.raw_paths:
            print("path:", raw_path)
            st = DimerStructure(raw_path)
            st.clean()
            embeddings = {}
            chains = []
            with tempfile.TemporaryDirectory() as tmp_dir:
                for chain_name, sequence in st.sequence_by_chains.items():
                    chains.append(chain_name)
                    with open(os.path.join(tmp_dir, 'tmp.fasta'), "w") as tmp_fasta_file:
                        print(chain_name, sequence)
                        tmp_fasta_file.write(f">{chain_name}")
                        tmp_fasta_file.write("\n")
                        tmp_fasta_file.write(sequence)
                        tmp_fasta_file.write("\n")
                        tmp_fasta_file.close()

                        command_template = self.pretrained_model._build_command_template(self.pretrained_model.path_to_esm_dir)
                        subprocess.call(command_template.format(fasta_file=os.path.join(tmp_dir, 'tmp.fasta'),
                                                                outdir=tmp_dir).split())
                        out = torch.load(os.path.join(tmp_dir, f"{chain_name}.pt"))
                        key = list(out["representations"].keys())[0]
                        embeddings[chain_name] = out["representations"][key]

            assert len(chains) == 2, "more than 2 chains"

            contacts = {}
            contacts[f"{st.chains[0].name}"] = []
            contacts[f"{st.chains[1].name}"] = []

            reses = []
            positions_1 = {}
            positions_2 = {}

            for i, res_1 in enumerate(st.chains[0]):
                for j, res_2 in enumerate(st.chains[1]):
                    crds_1 = []
                    crds_2 = []
                    for atom_1 in res_1:
                        if atom_1.name == "CA":
                            CA_1 = atom_1
                        crds_1.append([atom_1.pos.x, atom_1.pos.y, atom_1.pos.z])
                    for atom_2 in res_2:
                        if atom_2.name == "CA":
                            CA_2 = atom_2
                        crds_2.append([atom_2.pos.x, atom_2.pos.y, atom_2.pos.z])

                    distances = cdist(crds_1, crds_2, 'euclidean')
                    if distances[distances < self.cutoff].any():
                        if i not in contacts[f"{st.chains[0].name}"]:
                            contacts[f"{st.chains[0].name}"].append(i)
                            reses.append([res_1])
                            positions_1[i] = [CA_1.pos.x, CA_1.pos.y, CA_1.pos.z]
                        if j not in contacts[f"{st.chains[1].name}"]:
                            contacts[f"{st.chains[1].name}"].append(j)
                            reses.append([res_2])
                            positions_2[j] = [CA_2.pos.x, CA_2.pos.y, CA_2.pos.z]
            contacts[f"{st.chains[0].name}"] = sorted(contacts[f"{st.chains[0].name}"])
            contacts[f"{st.chains[1].name}"] = sorted(contacts[f"{st.chains[1].name}"])
            x = torch.cat((embeddings[f"{st.chains[0].name}"][contacts[f"{st.chains[0].name}"]], embeddings[f"{st.chains[1].name}"][contacts[f"{st.chains[1].name}"]]), dim=0)

            positions_1 = list(dict(sorted(positions_1.items(), key=lambda item: item[0])).values())
            positions_2 = list(dict(sorted(positions_2.items(), key=lambda item: item[0])).values())
            coords = positions_1 + positions_2
            distances = cdist(coords, coords, 'euclidean')

            edges = []
            edge_features = []

            for i in range(len(coords)):
                for j in range(len(coords)):
                    edges.append([i, j])
                    edge_features.append(distances[i, j])

            edges = torch.tensor(edges, dtype=torch.long).t()
            edge_features = torch.tensor(edge_features, dtype=torch.float)

            torch.set_printoptions(threshold=10000, linewidth=2000)
            embs = self._get_node_features(DimerStructure(raw_path).clean().renumber_residues())

            assert torch.all(torch.isclose(x, embs, atol=1e-03)), "embeddings are not similar"

            assert torch.all(torch.isclose(edges, self._get_adjacency_info(DimerStructure(raw_path).clean().renumber_residues()))), "edges are not similar"

            assert torch.all(torch.isclose(edge_features[:, None], self._get_edge_features(DimerStructure(raw_path).clean().renumber_residues()))), "edge features are not similar"



def main():
    esm_path = "/home/physicist/DataStorage/BioNMR/"

    path_to_target_csv = "/home/physicist/DataStorage/BioNMR/KDNN/examples/create_graph_dataset/log_kd.csv"  # basename for csv with target data
    path_to_pdb_dir = "/home/physicist/DataStorage/BioNMR/KDNN/experiments/create_graph_dataset/pdb"  # name of raw pdb directory
    pdb_fname_format = "{}.pdb"  # format string with pdb_id from path_to_target_csv to get file name
    n_process = 1  # number of processes using multiprocessing

    model_name = "esm2_t6_8M_UR50D"  # name of esm model
    path_to_esm_dir = f"{esm_path}/esm"  # path to downloaded esm repo
    esm_model = EsmPretrainedModel(path_to_esm_dir, model_name)

    dataset = TestExtraction(path_to_pdb_dir=path_to_pdb_dir,
                             path_to_target_csv=path_to_target_csv,
                             pdb_fname_format=pdb_fname_format,
                             pretrained_model=esm_model,
                             n_process=n_process
                             )

    dataset.process_manual(path_to_pdb_dir)


if __name__ == '__main__':
    main()
