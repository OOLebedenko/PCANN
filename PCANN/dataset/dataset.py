import torch
import pandas as pd
import os

from multiprocessing import Process
from typing import Union, List, Tuple, Callable, Optional

from scipy.spatial.distance import cdist
from torch_geometric.data import Data, Dataset

from PCANN.dataset.prepare import DimerStructure, PretrainedModel

AnyPath = Union[str, bytes, os.PathLike]


class KdDataset(Dataset):

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
        """
        :param root: where dataset should be stored. This folder is split into
                     raw_dir (original datset) and processed_dir (processed data (graph))
        :param transform:
        :param pre_transform:
        :param pre_filter:
        :param raw_dirname: override the 'raw' name in base class
                            self.raw_dir = os.path.join(self.root, 'raw')
        :param processed_dirname: override the 'processed' name in base class
                                  self.processed_dir = os.path.join(self.root, 'processed')
        """
        self.raw_dirname = path_to_pdb_dir
        self.raw_file_format = pdb_fname_format
        self.df_kd = pd.read_csv(path_to_target_csv)
        self.pretrained_model = pretrained_model
        self.cutoff = interface_cutoff
        self.n_process = n_process
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return self.raw_dirname

    @property
    def raw_file_names(self) -> List[str]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        return [self.raw_file_format.format(pdb_id) for pdb_id in self.df_kd["pdb_id"].values]

    @property
    def raw_paths(self) -> List[str]:
        return [os.path.join(self.raw_dir, raw_fname)
                for raw_fname in self.raw_file_names]

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, f"interface_cutoff_{self.cutoff}", self.pretrained_model.name)

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{pdb_id}_{phenotype}.pt" for pdb_id, phenotype in self.df_kd[["pdb_id", "phenotype"]].values]

    @property
    def processed_paths(self) -> List[str]:
        return [os.path.join(self.processed_dir, processed_fname)
                for processed_fname in self.processed_file_names]

    def __process(self, raw_paths, mutations, phenotypes) -> None:
        for raw_path, _mutations, phenotype in zip(raw_paths, mutations, phenotypes):
            st = DimerStructure(raw_path)
            st.clean()
            if not pd.isnull(_mutations):
                for mutation in _mutations.split("-"):
                    st.mutate_sequence(mutation)
            st.renumber_residues()

            # get node features
            node_features = self._get_node_features(st)

            # get edge features
            edge_features = self._get_edge_features(st)

            # get adjacency info
            edge_indexes = self._get_adjacency_info(st)

            # create data object
            data = Data(x=node_features,
                        edge_index=edge_indexes,
                        edge_attr=edge_features,
                        )

            torch.save(data, os.path.join(self.processed_dir, f"{st.name}_{phenotype}.pt"))

    def process(self) -> None:
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        tmp_not_processed = [(raw_path, mutation, phenotype) for (raw_path, mutation, phenotype, processed_path)
                             in zip(self.raw_paths, self.df_kd["mutation(s)"], self.df_kd["phenotype"],
                                    self.processed_paths)
                             if not os.path.exists(processed_path)
                             ]
        not_processed_raw_paths, mutations, phenotype = list(map(lambda *elem: list(elem), *tmp_not_processed))

        chunk_size = len(not_processed_raw_paths) // self.n_process
        processes = []
        for ind_1, ind_2 in [(chunk_size * ind, chunk_size * ind + chunk_size) for ind in
                             range(self.n_process)]:
            p = Process(
                target=self.__process, kwargs=(
                    {"raw_paths": not_processed_raw_paths[ind_1:ind_2],
                     "mutations": mutations[ind_1:ind_2],
                     "phenotypes": phenotype[ind_1:ind_2]
                     }
                )
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def get(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Gets the data object at index :obj:`idx`."""
        molfile_pt = self.processed_paths[idx]

        target = self.df_kd["target"][(self.df_kd["pdb_id"] == os.path.basename(molfile_pt).split("_")[0]) &
                                      (self.df_kd["phenotype"] ==
                                       os.path.basename(molfile_pt).split(".")[0].split("_", 1)[-1])].values
        target = torch.from_numpy(target).float()
        data = torch.load(os.path.join(molfile_pt))
        data.label = os.path.basename(molfile_pt).split(".")[0]
        data.target = target.view(-1, 1)

        st = DimerStructure(self.raw_paths[idx])
        st.clean().select_interface(self.cutoff).select_ca_atoms()
        data.chain_break_point = len(st.chains[0])
        return data

    def len(self) -> int:
        return len(self.processed_paths)

    def _get_node_features(self, st) -> torch.Tensor:
        """
        This will return a matrix / 2d array of the shape
        [number of nodes, node features size]
        """
        st_copy = st.copy()
        embeddings = st_copy.pretrained_embedding(self.pretrained_model)
        interface_rids = [residue.seqid.num - 1 for residue in
                          st_copy.select_interface(self.cutoff).select_ca_atoms().residues]
        return torch.tensor(embeddings[interface_rids], dtype=torch.float)

    def _get_edge_features(self, st) -> torch.Tensor:
        """
        This will return a matrix / 2d array of the shape
        [number of edges, edge features size]
        """
        st_copy = st.copy()
        st_copy.select_interface(self.cutoff).select_ca_atoms()
        features = cdist(st_copy.coords, st_copy.coords).reshape(-1, 1)
        return torch.tensor(features, dtype=torch.float)

    def _get_adjacency_info(self, st):
        st_copy = st.copy()
        st_copy.select_interface(self.cutoff).select_ca_atoms()
        edges = [[i, j] for i in range(st_copy.coords.shape[0]) for j in range(st_copy.coords.shape[0])]
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
