import torch
import pandas as pd
import os

from multiprocessing import Process
from typing import Union, List, Tuple, Callable, Optional

from scipy.spatial.distance import cdist
from torch_geometric.data import Data, Dataset

from KDNN.dataset.prepare import DimerStructure, PretrainedModel

AnyPath = Union[str, bytes, os.PathLike]


class KdDataset(Dataset):

    def __init__(self,
                 root: AnyPath,
                 pdb_fnames: Union[List, Tuple],
                 raw_dirname: AnyPath,
                 pretrained_model: PretrainedModel,
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
        self.raw_dirname = raw_dirname
        self.pdb_fnames = pdb_fnames
        self.pretrained_model = pretrained_model
        self.cutoff = interface_cutoff
        self.n_process = n_process
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.raw_dirname)

    @property
    def raw_file_names(self) -> List[str]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        return [f"{pdb_id}.meta" for pdb_id in self.pdb_ids]

    @property
    def raw_paths(self) -> List[str]:
        return [os.path.join(self.raw_dir, fname)
                for fname in self.pdb_fnames
                if not os.path.exists(
                os.path.join(self.pretrained_model.name, f"{os.path.basename(fname).split('.')[0]}.pt"))
                ]

    @property
    def processed_paths(self) -> List[str]:
        return [os.path.join(self.pretrained_model.name, f"{os.path.basename(fname).split('.')[0]}.pt")
                for fname in self.pdb_fnames
                ]

    def __process(self, raw_paths) -> None:
        for raw_path in raw_paths:
            st = DimerStructure(raw_path)
            st.clean()
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

            torch.save(data, os.path.join(self.pretrained_model.name, f"{st.name}.pt"))

    def process(self) -> None:
        processes = []
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        os.makedirs(self.pretrained_model.name, exist_ok=True)
        chunk_size = len(self.raw_paths) // self.n_process
        for ind_1, ind_2 in [(chunk_size * ind, chunk_size * ind + chunk_size) for ind in
                             range(self.n_process)]:
            p = Process(
                target=self.__process, kwargs=(
                    {"raw_paths": self.raw_paths[ind_1:ind_2],
                     }
                )
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def get(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Gets the data object at index :obj:`idx`."""
        kd = pd.read_csv(os.path.join(self.root, 'log_kd.csv'))
        molfile_pt = self.processed_paths[idx]

        target = kd["target"][kd["pdb_id"] == os.path.basename(molfile_pt).split(".pt")[0]].values
        target = torch.from_numpy(target).float()
        data = torch.load(os.path.join(molfile_pt))

        return data, target

    def len(self) -> int:
        return len(self.processed_paths)

    def _get_node_features(self, st) -> torch.Tensor:
        """
        This will return a matrix / 2d array of the shape
        [number of nodes, node features size]
        """
        interface_rids = [residue.seqid.num for residue in st.select_interface(self.cutoff).residues()]
        return torch.tensor(st.pretrained_embedding(self.pretrained_model)[interface_rids], dtype=torch.float)

    def _get_edge_features(self, st) -> torch.Tensor:
        """
        This will return a matrix / 2d array of the shape
        [number of edges, edge features size]
        """
        st.select_interface(self.cutoff).select_ca_atoms()
        features = cdist(st.coords, st.coords).reshape(-1, 1)
        st.to_original()
        return torch.tensor(features, dtype=torch.float)

    def _get_adjacency_info(self, st):
        st.select_interface(self.cutoff).select_ca_atoms()
        edges = [[i, j] for i in range(st.coords.shape[0]) for j in range(st.coords.shape[0])]
        st.to_original()
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
