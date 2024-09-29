import os
import subprocess
import typing
import tempfile

import gemmi
import numpy as np
import torch

from copy import copy
from typing import Iterator, List, Tuple, Union
from scipy.spatial import cKDTree

AnyPath = Union[str, bytes, os.PathLike]


def convert_3to1(list_of_aa: Union[List, Tuple]):
    dict_3to1 = {
        "ALA": 'A', "ARG": 'R', "ASH": 'D', "ASN": 'N', "ASP": 'D',
        "CYM": 'C', "CYS": 'C', "CYX": 'C', "GLH": 'E', "GLN": 'Q',
        "GLU": 'E', "GLX": 'E', "GLY": 'G', "HID": 'H', "HIE": 'H', "HIP": 'H',
        "HIS": 'H', "HYP": 'O', "ILE": 'I', "LEU": 'L', "LYN": 'K',
        "LYS": 'K', "MET": 'M', "PHE": 'F', "PRO": 'P', "SER": 'S',
        "THR": 'T', "TRP": 'W', "TYR": 'Y', "VAL": 'V'
    }
    return "".join([dict_3to1[aa] for aa in list_of_aa])


def convert_1to3(list_of_aa: Union[List, Tuple]):
    dict_1to3 = {'A': 'ALA', 'R': 'ARG', 'D': 'ASP', 'N': 'ASN', 'C': 'CYS',
                 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'O': 'HYP',
                 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
                 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
                 }
    return "".join([dict_1to3[aa] for aa in list_of_aa])


class DimerStructure:

    def __init__(self,
                 pdb_file: AnyPath) -> None:
        self.st = gemmi.read_pdb(pdb_file, split_chain_on_ter=True)
        self.st.setup_entities()
        self._copy = self.st

    def iterate_over_atoms(self) -> Iterator[Tuple[gemmi.Model, gemmi.Chain, gemmi.Residue, gemmi.Atom]]:
        for model in self.st:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        yield model, chain, residue, atom

    def iterate_over_residues(self) -> Iterator[Tuple[gemmi.Model, gemmi.Chain, gemmi.Residue]]:
        for model in self.st:
            for chain in model:
                for residue in chain:
                    yield model, chain, residue

    def remove_hetatm(self) -> "DimerStructure":
        for _, _, residue in self.iterate_over_residues():
            if residue.het_flag == "A":
                residue.flag = "A"
        selection = gemmi.Selection('/1').set_residue_flags('A')
        selection.remove_not_selected(self.st)
        return self

    def check_gaps(self):
        residue_ids = [residue.seqid.num
                       for chain in self.st[0]
                       for residue in chain
                       ]

        residue_ids = np.array(residue_ids)
        gaps = residue_ids[1:] - residue_ids[:-1] - 1
        return sum(gaps) > 0

    def remove_unk_residues(self) -> "DimerStructure":
        for _, _, residue in self.iterate_over_residues():
            if residue.name == "UNK":
                residue.flag = "U"
        selection = gemmi.Selection('/1').set_residue_flags('U')
        selection.remove_selected(self.st)
        return self

    def clean(self) -> "DimerStructure":
        self.st.remove_alternative_conformations()
        self.st.remove_ligands_and_waters()
        self.remove_hetatm()
        self.remove_unk_residues()
        self.st.remove_empty_chains()
        return self

    def copy(self) -> "DimerStructure":
        return copy(self)

    def select(self, gemmi_selection) -> "DimerStructure":
        selection = gemmi.Selection(gemmi_selection)
        self.st = selection.copy_structure_selection(self.st)
        return self

    def select_ca_atoms(self) -> "DimerStructure":
        self.st = gemmi.Selection('CA[C]').copy_structure_selection(self.st)
        return self

    def select_interface(self,
                         cutoff: float) -> "DimerStructure":
        coords_chain = {}
        for model, chain, residue, atom in self.iterate_over_atoms():
            if not coords_chain.get(chain.name):
                coords_chain[chain.name] = []
            coords_chain[chain.name].append(atom)
        assert len(coords_chain) == 2, f"{self.st.name} is not dimer"

        monomer_1_ats, monomer_2_ats = coords_chain.values()
        tree_1 = cKDTree([at.pos.tolist() for at in monomer_1_ats])
        tree_2 = cKDTree([at.pos.tolist() for at in monomer_2_ats])
        dist = tree_1.sparse_distance_matrix(tree_2, max_distance=cutoff, output_type='coo_matrix')

        monomer_1_ats_interface: typing.List[gemmi.Atom] = [monomer_1_ats[ind].serial for ind in dist.row]
        monomer_2_ats_interface: typing.List[gemmi.Atom] = [monomer_2_ats[ind].serial for ind in dist.col]

        for model, chain, residue, atom in self.iterate_over_atoms():
            if (atom.serial in monomer_1_ats_interface) or (atom.serial in monomer_2_ats_interface):
                residue.flag = "i"

        selection = gemmi.Selection('/1').set_residue_flags('i')
        self.st = selection.copy_structure_selection(self.st)
        return self

    @property
    def residues(self) -> List[gemmi.Residue]:
        return [residue for _, _, residue in self.iterate_over_residues()]

    @property
    def atoms(self):
        return [atom for _, _, _, atom in self.iterate_over_atoms()]

    def renumber_residues(self):
        residue_id = 1
        for model in self.st:
            for chain in model:
                for residue in chain:
                    residue.seqid.num = residue_id
                    residue_id += 1
        return self

    def pretrained_embedding(self,
                             pretrained_model: "PretrainedModel",
                             clip_terminal_tags=True
                             ):
        pretrained_embeddings = []
        indexes_by_chains = self._get_model_idx_in_fasta()
        for chain in self.chains:
            sequence = self.full_sequence_by_chains[chain.name]
            indexes = indexes_by_chains[chain.name]
            if clip_terminal_tags:
                start, end = indexes[0] - 1, indexes[-1]
                sequence = sequence[start:end]
                indexes -= indexes[0]
            chain_embeddings = pretrained_model.predict(sequence)[indexes]
            pretrained_embeddings.append(chain_embeddings)
        return np.concatenate(pretrained_embeddings)

    def write_pdb(self, path: AnyPath):
        return self.st.write_pdb(str(path), gemmi.PdbWriteOptions(minimal=True, numbered_ter=False))

    @property
    def model_sequence_by_chains(self):
        sequences = {}
        for chain in self.st[0]:
            three_letter_seq = [residue.name for residue in chain if residue.name]
            sequences[chain.name] = convert_3to1(three_letter_seq)
        return sequences

    @property
    def full_sequence_by_chains(self):
        sequences = {}
        entites_dict = {entity.name: entity.full_sequence for entity in self.st.entities
                        if entity.polymer_type.name == "PeptideL"}
        for chain in self.chains:
            chain_sequence = gemmi.one_letter_code(entites_dict[chain.name])
            sequences[chain.name] = chain_sequence
        return sequences

    def _get_model_idx_in_fasta(self):
        indexes_by_chains = {}
        for chain in self.chains:
            result = self._align_model_to_full_sequence()[chain.name]
            seq_with_gaps = result.add_gaps(self.model_sequence_by_chains[chain.name], 2)
            indexes = np.array([ind for ind, resname in enumerate(seq_with_gaps, 1)
                                if resname != "-"])
            indexes_by_chains[chain.name] = indexes
        return indexes_by_chains

    def _align_model_to_full_sequence(self):
        alignment = {}

        for entity in self.st.entities:
            if entity.polymer_type.name == "PeptideL":
                alignment[entity.name] = gemmi.align_sequence_to_polymer(entity.full_sequence,
                                                                         self.st[0][entity.name].get_polymer(),
                                                                         gemmi.PolymerType.PeptideL,
                                                                         gemmi.AlignmentScoring())
        return alignment

    def mutate_sequence(self, mutation):

        mutated_chain_name = mutation[0]
        ref = mutation.split(":")[-1][0]
        pos = int(mutation.split(":")[-1][1:-1])
        mut = mutation.split(":")[-1][-1]

        for model, chain, residue in self.iterate_over_residues():
            if chain.name == mutated_chain_name:
                if residue.seqid.num == pos:
                    assert residue.name == convert_1to3([ref])
                    residue.name = convert_1to3([mut])

    @property
    def coords(self):
        crds = []
        for _, chain, residue, atom in self.iterate_over_atoms():
            crds.append([atom.pos.x, atom.pos.y, atom.pos.z])
        return np.array(crds)

    @property
    def entities(self):
        return self.entities

    @property
    def chains(self):
        return list(self.st[0])

    @property
    def name(self):
        return self.st.name.split(".")[0]


class PretrainedModel:
    def __init__(self,
                 path_to_model_dir: AnyPath,
                 name: str,
                 use_gpu: bool = False):
        self.path_to_model_dir = path_to_model_dir
        self.name = name
        self.use_gpu = use_gpu

    def _build_command_template(self) -> str:
        raise NotImplementedError

    def load_data(self,
                  path_to_directory,
                  path_to_file):
        raise NotImplementedError

    def predict(self,
                sequence: "str"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'tmp.fasta'), "w") as tmp_fasta_file:
                tmp_fasta_file.write(f">tmp")
                tmp_fasta_file.write("\n")
                tmp_fasta_file.write(sequence)
                tmp_fasta_file.write("\n")

            command_template = self._build_command_template()
            subprocess.call(command_template.format(fasta_file=os.path.join(tmp_dir, 'tmp.fasta'),
                                                    outdir=tmp_dir).split())
            embeddings = self.load_data(tmp_dir, "tmp.pt")
        return embeddings


class EsmPretrainedModel(PretrainedModel):

    def __init__(self,
                 path_to_model_dir: AnyPath,
                 name: str,
                 use_gpu: bool = False):
        super(EsmPretrainedModel, self).__init__(path_to_model_dir, name, use_gpu)

    def _build_command_template(self) -> str:
        if self.use_gpu:
            device_flag = ""
        else:
            device_flag = "--nogpu"
        os.putenv("PYTHONPATH", os.pathsep.join([os.getenv("PYTHONPATH", ""), self.path_to_model_dir]))
        command_template = f"python3 {os.path.join(self.path_to_model_dir, 'scripts', 'extract.py')} " \
                           f"{self.name} {{fasta_file}} {{outdir}} --include per_tok {device_flag}"
        return command_template

    def load_data(self,
                  path_to_directory,
                  path_to_file
                  ):
        return next(iter(torch.load(os.path.join(path_to_directory, path_to_file))["representations"].values())).numpy()
