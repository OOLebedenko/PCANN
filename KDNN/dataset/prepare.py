import os
import subprocess
import typing
import tempfile

import gemmi
import numpy as np
import torch

from typing import Iterator, List, Tuple, Union, Dict
from scipy.spatial import cKDTree

AnyPath = Union[str, bytes, os.PathLike]


def convert_3to1(list_of_aa: Union[List, Tuple]):
    dict_3to1 = {
        "ALA": 'A', "ARG": 'R', "ASH": 'D', "ASN": 'N', "ASP": 'D',
        "CYM": 'C', "CYS": 'C', "CYX": 'C', "GLH": 'E', "GLN": 'Q',
        "GLU": 'E', "GLY": 'G', "HID": 'H', "HIE": 'H', "HIP": 'H',
        "HIS": 'H', "HYP": 'O', "ILE": 'I', "LEU": 'L', "LYN": 'K',
        "LYS": 'K', "MET": 'M', "PHE": 'F', "PRO": 'P', "SER": 'S',
        "THR": 'T', "TRP": 'W', "TYR": 'Y', "VAL": 'V'
    }
    return "".join([dict_3to1[aa] for aa in list_of_aa])


class DimerStructure:

    def __init__(self,
                 pdb_file: AnyPath) -> None:
        self.st = gemmi.read_pdb(str(pdb_file), split_chain_on_ter=True)
        self.st.setup_entities()
        self._copy = self.st
        self.pretrained_embeddings = None

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

    def remove_hetatm(self):
        for _, _, residue in self.iterate_over_residues():
            if residue.het_flag == "A":
                residue.flag = "A"
        selection = gemmi.Selection('/1').set_residue_flags('A')
        selection.remove_not_selected(self.st)
        return self

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
        self.st.remove_empty_chains()
        self.remove_hetatm()
        self.remove_unk_residues()
        return self

    def clone(self) -> "DimerStructure":
        return self.clone()

    def to_original(self) -> "DimerStructure":
        return self._copy

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

    def residues(self) -> List[gemmi.Residue]:
        return [residue for _, _, residue in self.iterate_over_residues()]

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
                             pretrained_model: "PretrainedModel"
                             ):
        _embeddings = pretrained_model.predict(self)
        for entity in self.st.entities:
            if entity.entity_type.name == "Polymer":
                if self.pretrained_embeddings is None:
                    self.pretrained_embeddings = _embeddings[entity.name]
                else:
                    self.pretrained_embeddings = np.concatenate([self.pretrained_embeddings, _embeddings[entity.name]],
                                                                axis=0)
        return self.pretrained_embeddings

    def write_pdb(self, path: AnyPath):
        return self.st.write_pdb(str(path), gemmi.PdbWriteOptions(minimal=True, numbered_ter=False))

    @property
    def sequence_by_chains(self):
        sequences = {}
        for chain in self.st[0]:
            three_letter_seq = [residue.name for residue in chain if residue.name]
            sequences[chain.name] = convert_3to1(three_letter_seq)
        return sequences

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
        return list(self.st.model)

    @property
    def name(self):
        return self.st.name.split(".")[0]


class PretrainedModel:
    def __init__(self,
                 path_to_esm_dir: AnyPath,
                 name: str):
        self.path_to_esm_dir = path_to_esm_dir
        self.name = name

    def _build_command_template(self,
                                path_to_esm_dir: AnyPath) -> str:
        raise NotImplementedError

    def predict(self,
                st: "DimerStructure") -> Dict[str, np.array]:
        embeddings = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'tmp.fasta'), "w") as tmp_fasta_file:
                for chain_name, sequence in st.sequence_by_chains.items():
                    tmp_fasta_file.write(f">{chain_name}")
                    tmp_fasta_file.write("\n")
                    tmp_fasta_file.write(sequence)
                    tmp_fasta_file.write("\n")
            command_template = self._build_command_template(self.path_to_esm_dir)
            subprocess.call(command_template.format(fasta_file=os.path.join(tmp_dir, 'tmp.fasta'),
                                                    outdir=tmp_dir).split())
            for chain_name in st.sequence_by_chains.keys():
                embeddings[chain_name] = next(
                    iter(torch.load(os.path.join(tmp_dir, f"{chain_name}.pt"))["representations"].values())).numpy()
        return embeddings


class EsmPretrainedModel(PretrainedModel):

    def _build_command_template(self,
                                path_to_esm_dir: AnyPath) -> str:
        os.putenv("PYTHONPATH", os.pathsep.join([os.getenv("PYTHONPATH", ""), path_to_esm_dir]))
        command_template = f"python3 {os.path.join(path_to_esm_dir, 'scripts', 'extract.py')} " \
                           f"{self.name} {{fasta_file}} {{outdir}} --include per_tok"
        return command_template
