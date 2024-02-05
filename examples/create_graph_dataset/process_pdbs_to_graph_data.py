from KDNN.dataset import KdDataset, EsmPretrainedModel

#######################################################################################################################
# This examples use the EsmPretrainedModel. Please, download code from github:
# git clone git@github.com:facebookresearch/esm.git
# and fill the esm_path according to your directory tree
#######################################################################################################################

# esm_path = ""
esm_path = "/home/olebedenko/Projects/kd_pred/"

target_csv_fname = "log_kd.csv"  # basename for csv with target data
pdb_gz_files = ["5z5k.pdb1"]  # list of pdb or pdb.gz basenames
root = "."  # root directory with raw pdbs, processed graphs and target data
raw_dirname = "pdb"  # name of raw pdb directory
path_to_esm_dir = f"{esm_path}/esm"  # path to downloaded esm repo
model_name = "esm2_t33_650M_UR50D"  # name of esm model
n_process = 1  # number of processes using multiprocessing

esm_model = EsmPretrainedModel(path_to_esm_dir, model_name)

dataset = KdDataset(root=root,
                    pdb_fnames=pdb_gz_files,
                    raw_dirname=raw_dirname,
                    target_csv_fname=target_csv_fname,
                    pretrained_model=esm_model,
                    n_process=n_process
                    )
