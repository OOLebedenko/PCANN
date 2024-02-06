from KDNN.dataset import KdDataset, EsmPretrainedModel

#######################################################################################################################
# This examples use the EsmPretrainedModel. Please, download code from github:
# git clone git@github.com:facebookresearch/esm.git
# and fill the esm_path according to your directory tree
#######################################################################################################################

# esm_path = ""
esm_path = "/home/olebedenko/Projects/kd_pred/"

path_to_target_csv = "log_kd.csv"  # basename for csv with target data
path_to_pdb_dir = "pdb"  # name of raw pdb directory
pdb_fname_format = "{}.pdb1"  # format string with pdb_id from path_to_target_csv to get file name
n_process = 1  # number of processes using multiprocessing

model_name = "esm2_t33_650M_UR50D"  # name of esm model
path_to_esm_dir = f"{esm_path}/esm"  # path to downloaded esm repo
esm_model = EsmPretrainedModel(path_to_esm_dir, model_name)

dataset = KdDataset(path_to_pdb_dir=path_to_pdb_dir,
                    path_to_target_csv=path_to_target_csv,
                    pdb_fname_format=pdb_fname_format,
                    pretrained_model=esm_model,
                    n_process=n_process
                    )
