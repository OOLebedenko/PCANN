from KDNN import test

#######################################################################################################################
# This examples use the EsmPretrainedModel. Please, download code from github:
# git clone git@github.com:facebookresearch/esm.git
# and fill the path_to_esm_dir in config.json according to your directory tree
#######################################################################################################################

config = "config.json"
log_config = "../../KDNN/logger_config.json"

for dataset_type in ["train", "test", "valid"]:
    test.main(config=config,
              log_config=log_config,
              dataset_type=dataset_type
              )
