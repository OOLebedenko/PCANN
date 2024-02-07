from KDNN import train

#######################################################################################################################
# This examples use the EsmPretrainedModel. Please, download code from github:
# git clone git@github.com:facebookresearch/esm.git
# and fill the path_to_esm_dir in config.json according to your directory tree
#######################################################################################################################

run_dir = "example_experiment"
config = "config.json"
log_config = "../../KDNN/logger_config.json"

train.main(config=config,
           log_config=log_config,
           )
