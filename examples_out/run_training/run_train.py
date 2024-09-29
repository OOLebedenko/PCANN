from KDNN import train

#######################################################################################################################
# This examples use the EsmPretrainedModel.
# Please, download code from github by using script: {YOUR_PATH}/KDNN/trained_models/ESM-2/download.sh
#######################################################################################################################

config = "config.json"
log_config = "../../KDNN/logger_config.json"

train.main(config=config,
           log_config=log_config,
           )
