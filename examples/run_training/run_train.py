from PCANN import train

#######################################################################################################################
# This examples use the EsmPretrainedModel.
# Please, download code from github by using script: {YOUR_PATH}/PCANN/trained_models/ESM-2/download.sh
#######################################################################################################################

config = "config.json"
log_config = "../../PCANN/logger_config.json"

train.main(config=config,
           log_config=log_config,
           )
