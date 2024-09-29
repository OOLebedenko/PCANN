from PCANN import test

#######################################################################################################################
# This examples use the EsmPretrainedModel.
# Please, download code from github by using script: {YOUR_PATH}/PCANN/trained_models/ESM-2/download.sh
#######################################################################################################################

config = "config.json"
log_config = "../../PCANN/logger_config.json"

for dataset_type in ["train", "valid", "test"]:
    test.main(config=config,
              log_config=log_config,
              dataset_type=dataset_type
              )
