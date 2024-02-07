from KDNN import train

run_dir = "example_experiment"
config = "config.json"
log_config = "../../KDNN/logger_config.json"

train.main(config=config,
           log_config=log_config,
           )
