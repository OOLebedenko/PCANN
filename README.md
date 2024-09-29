<div align="justify">

# PCANN - Protein Complex Affinity Neural Network

This repository contains the scripts, trained models and sample data for protein complex affinity prediction based on its 3D-coordinates.
We trained 25 different models and provided the final result as average over predicitions of these models.

### System requirements

Key packages and programs:

- Linux platform (tested on ubuntu 20.04 and ubuntu 22.04)
- [python3](https://www.python.org/) (tested with python3.9)

### Installation dependencies

Install all dependencies except for PyTorch and PyTorch Geometric.

```code-block:: bash
# create virtual enviroment
python3 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -U setuptools wheel pip

# install python packages
pip install -r requirements.txt

# install PCANN in PYTHONPATH
python setup.py install

# download pretrained esm model
cd trained_models/ESM-2/
bash download.sh
```

Note! Due to the variation in GPU types and drivers that users have access to, we are not able to make one environment that will run on all systems.
As such, we are only providing a ```requirements.txt``` with support for CUDA 11.3 and leaving it to each user to customize it to work on their setups. This customization will involve changing the cudatoolkit, PyTorch version and python version specified in the ```requirements.txt``` file.


### Make predictions. Example

We provide the templates to make predictions in [examples/make_predictions/](examples/make_predictions/)
Please, change ```path_to_dimer_pdb``` parameter in [examples/make_predictions/load.py](examples/make_predictions/load.py) for the complex structure of your interest

```code-block:: bash
   # run the script to extract Dtr
   cd examples/make_predictions/
   python load.py
```
See the expected result in [examples_output/make_predictions/](examples_output/make_predictions/)
Predictions with 25 PCANN model and the averged result are provided in [examples_out/make_predictions/output/output.csv](examples_out/make_predictions/output/output.csv).
We recommended to use PCANN_avg

### Run training and testing. Example

We provide the template to run and test the PCANN model. The purpose is to demonstrate the framework. 

```code-block:: bash
   # run the script to extract Dtr
   cd examples_out/run_training
   python run_train.py
   python run_training/run_test.py
```

</div>



