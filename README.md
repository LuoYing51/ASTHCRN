# ASTHCRN
## Modeling dynamic higher-order spatiotemporal relationships: hypergraph-enhanced load forecasting for power grids

> **⚠️ Note: Academic Work Disclosure**:
> This repository contains the implementation code for the academic paper "Modeling dynamic higher-order spatiotemporal relationships: hypergraph-enhanced load forecasting for power grids" which is currently under review. The code is provided as a research prototype for transparency and reproducibility. Please refrain from using the core methodology or results for commercial purposes or in publications until the associated paper is officially published.
## Framework
![整体框架AHGCRN](https://github.com/user-attachments/assets/9fb73043-3b81-45dc-8ac0-676a5cde1680)

## Requirements
### Environments
Python 3.10
### Packages
```
numpy==2.3.4
pandas==2.3.3
scikit_learn==1.1.0
torch==2.6.0+cu126
torch_geometric==2.7.0
```

## Dataset
  We evaluated the model on four hourly power load datasets: two real-world datasets from the State Grid Corporation of China and two public benchmarks (CAISO and GEFCOM2012 dataset). The chosen datasets encompass various characteristics, including different periodicities and spatial distributions of substations or load zones, enabling a robust assessment of adaptability and predictive capability across diverse scenarios.  
* **CAISO**. This dataset provides hourly load data for 34 regional zones in California, recorded from May 1 to December 31, 2014. It also includes an aggregated series for the entire state. (https://energyonline.com/Data/)  
* **GEFCOM2012**. This dataset comprises two subsets: a power load dataset and a wind power dataset. The load subset contains backcast and forecast hourly load values (in kW) for 20 regions served by a U.S. utility, with data recorded hourly from November 29, 2006, to June 30, 2008. (https://doi.org/10.1016/j.ijforecast.2013.07.001)  
* The original raw dataset from the State Grid Corporation of China used for this research cannot be made publicly available due to its confidential nature and proprietary restrictions.  
To ensure the reproducibility of our methodology, a simulated dataset has been generated and is provided at the aforementioned link. This simulated data shares the same format and structure as the original data, and is sufficient to demonstrate the functionality of the provided analysis codes and to allow for the verification of our data processing and analysis workflow.  
## Folder Structure
```
ASTHCRN/
├── .idea/
│   ├── .gitignore
│   ├── ASTHCRN.iml
│   ├── inspectionProfiles/
│   │   ├── Project_Default.xml
│   │   ├── profiles_settings.xml
│   ├── misc.xml
│   ├── modules.xml
│   ├── workspace.xml
├── ASTHCRN_Nanpu/                      # Folder for storing dataset results
│   ├── in_24h_out_12h/                    # Folder for storing prediction task results
│   │   ├── best_model.pth                 # Best model for this task
│   │   ├── in_24h_out_12hmae_per_timestep.csv     # MAE results for this task
│   │   ├── in_24h_out_12hmape_per_timestep.csv    # MAPE results for this task
│   │   ├── in_24h_out_12hmse_per_timestep.csv     # MSE results for this task
│   │   ├── in_24h_out_12hrmse_per_timestep.csv    # RMSE results for this task
├── ASTHCRN_main.py                            # Main program entry code
├── ASTHCRN_main_without_AdaHG.py             #Code for ablation experiments.
├── ASTHCRN_main_without_GRU.py
├── ASTHCRN_main_without_HGCN.py
├── ASTHCRN_main_without_TempHG.py
├── ablation/                                 File folder for ablation experiments.
│   ├── AHGCRU_without_AdaHG.py
│   ├── AHGCRU_without_GRU.py
│   ├── AHGCRU_without_HGCN.py
│   ├── AHGCRU_without_TempHG.py
│   ├── ASTHCRN_model_without_AdaHG.py
│   ├── ASTHCRN_model_without_GRU.py
│   ├── ASTHCRN_model_without_HGCN.py
│   ├── ASTHCRN_model_without_TempHG.py
│   ├── __pycache__/
│   │   ├── AHGCRU_without_AdaHG.cpython-310.pyc
│   │   ├── ASTHCRN_model_without_AdaHG.cpython-310.pyc
│   ├── data_hyperedge_index.csv        # Hypergraph adjacency matrix for virtual data
├── components/                        # Folder for storing model-related components
│   ├── AHGCRU.py
│   ├── ASTHCRN_model.py
│   ├── __pycache__/
│   │   ├── AHGCRU.cpython-310.pyc
│   │   ├── ASTHCRN_model.cpython-310.pyc
├── data/                              # Virtual data folder
│   ├── data.xlsx
├── readme.docx
├── requirements.txt
├── utils_/
│   ├── All_Metrics.py                 # Code for calculating evaluation metrics
│   ├── __pycache__/
│   │   ├── All_Metrics.cpython-310.pyc
│   │   ├── build_hypergraph.cpython-310.pyc
│   │   ├── data_pro.cpython-310.pyc
│   │   ├── get_logger.cpython-310.pyc
│   │   ├── init_seed.cpython-310.pyc
│   │   ├── propera_data.cpython-310.pyc
│   │   ├── save_result.cpython-310.pyc
│   ├── build_hypergraph.py            # Code for hypergraph construction methods
│   ├── data_pro.py                   # Data preprocessing code
│   ├── get_logger.py                  # Logging code
│   ├── init_seed.py                   # Code for initializing random seeds
│   ├── inverse_normalize.py            # Denormalization code
│   ├── propera_data.py
│   ├── save_result.py                  # Code for saving results
```
## Arguments
We introduce some major arguments of our main function here.

**Training settings:**

* train_rate: data split rate of training set
* val_rate: data split rate of validation set
* test_rate: data split**** rate of test set
* T_dim: time length of historical steps
* output_T_dim: time length of future steps
* num_nodes: the number of stations
* batch_size: training or testing batch size
* in_channels: the feature dimension of inputs
* output_dim: the feature dimension of outputs
* learning_rate: the learning rate at the beginning
* epochs: training epochs
* early_stop_patience: the patience of early stopping
* device: using which GPU to train our model
* seed: the random seed for experiments

**Model hyperparameters:**

* embed_size: dimension of feature embedding for convolution expansion 
* d_inner: hidden dimension of the HGCRU
* hyperedge_rate: hyperedge rate
* HGCNADP_embed_dims: embedding dimensions for nodes and hyperedges in HGCNADP module
* dropout: dropout rate
