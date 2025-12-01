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
The original raw dataset from the State Grid Corporation of China used for this research cannot be made publicly available due to its confidential nature and proprietary restrictions. To ensure the reproducibility of our methodology, a simulated dataset has been generated and is provided at the aforementioned link. This simulated data shares the same format and structure as the original data, and is sufficient to demonstrate the functionality of the provided analysis codes and to allow for the verification of our data processing and analysis workflow.
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

