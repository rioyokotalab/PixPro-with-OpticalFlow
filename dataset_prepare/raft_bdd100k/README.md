# Dataset prepareration
## Create Optical Flow using RAFT dataset for BDD100k
`${BDD100k-Path}` is place of dataset which BDD100k is located when bdd100k dataset created.  

### requiment

* Python 3.8.6
* PyTorch == 1.8.2
* Torchvision == 0.9.2
* CUDA == 10.2
* NCCL == 2.7.3
* Open MPI == 4.0.4
* Other dependencies are same as this repo dependencies

#### Assumed data structure
data structure before execution the below instructions

```python
${BDD100k-Path}
 |-- bdd100k
 |   |-- videos # 1.5TB
 |   |   |-- train # 1.3TB
 |   |   |-- val # 184GB
 |   |-- images # 3.5TB
 |   |   |-- train # 3.1TB
 |   |   |-- val # 443GB
```

### create dataset
1. Clone repo for RAFT and download models for RAFT

    ```shell script
    cd ~
    git clone https://github.com/rioyokotalab/RAFT.git
    cd RAFT
    pyenv local pixpro-wt-of-cu102-wandb # pyenv virtualenv for this repo
    bash scripts/download_models.sh
    mkdir ${BDD100k-Path}/pretrained_flow
    cp -ra models ${BDD100k-Path}/pretrained_flow
    ```

1. Create optical flow dataset  
    Ex. use 256 gpus using 64 machines which have 4 gpus

    ```shell script
    mpirun -np 256 -npernode 4 python flow_save_scripts.py --path ${BDD100k-Path}/bdd100k/images/train --model models/raft-small.pth --small
    ```

    When this execution is complete, Optical Flow dataset is created in `${BDD100k-Path}/bdd100k/flow`

## Final data structure
data structure after completing the above instructions

```python
${BDD100k-Path}
 |-- bdd100k
 |   |-- videos # 1.5TB
 |   |   |-- train # 1.3TB
 |   |   |-- val # 184GB
 |   |-- images # 3.5TB
 |   |   |-- train # 3.1TB
 |   |   |-- val # 443GB
 |   |-- flow # 5.8TB
 |   |   |-- pth
 |   |   |   |-- train
 |   |   |   |   |-- forward # 2.9TB
 |   |   |   |   |-- backward # 2.9TB
```
