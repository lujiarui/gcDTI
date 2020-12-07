#!/bin/bash

echo "##### Checking the environment ##### "
echo "##### Working dir: `pwd` #####"
echo "##### Pytorch Version: `python -c "import torch; print(torch.__version__)"`"
echo "##### CUDA Version: `python -c "import torch; print(torch.version.cuda)"`"

echo "##### Preprocessing the data #####"
python data_helper.py

echo "###### Train on Davis Dataset ##### "
python train.py 0

echo "###### Train on KIBA Dataset ##### "
python train.py 1

echo "##### Having all tasks done, exit now. #####"
sleep 1
