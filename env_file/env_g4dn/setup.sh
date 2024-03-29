#!/bin/bash

export TZ=Asia/Seoul
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
apt-get update -y
apt-get install wget curl git sudo gnupg2 software-properties-common -y 




add-apt-repository ppa:graphics-drivers/ppa
apt update
apt-get install nvidia-driver-418 -y


add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt-get install -y python3.7 python3.7-dev python3.7-venv wget curl python3.7-distutils git apt-transport-https
alias python=python3.7 
apt install -y protobuf-compiler 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py 
python3.7 get-pip.py 
ln -s /usr/bin/python3.7 /usr/bin/python
rm -rf ./get-pip.py 
apt install -y libsm6 libfontconfig1 libxrender1 libxtst6 libglib2.0-0 libgl1-mesa-glx gcc 
pip install googledrivedownloader torch torchvision "transformers==4.21.3"
reboot now