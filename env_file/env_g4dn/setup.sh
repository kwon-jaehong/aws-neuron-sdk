#!/bin/bash
add-apt-repository ppa:graphics-drivers/ppa
apt update
apt-get install nvidia-driver-418 -y
apt-get install -y python3.7 python3.7-dev wget curl python3.7-distutils git apt-transport-https
alias python=python3.7 
apt install -y protobuf-compiler 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py 
python3.7 get-pip.py 
ln -s /usr/bin/python3.7 /usr/bin/python
rm -rf ./get-pip.py 
apt install -y libsm6 libfontconfig1 libxrender1 libxtst6 libglib2.0-0 libgl1-mesa-glx gcc 
# pip install googledrivedownloader torch torchvision
# reboot now