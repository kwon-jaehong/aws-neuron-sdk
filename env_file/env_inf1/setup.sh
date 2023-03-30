#!/bin/bash

## aws 뉴런 sdk 드라이버 설치 밑 환경변수 셋팅
. /etc/os-release
tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF

wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -


apt-get update -y
apt-get install linux-headers-$(uname -r) -y
apt-get install aws-neuronx-dkms -y
apt-get install aws-neuronx-runtime-lib -y
apt-get install aws-neuronx-tools -y

export PATH=/opt/aws/neuron/bin:$PATH

## aws 뉴런 sdk 드라이버 설치 밑 환경변수 셋팅 끝


## python 3.7설치
add-apt-repository -y ppa:deadsnakes/ppa
apt-get install -y python3.7 python3.7-dev python3.7-distutils apt-transport-https g++
alias python=python3.7 
apt install -y protobuf-compiler 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py 
python3.7 get-pip.py 
ln -s /usr/bin/python3.7 /usr/bin/python


