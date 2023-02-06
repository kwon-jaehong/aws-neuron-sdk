#!/bin/bash

apt-get update -y
apt-get install curl wget git gnupg2 software-properties-common unzip -y
## aws 뉴런 sdk 드라이버 설치 밑 환경변수 셋팅 끝

## 도커 설치
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# aws cli 설치
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
sudo ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update

