참조
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron/setup/pytorch-install.html

sudo su
apt install git
git clone https://github.com/kwon-jaehong/aws-neuron-sdk.git

## inf 셋팅
source ./aws-neuron-sdk/env_inf1/setup.sh

뉴런 드라이버 확인
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-troubleshoot.html
 lsmod | grep neuron