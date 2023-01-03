sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF

wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

apt-get update -y
apt-get install linux-headers-$(uname -r) -y
apt-get install aws-neuronx-dkms -y
export PATH=/opt/aws/neuron/bin:$PATH
apt-get install -y python3.7 python3.7-dev wget curl python3.7-distutils git apt-transport-https
alias python=python3.7 
apt install -y protobuf-compiler 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py 
python3.7 get-pip.py 
ln -s /usr/bin/python3.7 /usr/bin/python


pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install torch-neuron torchvision