#!/bin/bash

## aws 뉴런 sdk 드라이버 설치 밑 환경변수 셋팅
. /etc/os-release
tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
