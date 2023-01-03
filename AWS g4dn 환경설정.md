우분투 18.04 LTS nvidia 드라이버 설치
https://codechacha.com/ko/install-nvidia-driver-ubuntu/

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

apt-cache search는 설치 가능한 드라이버 목록을 출력합니다.

apt-cache search nvidia | grep nvidia-driver-418
nvidia-driver-418 - NVIDIA driver metapackage

sudo apt-get install nvidia-driver-418

aws python pip로 패키지 설치시, --no-cache-dir 옵션 꼭 붙일것
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir