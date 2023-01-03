AWS 뉴런 sdk 실험


--------------------------------------------------------
먼저, 아무 환경에서 mnist_train.py를 실행하여 간단한 cnn 모델을 학습 시킨다. ("mnist_base.pth")

그리고 cpu환경, g4dn.xlarge 환경, inf1 환경에서 mnist 테스트셋 (1채널 28*28 이미지 10000장, 배치 512)


AWS inf 인스턴스를 사용하기 위해서는 다음과 같은 과정을 따른다.
1. 파이토치, 텐서플로우 모델을 AWS neuron sdk 사용하여 neuron compile을 한다.
2. 컴파일된 모델을 쓴다.

----------------------------------------------------------
inf1 인스턴스를 사용하기위한 조건
https://tech.scatterlab.co.kr/aws-inferentia/

Inferentia 하드웨어를 이용해 모델을 추론하기 위해서는 그래프를 컴파일하는 과정이 필요합니다. 컴파일 과정에서는 모델의 추론 과정을 Tracing 하고 고정된 추론 그래프 형식으로 만들게 됩니다. 고정된 추론 그래프의 특성상 Python 로직이 복잡하게 포함되어 있는 모델 코드나 입력에 따라서 추론 Flow 가 동적으로 달라지는 모델은 Inferentia에서 추론할 수 없습니다. 또한 Batch 크기를 제외하고, 입출력 시 Tensor의 shape이 dynamic 하게 달라질 수도 없습니다. 위와 같은 제약조건에 부합하지 않는다면 Inferentia에서 추론하는 것은 적합하지 않습니다. 입출력의 크기가 달라지는 경우, 입/출력에 Padding을 통해서 항상 고정된 크기의 입력 Tensor 를 보장해 주는 것도 한 가지 방법입니다.
-> "모델 forward에 텐서에 따라 if else문 들어간것은 컴파일이 잘 안됨"
-----------------------------------------------------

cpu 
batch_size : 16, 2.82895 sec
batch_size : 32, 2.26020 sec
batch_size : 64, 2.19599 sec
batch_size : 128, 2.20125 sec
batch_size : 256, 2.23990 sec
batch_size : 512, 2.23515 sec
batch_size : 1024, 2.30371 sec

GPU - Tesla T4 (g4dn.x라지)
batch_size : 16, 2.92064 sec
batch_size : 32, 1.07155 sec
batch_size : 64, 0.92514 sec
batch_size : 128, 0.86983 sec
batch_size : 256, 0.84628 sec
batch_size : 512, 0.81790 sec
batch_size : 1024, 0.80388 sec
batch_size : 2048, 0.83201 sec
batch_size : 4096, 0.79397 sec

inf1.xlarge
batch_size : 16, 2.72652 sec
batch_size : 32, 2.35686 sec
batch_size : 64, 2.19344 sec
batch_size : 128, 2.04165 sec
batch_size : 256, 2.00741 sec
batch_size : 512, 2.08360 sec
batch_size : 1024, 2.69540 sec
batch_size : 2048, 3.03630 sec
batch_size : 4096, 3.14335 sec

