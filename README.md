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

# simple cnn 기준

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


-------------------------------------------------------------------

레즈넷 50

CPU
batch_size : 16, 0.72488 sec
batch_size : 32, 1.31074 sec
batch_size : 64, 2.63903 sec

AWS inf1.xlarge
batch_size : 16, 1.21079 sec
batch_size : 32, 2.73513 sec
Killed









-------------------
root@ip-172-31-14-182:/home/ubuntu/aws-neuron-sdk# python ./resnet-50_mnist/inference_inf.py 
INFO:Neuron:The following operations are currently supported in torch-neuron for this model:
INFO:Neuron:aten::flatten
INFO:Neuron:aten::relu_
INFO:Neuron:aten::adaptive_avg_pool2d
INFO:Neuron:aten::max_pool2d
INFO:Neuron:aten::batch_norm
INFO:Neuron:aten::t
INFO:Neuron:aten::add_
INFO:Neuron:aten::addmm
INFO:Neuron:prim::ListConstruct
INFO:Neuron:prim::Constant
INFO:Neuron:aten::_convolution
INFO:Neuron:100.00% of all operations (including primitives) (1698 of 1698) are supported
INFO:Neuron:100.00% of arithmetic operations (176 of 176) are supported
INFO:Neuron:All operators are compiled by neuron-cc (this does not guarantee that neuron-cc will successfully compile)
INFO:Neuron:Number of arithmetic operators (pre-compilation) before = 176, fused = 176, percent fused = 100.0%
/usr/local/lib/python3.7/dist-packages/torch/tensor.py:590: RuntimeWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
  'incorrect results).', category=RuntimeWarning)
INFO:Neuron:Compiling function _NeuronGraph$1106 with neuron-cc
INFO:Neuron:Compiling with command line: '/usr/local/bin/neuron-cc compile /tmp/tmpsflg89o8/graph_def.pb --framework TENSORFLOW --pipeline compile SaveTemps --output /tmp/tmpsflg89o8/graph_def.neff --io-config {"inputs": {"0:0": [[1, 3, 224, 224], "float32"]}, "outputs": ["Linear_22/aten_addmm/Add:0"]} --verbose 35'
...
Compiler status PASS
INFO:Neuron:Number of arithmetic operators (post-compilation) before = 176, compiled = 176, percent compiled = 100.0%
INFO:Neuron:The neuron partitioner created 1 sub-graphs
INFO:Neuron:Neuron successfully compiled 1 sub-graphs, Total fused subgraphs = 1, Percent of model sub-graphs successfully compiled = 100.0%
INFO:Neuron:Compiled these operators (and operator counts) to Neuron:
INFO:Neuron: => aten::_convolution: 53
INFO:Neuron: => aten::adaptive_avg_pool2d: 1
INFO:Neuron: => aten::add_: 16
INFO:Neuron: => aten::addmm: 1
INFO:Neuron: => aten::batch_norm: 53
INFO:Neuron: => aten::flatten: 1
INFO:Neuron: => aten::max_pool2d: 1
INFO:Neuron: => aten::relu_: 49
INFO:Neuron: => aten::t: 1
CPU top-5 labels: ['tiger', 'lynx', 'tiger_cat', 'Egyptian_cat', 'tabby']
Neuron top-5 labels: ['tiger', 'lynx', 'tiger_cat', 'Egyptian_cat', 'tabby']
Input image shape is [16, 3, 224, 224]
Avg. Throughput: 722, Max Throughput: 783
Latency P50: 21
Latency P90: 25
Latency P95: 26
Latency P99: 26

INFO:Neuron:All operators are compiled by neuron-cc (this does not guarantee that neuron-cc will successfully compile)
INFO:Neuron:Number of arithmetic operators (pre-compilation) before = 176, fused = 176, percent fused = 100.0%
/usr/local/lib/python3.7/dist-packages/torch/tensor.py:590: RuntimeWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
  'incorrect results).', category=RuntimeWarning)
INFO:Neuron:Compiling function _NeuronGraph$1677 with neuron-cc
INFO:Neuron:Compiling with command line: '/usr/local/bin/neuron-cc compile /tmp/tmp5aspfckk/graph_def.pb --framework TENSORFLOW --pipeline compile SaveTemps --output /tmp/tmp5aspfckk/graph_def.neff --io-config {"inputs": {"0:0": [[5, 3, 224, 224], "float32"]}, "outputs": ["Linear_22/aten_addmm/Add:0"]} --verbose 35'
....
Compiler status PASS
INFO:Neuron:Number of arithmetic operators (post-compilation) before = 176, compiled = 176, percent compiled = 100.0%
INFO:Neuron:The neuron partitioner created 1 sub-graphs
INFO:Neuron:Neuron successfully compiled 1 sub-graphs, Total fused subgraphs = 1, Percent of model sub-graphs successfully compiled = 100.0%
INFO:Neuron:Compiled these operators (and operator counts) to Neuron:
INFO:Neuron: => aten::_convolution: 53
INFO:Neuron: => aten::adaptive_avg_pool2d: 1
INFO:Neuron: => aten::add_: 16
INFO:Neuron: => aten::addmm: 1
INFO:Neuron: => aten::batch_norm: 53
INFO:Neuron: => aten::flatten: 1
INFO:Neuron: => aten::max_pool2d: 1
INFO:Neuron: => aten::relu_: 49
INFO:Neuron: => aten::t: 1
Input image shape is [80, 3, 224, 224]
Avg. Throughput: 1199, Max Throughput: 1260
Latency P50: 66
Latency P90: 70
Latency P95: 71
Latency P99: 73





