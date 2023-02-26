AWS 뉴런 sdk 실험

ec2 instance vscode 접속 https://gre-eny.tistory.com/344

꼭 파이썬 3.7 쓸것
----------------------------------------------------------
각 환경 설정파일 실행법은
source env_file/env_inf1/setup.sh
----------------------------------------------------------
그리고 cpu환경, g4dn.xlarge 환경, inf1 환경에서 실험 진행


AWS inf 인스턴스를 사용하기 위해서는 다음과 같은 과정을 따른다.
1. 파이토치, 텐서플로우 모델을 AWS neuron sdk 사용하여 neuron compile을 한다.
2. 컴파일된 모델을 쓴다.
컴파일은 순수 CPU만 쓴다, neuron-top로 확인해봄
-> 꼭 inf1 인스턴스에서 진행하지 않아도 된다, 걍 일반컴에서 컴파일 가능함


----------------------------------------------------------
inf1 인스턴스를 사용하기위한 조건
https://tech.scatterlab.co.kr/aws-inferentia/

Inferentia 하드웨어를 이용해 모델을 추론하기 위해서는 그래프를 컴파일하는 과정이 필요합니다. 컴파일 과정에서는 모델의 추론 과정을 Tracing 하고 고정된 추론 그래프 형식으로 만들게 됩니다. 고정된 추론 그래프의 특성상 Python 로직이 복잡하게 포함되어 있는 모델 코드나 입력에 따라서 추론 Flow 가 동적으로 달라지는 모델은 Inferentia에서 추론할 수 없습니다. 또한 Batch 크기를 제외하고, 입출력 시 Tensor의 shape이 dynamic 하게 달라질 수도 없습니다. 위와 같은 제약조건에 부합하지 않는다면 Inferentia에서 추론하는 것은 적합하지 않습니다. 입출력의 크기가 달라지는 경우, 입/출력에 Padding을 통해서 항상 고정된 크기의 입력 Tensor 를 보장해 주는 것도 한 가지 방법입니다.
-> "모델 forward에 텐서에 따라 if else문 들어간것은 컴파일이 잘 안됨"
-----------------------------------------------------
파이프라인 코어 최적화
https://mkai.org/achieve-12x-higher-throughput-and-lowest-latency-for-pytorch-natural-language-processing-applications-out-of-the-box-on-aws-inferentia/

neuronCore_pipeline_cores = 4*round(모델 내 가중치 수/(2E7))
20000000
2천만
----------------------------------------------
실시간 (1장)처리 레이턴시를 최소화 할것이냐,
배치단위 최적화를 할것이냐

https://pytorch.org/blog/amazon-ads-case-study/



-------------------------

# simple cnn 기준

모델 파라미터 3165826
3백만

cpu g4dn cpu기준
0.0007479619979858398  
1336.9663200708917

AWS inf1.xlarge (latency/throughput)
0.00028248310089111327 
3540.034773214497 1초/처리

GPU 
0.014086396694183349    
70.99047554247332

-> gpu기반은 배치성 처리에 어울리지, 1장 처리는 최소값이 있는듯 하다

-------------------------------------------------------------------

# 레즈넷 50

모델 파라미터 25557032
2천5백만개

CPU g4dn cpu 기준 (latency/throughput)
0.06929637432098389     
14.43076942767533



AWS inf1.xlarge (latency/throughput)
0.0033380889892578123  
299.57260073594955 

데이터 페러렐
0.001828353
546.940333732

코드파이프라인(스샷 참조)
0.0039307351350784305
254.40533784020704

AWS inf1.6xlarge (latency/throughput) 뉴런칩 4개 / 뉴런코어 16개




GPU g4dn.xlarge (latency/throughput)
0.02209744930267334   
45.25409183217452 처리


-> AWS 뉴런 코어가 GPU 대비 8.666배 정도 빠름
ssssxwsc
-------------------------------------------------------------------------------

버트
파라미터 갯수 108311810
약 1억개


CPU g4dn cpu 기준
1시퀀스 
0.13735679149627686     
7.280309834749639


AWS inf1.xlarge (latency/throughput)
0.02777846097946167     
35.999114592394506 처리
데이터 페러렐 쓸시,
0.008607971668243408    
116.17138607567766 처리



AWS inf1.6xlarge (latency/throughput) 뉴런칩 4개 / 뉴런코어 16개
0.06933747053146362     14.422216333176227
데이터 페러렐 쓸시,
0.00887235403060913     112.70965930237402

GPU
1시퀀스 0.01782888889312744 
56.08874484519746 처리
-------------------
크래프트
cpu
1.2017473220825194 
0.832121679511705 처리

inf 2x 라지로 해야됨
-> 뉴런 모델 컴버팅시, x라지 4cpu로는 컴퓨터 뻑남 -> 이유는?
0.0949544906616211
10.531360792230357

----------------------------------
다중 모델 실험

inf1 칩은 다음과 같은 아키텍쳐를 가진다
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inf1-arch.html?highlight=Device%20Memory

Device memory 를 봐야하며
resnet-50 컴파일한 파일, (resnet50_neuron.pt) 42.3메가가
디바이스 메모리에 올라오면 58.7메가가 된다.
이론상 inf1라지에(디바이스 메모리 8기가 제공) 136 개쯤 올릴 수 있다. (로드만... )

226M 컴파일된 버트 모델 크기 -> 적재시 197.3메가

실무에 적용하려면 실제로 로드된 메모리를 확인하며 서비스 해야 될듯 하다.

-----------------------------------

torch.neuron.DataParallel은 큰 기능은 두가지로 나뉜다
1. 알아서, 모든 뉴럴코어 사용 (걍 코어당)
2. 동적 배치처리


레즈넷-50 페러렐 모델 1장씩 입력하면, 데이터 페러렐을 쓴다고, 모든 뉴럴 코어를 사용하지 않는다.... 속음
neuron-top 찍어봄.
혹시나 빨리처리해서 1코어만 괴롭히는가 싶어, 버트에도 실험해 보았다.
뉴런코어 스케쥴링 기능은 없는 듯 하다.
-> 버트모델 데이터 페러렐 + 배치사이즈 1로 하면, 1코어만 쳐먹고 분배하지 않는다.

https://aws.amazon.com/ko/blogs/machine-learning/achieve-12x-higher-throughput-and-lowest-latency-for-pytorch-natural-language-processing-applications-out-of-the-box-on-aws-inferentia/
정확히 이그림이다.


예시 소스는 다음과 같다
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron/api-torch-neuron-dataparallel-api.html

------------------------------
코드 파이프라인
덩치가 큰 모델이라면 파이프라인으로 나눠라 (모델을  여러코어에 나누어 계산)
model_neuron = torch.neuron.trace(model, example_inputs=[image],compiler_args = ['--neuroncore-pipeline-cores', str(num_cores)])
코어 나누는 공식은 neuronCore_pipeline_cores = 4*round(number-of-weights-in-model/(2E7))
대충 이렇게 정의를 해놓았다

---------------------------------------------------

자원 모니터링
neuron-top 

neuron-monitor | /opt/aws/neuron/bin/neuron-monitor-prometheus.py --port 9000
neuron-monitor | python3.7 /opt/aws/neuron/bin/neuron-monitor-prometheus.py

neuron-monitor -c ./monitor.conf |python3.7 /opt/aws/neuron/bin/neuron-monitor-prometheus.py --port 9000
neuron-monitor -c ./monitor.conf | /opt/aws/neuron/bin/neuron-monitor-prometheus.py --port 9000
curl http://localhost:9000/


---------------------------------------------------------------
프로메테우스 커스텀 매트릭스
https://towardsdatascience.com/kubernetes-hpa-with-custom-metrics-from-prometheus-9ffc201991e


------------------------
뉴런 트레이스를 한 정적 그래프 모델은 가변적인 입력크기에 대응 할 수 없다.
그래서 공식 문서에는 버켓팅이라는 기술을 사용하거나 패딩을 추천한다.
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/appnotes/torch-neuron/bucketing-app-note.html#bucketing-app-note

--------------------
함수도 컴파일 가능하다!!!!!
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron/api-compilation-python-api.html?highlight=trace
