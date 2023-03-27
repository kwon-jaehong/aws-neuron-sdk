# AWS neuron sdk tutorial

----------------
## 1. AWS Neuron SDK에 대한 간단한 개요


AWS ec2 인스턴스 패밀리 'inf(inferntia)'에 관한 설명 및 실습 튜토리얼을 위해 작성하였다. 기본적으로 **inf 인스턴스**는 딥러닝 모델의 추론를 위한 장비라고 생각 하면 된다 (목적은 거의 GPU와 유사하다 - 특정 계산 특화)  
딥러닝 모델 추론을 **반드시 GPU에서 할 필요가 없다**라고 나는 생각한다. 아래 그림은 inf 인스턴스 패밀리의 하드웨어 사양 및 구조이다.

<p align="center">
  <img src="ETC/image/inf_ec2_family.png">
</p>
<p align="center"><a href="https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inf1-arch.html#aws-inf1-arch"> [ AWS inf 인스턴스 패밀리 ] </a> </p>

<br><br>
예를 들어 **inf1.xlarge**는 CPU 4개, RAM 8기가, 뉴런 칩(neuron chip) 1개를 인스턴스로 구성한다고 보면 된다.
여기서 1개의 뉴런칩은 아래 그림과 같은 구성과 같다
<p align="center">
  <img src="ETC/image/neuron-chip.png">
</p>
<p align="center"><a href="https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inferentia.html#inferentia-arch"> [ AWS neuron chip 구조 ] </a> </p>

위 그림과 같이 구성되어 있고, 하나의 뉴런칩에는 뉴런코어(neuron-core) 4개가 들어 있다.
뉴런 코어의 구성을 보면 SRAM, 텐서 엔진, 벡터 엔진, 스칼라 엔진이 있다.  


<br>

뉴런코어의 구성요소를 간략하게 설명하면 다음과 같다.
 - SRAM : 캐시 메모리와 같이 **빠른 데이터 접근이 필요한 작업**에 사용.
 - ScalarEngine : 출력의 모든 요소가 입력의 한 요소(**예: GELU, SIGMOID 또는 EXP와 같은 비선형성**)에 의존하는 스칼라 계산에 최적화되어 있습니다.
 - VectorEngine :  출력의 모든 요소가 여러 입력 요소에 종속되는 벡터 계산에 최적화되어 있습니다. **예로는 'axpy' 작업(Z=aX+Y), 레이어 정규화, 풀링 작업 등이 있습니다**.
 - TensorEngine : 텐서 계산(**예: GEMM, CONV, Reshape, Transpose**)에 고도로 최적화된 전력 최적화 시스톨릭 어레이를 기반으로 하며 혼합 정밀도 계산(FP16/BF16/INT8 입력, FP32/INT32 출력)을 지원합니다. 각 NeuronCore-v1 TensorEngine은 16TFLOPS의 FP16/BF16 텐서 계산을 제공합니다.
- 하나의 뉴런칩에는 2 GB의 램을 보유한다 (GPU에서 vram이라고 생각하면 된다)  
 

스펙상 뉴런칩 장치 메모리가 8GB라고 명시 되어 있지만, 잘못된?(오해 할만한) 표현인것 같다. 이유는 아래와 같다.
- 2GB 램 * 4개 뉴런코어 = 장치 메모리 **총합 8GB**  


만약, 내 딥러닝 모델의 저장 파일의 크기가 4GB라면? 아무런 처리없이는 뉴런코어에 적재가 불가능 하다. ( 왜냐하면 뉴런코어 하나는 2GB램이 한계다 )
**즉, 큰 딥러닝 모델은 아무런 처리를 하지 않고, 하나의 뉴런코어에 적재가 불가능 하다** (물론 큰 모델을 쪼개거나 뉴런 컴파일 설정하여 여러 뉴런 코어에 분산 그래프 계산을 적용, 큰 모델도 적재가 가능하다) AWS 뉴런 코어를 구성, 컴파일, 테스트를 하려면 많은 공부가 필요하다.  

<br><br>

하지만 내가 inf 인스턴스 (AWS neuron sdk)를 이용해서 딥러닝 서비스 구성하려는 이유는 단 한가지 이다.
- **저렴한 가격**  

<br><br><br>
이 문서를 작성한 현재 기준 인스턴스 가격은 다음과 같다

<p align="center">
  <img src="ETC/image/%EC%84%9C%EC%9A%B8g4dn%EA%B0%80%EA%B2%A9.png" width="70%" height="70%">
  <img src="ETC/image/%EC%84%9C%EC%9A%B8inf1%EA%B0%80%EA%B2%A9.png" width="70%" height="70%">
</p>
<p align="center"> [ AWS 서울리젼 인스턴스 가격표, 2023.03.28 - 기준] </p>

|인스턴스 타입|시간당 요금|한달 유지 요금|
|------|---|---|
|g4dn.xlarge|839.01원|604087.2원|
|inf1.xlarge|364.39원|262360.8원|
(현재)환율 1달러 = 1296.77원  

한달동안 딥러닝 모델을 서비스 한다면, GPU기반으로 서비스 했을때는 최소 **60만원**, 뉴런 코어로 했을때는 최소 **26만원**이 소요된다.
그런데 뉴런 칩에는 뉴런코어 4개가 탑재 되는데, inf.xlarge 인스턴스를 사용하면 GPU 4개를 사용하는 효과라고 나는 생각한다. 뉴런 코어 4개에 각기 다른 모델을 적재하여 서비스를 할 수 있다. 저렴해진 가격 덕분에, 구성할 수 있는 시스템에 대해 많은 선택지가 열리게 된다. 

<br>

물론 하나의 GPU(뉴런코어)에도 VRAM(SRAM)이 허락하는한, 여러 딥러닝 모델을 적재할 수 있다. 하나의 GPU에 A,B,C,D모델이 적제 되어 있다고 가정해보자. 물론 구성&환경에 따라 다르겠지만 이런 환경에서 나는 2가지 문제가 있다고 본다.

1. 서비스(모델 또는 소스코드) 관리가 어렵다
2. 서비스에서 특정 모델 과부하시 다른 모델에도 영향을 준다. 또한 특정 모델을 스케일링 할 수 없다 (통째로 스케일링 된다) 

<br>
1번은 그냥 통상적으로 생각하는 **통합관리 어려움**이라고 생각 하면 된다.
2번에 문제에 대한 **예시는 아래과 같다.**  
특정 시간 A라는 모델이 과부하가 걸렸다고 가정해보자. A모델 서비스 때문에 다른 B,C,D 모델 서비스도 불안정하게 된다(추론 속도, 에러 등). 또한 이러한 환경구성을 스케일링 하려면 번거롭게 소스코드를 분리하거나, 통째로 복사하여 장비를 추가해야된다. 나는 이러한 문제점들을 해결하기 위해 inf1 인스턴스를 활용하여 MSA(MicroService Architecture)를 구성하였다.  

물론 뉴런 코어 하나가 g4dn의 T4(GPU)만큼 비록 성능을 보여주지 않지만, 클라우드로 서비스를 운영할 때 **유지비**를 생각해 본다면 합리적인 선택이 아닐까 생각한다. 나는 **클라우드를 이용하여 딥러닝 서비스 하는 회사**라면 '반드시 이 기술을 써야한다' 라고 생각한다.


참조
- https://aws.amazon.com/ko/machine-learning/neuron/
- https://awsdocs-neuron.readthedocs-hosted.com/en/latest/


-------------------------
## 2. AWS Neuron SDK 설치 및 설정

### 준비 사항 : AWS 계정


- AWS 뉴런 SDK를 사용하기 위해 환경 설정부터 진행하도록 하겠다. 먼저 다음과 같이 콘솔 검색에 'ec2'를 치고 ec2서비스에 접속한다. **(오른쪽 상단에 AWS지역은 '오하이오'로 하자. 다른 지역보다 싸다)**

![Alt text](ETC/image/ec2%EC%84%A4%EC%A0%951.png)

<br><br><br>  

### 1. ssh 키 생성  
인스턴스를 생성하고 ssh 접속하기 위해서는 리전별 키를 생성 해줘야 한다. 절차는 다음과 같다
좌측 하단의 메뉴에서 '네트워크 및 보안' > '키 페어' 메뉴를 누르고, 우측 상단의 주황색으로 표시된 **키 페어 생성**을 클릭한다  


![Alt text](ETC/image/ssh%ED%82%A4%EC%83%9D%EC%84%B11.png)
<br><br><br>

다음과 같이 생성 창이 뜨고, 키페어 이름만 적고 **키페어 생성**을 누른다.   
![Alt text](ETC/image/ssh%ED%82%A4%EC%83%9D%EC%84%B12.png)
<br><br><br>  

생성을 누르면 키페어이름.pem 이라는 파일을 로컬 컴퓨터에 다운 받는다.  
![Alt text](ETC/image/ssh%ED%82%A4%EC%83%9D%EC%84%B13.png)
<br><br><br>  

키페어 생성이 완료되고, 키가 등록된 모습은 다음과 같다
![Alt text](ETC/image/ssh%ED%82%A4%EC%83%9D%EC%84%B14.png)

### 2. 인스턴스 생성  

ec2메뉴에서 '인스턴스' > 인스턴스에서 우측 상단의 주황색으로 표시된 **인스턴스 시작**을 클릭한다   

![Alt text](ETC/image/ec2%EC%83%9D%EC%84%B11.png)  

<br><br><br>  

인스턴스 이름에 아무거나 적는다  
![Alt text](ETC/image/ec2%EC%83%9D%EC%84%B12.png)
<br><br><br>  

인스턴스 이미지(OS)는 Ubuntu 20.04 LST를 선택한다 ( **이 문서에서는 우분투 20.04 환경으로 셋팅을 기본으로 함** )  
![Alt text](ETC/image/ec2%EC%83%9D%EC%84%B13.png)
<br><br><br>  


인슨턴스 유형을 클릭후 'inf'를 치고, 제일싼 inf1.xlarge를 선택한다  
![Alt text](ETC/image/ec2%EC%83%9D%EC%84%B14.png)
<br><br><br>  


키 페어는 방금 등록한 키로 선택한다  
![Alt text](ETC/image/ec2%EC%83%9D%EC%84%B15.png)
<br><br><br>  

네트워크 설정은 따로 수정 할 필요 없다. (기본값으로 보안그룹을 생성, 22번 포트에 접속할 수 있게 모든IP를 뚫어준다)  
![Alt text](ETC/image/ec2%EC%83%9D%EC%84%B16.png)
<br><br><br>  


스토리지는 넉넉하게 50GB정도 설정해 준다 ( 설치파일 및 딥러닝 실험하려면 공간이 부족하다 )  
![Alt text](ETC/image/ec2%EC%83%9D%EC%84%B17.png)
<br><br><br>  

설정을 다 했으면 우측 하단에 주황색으로 표시된 **인스턴스 시작**을 클릭한다  
![Alt text](ETC/image/ec2%EC%83%9D%EC%84%B18.png)
<br><br><br>  


주소보는법, ssh 접속화면


aws rds 엔드포인트 
database-2.crelw7ywu7sd.us-east-2.rds.amazonaws.com

------------------
### 2. 인스턴스 접속 및 AWS neuron sdk 환경 설치

vscode 원격환경에서 환경 셋팅을 진행 하도록 한다.

이제 이 git 파일 수정해야됨



```
python as pt
```










--------------------------





AWS 뉴런 sdk 실험

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
