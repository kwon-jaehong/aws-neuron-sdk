# AWS ec2 inf 인스턴스 생성
---------------


- ec2메뉴에서 '인스턴스' > 인스턴스에서 우측 상단의 주황색으로 표시된 **인스턴스 시작**을 클릭한다   
![Alt text](../../ETC/image/ec2%EC%83%9D%EC%84%B11.png)
<br><br><br>  


- 인스턴스 이름에 아무거나 적는다  
![Alt text](../../ETC/image/ec2%EC%83%9D%EC%84%B12.png)
<br><br><br>  

- 인스턴스 이미지(OS)는 Ubuntu 20.04 LST를 선택한다 ( **이 문서에서는 우분투 20.04 환경으로 셋팅을 기본으로 함** )  
![Alt text](../../ETC/image/ec2%EC%83%9D%EC%84%B13.png)
<br><br><br>  


- 인슨턴스 유형을 클릭후 'inf'를 치고, 제일싼 inf.xlarge를 선택한다  
![Alt text](../../ETC/image/ec2%EC%83%9D%EC%84%B14.png)
<br><br><br>  


- 키 페어는 방금 등록한 키로 선택한다  
![Alt text](../../ETC/image/ec2%EC%83%9D%EC%84%B15.png)
<br><br><br>  

- 네트워크 설정은 따로 수정 할 필요 없다. (기본값으로 보안그룹을 생성, 22번 포트에 접속할 수 있게 모든IP를 뚫어준다)  
![Alt text](../../ETC/image/ec2%EC%83%9D%EC%84%B16.png)
<br><br><br>  


- 스토리지는 넉넉하게 50GB정도 설정해 준다 ( 설치파일 및 딥러닝 실험하려면 공간이 부족하다 )  
![Alt text](../../ETC/image/ec2%EC%83%9D%EC%84%B17.png)
<br><br><br>  

- 설정을 다 했으면 우측 하단에 주황색으로 표시된 **인스턴스 시작**을 클릭한다  
![Alt text](../../ETC/image/ec2%EC%83%9D%EC%84%B18.png)
<br><br><br>  

