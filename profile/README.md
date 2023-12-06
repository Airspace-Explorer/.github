# 데이터분석캡스톤디자인 프로젝트

##  Airspace-Explorer 
- 지도 교수: 이대호
- 팀원:  
  소프트웨어융합학과 구현서 2018102091 (PM)  
  소프트웨어융합학과 김민환 2019102081  
  소프트웨어융합학과 이정원 2020110480  

## 개요
[조류 충돌]      
우리 말로 '조류 충돌'이라고 불리는 '버드 스트라이크(Bird Strike)'는 조류가 비행기에 부딪히거나 엔진 속에 빨려 들어가는 현상을 말한다. 주로 공항 부근, 그리고 이착륙 시 주로 발생하는데, 우리나라에서는 조류충돌 사고가 매년 100~200건 이상 발생 하고 있으며 지난해 미국 연방항공청에서는 1만 7천 건이 넘는 신고가 접수 되었다. 실제로 1.8kg의 새가 시속 960km로 비행하는 항공기와 부딪치면 64t 무게의 충격이 발생하며, 전 세계적 피해규모는 연간 약 1조원으로 추정된다. 최근 5년간 항공기-조류간 충돌은 주로 공항구역에서 발생하고 있으며, 이를 예방하기 위해 공항에서는 사격팀 운영, 천적류 사육을 통해 노력하고 있지만 효과가 미비한 상황이다. 그래서 본팀은 최소 수십명에서 많게는 수백명 이상의 사망자를 발생시키는 버드 스트라이크 방지를 위해 Faster-RCNN, YOLOF, SSD 기반의 조류를 실시간으로 Detection 하는 모델을 학습을 통해 구축하여 다양한 기상 환경에서 비행중인 상공물체(조류)를 실시간으로 탐지할 것이며, 또한 Detection 모델과 ReID 모델을 결합하여 DeepSORT 모델을 구축한 뒤 실시간으로 상공 물체(조류)를 추적 할 것이다.

## 프로젝트 목표  
- 3개의 Object Detection Model(Faster-RCNN, YOLOF, SSD) Training & Evaluation & Visualization & Inference 및 성능 최적화 수행
- DeepSORT에 대한 Detection Model Training, Evaluation & ReID Model Training, Evaluation
- ReID Model을 사용한 DeepSORT Model과 ReID Model을 사용하지 않은 DeepSORT Model 간의 성능 비교

## SPEC & Runtime Environment
[공통 Spec]  
Framework: MMDetection  
Learning_rate=0.02 / 8  
Workers_per_gpu: 4  
Batch_size: 16  
Epochs: 100  
Classes = ('Bird', 'Airplane', 'Helicopter', 'FighterPlane', 'Paragliding’ , 'Drone’)  
Visualization Tool: Tensorboard, Matplotlib  

[Runtime Environment]  
Sys.platform: linux  
Python: 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0]  
CUDA available: True  
GPU 0: NVIDIA GeForce RTX 3090  
CUDA_HOME: /usr/local/cuda  
NVCC: Cuda compilation tools, release 11.7, V11.7.99  
GCC: gcc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0  
PyTorch: 1.13.0+cu116  
PyTorch compiling details: PyTorch built with:(- GCC 9.3, - C++ Version: 201402,- Intel(R) 64        
architecture applications)  

## 데이터세트  
![ASD](https://github.com/Airspace-Explorer/.github/assets/104192273/46c5aa27-e176-4d59-9c72-51da3534ca09)  
AI-Hub의 Small object detection을 위한 이미지 데이터셋을 이용하였다. 해당 데이터 셋에서는 이미지(2800x2100 해상도) 내에 일정 크기 이하의 소형 객체(200x200 픽셀 크기 이하)들만 존재하며   
이미지에 대한 JSON 형태의 어노테이션 파일 또한 포함하고 있다.  




  

