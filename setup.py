#setup

import os                      # 파일/폴더 경로 처리, 디렉토리 목록 조회
import numpy as np             # 수치 계산
import pandas as pd            # CSV 로딩 및 데이터프레임 처리
import kagglehub               # Kaggle 데이터셋 다운로드
import matplotlib.pyplot as plt  # 시각화(기본)
import matplotlib.font_manager as fm  # 한글 폰트 설정용
import seaborn as sns          # 시각화(고급)

# 데이터셋 다운로드 : kagglehub가 데이터셋 내려받고, 로컬에 저장된 폴더 경로 반환
path = kagglehub.dataset_download(
    "ziya07/smart-manufacturing-iot-cloud-monitoring-dataset"
)

#print("Path to dataset files:", path) # @디버깅용 파일경로 불러옴

# csv 파일 불러오기
files = os.listdir(path) # 다운된 폴더(path) 안의 파일 목록을 리스트로 가져오게 됨.
csv_file = [f for f in files if f.endswith('.csv')][0] # csv.인 파일만 골라서 그 중 첫 번째 파일 선

df = pd.read_csv(os.path.join(path, csv_file)) #(path, csv_file)을 합쳐 경로를 만들고 DataFrame으로 로드 
import matplotlib.font_manager as fm

# ---- 한글 폰트 설정(반드시 그래프 그리기 전에) ----#
font_path = r"C:\Windows\Fonts\malgun.ttf" #윈도우에 설치된 '맑은고딕' 폰트 실제 경로
fp = fm.FontProperties(fname=font_path) #위의 경로에 있는 폰트를 쓰게하는 코드

# ----------------------------------------------------------------------------------------------------------------#
