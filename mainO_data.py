#setup

# 사용할 라이브러리 정리 
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
# ================================================================
# 유지보수에 영향을 미치는 데이터들의 관계분석
# - machine_status가 0 또는 1인 데이터만 대상으로
# - (temperature >= 90) 또는 (vibration >= 80) 조건(cond)이
#   anomaly_flag / downtime_risk(0/1)에 어떤 분포 차이를 만드는지
# ================================================================

# 필터: machine_status가 0 또는 1인 행만 True로 표시하는 마스크
ms01 = df["machine_status"].isin([0, 1])

# 조건(cond) 정의
# 온도 90도 이상 OR 진동 80 이상이면 True
cond = (df["temperature"] >= 90) | (df["vibration"] >= 80)

# 분석에 필요한 컬럼만 뽑아서 부분 데이터프레임 생성
#    - anomaly_flag, downtime_risk만 가져오고
#    - 원본 df 보호를 위해 copy() 사용
sub = df.loc[ms01, ["anomaly_flag", "downtime_risk"]].copy()

# cond도 sub에 추가
#  - cond는 df 전체 길이의 시리즈이므로 ms01로 같은 행만 맞춰서 가져옴
#  - heatmap을 0/1 테이블로 만들기 좋게 True/False를 int(0/1)로 변환
sub["cond"] = cond.loc[ms01].astype(int)

# anomaly_flag / downtime_risk가 bool/object로 들어있을 수 있으니
# 전부 int(0/1)로 강제 변환해 교차표 계산이 안정적으로 되게 함
for c in ["cond", "anomaly_flag", "downtime_risk"]:
    sub[c] = sub[c].astype(int)

# 2x2 교차표(Counts)와 행 기준 비율(Row %)을 같이 만드는 함수
#  - index: 행(여기서는 cond: 0/1)
#  - col:   열(여기서는 anomaly_flag 또는 downtime_risk: 0/1)
def row_pct_crosstab(index, col):
    # 교차표 생성 후, 행/열을 [0,1]로 고정(reindex)해서
    # 특정 값이 데이터에 없더라도 2x2 형태를 유지하게 함(fill_value=0)
    ct = pd.crosstab(index, col).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    # 행 기준 비율(%) 계산
    # - 각 행의 합(ct.sum(axis=1))으로 나눠서 cond=0/1 각각의 분포로 해석
    row_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    return ct, row_pct

# anomaly_flag에 대한 (Counts, Row%) 테이블
ct_anom, rt_anom = row_pct_crosstab(sub["cond"], sub["anomaly_flag"])

# downtime_risk에 대한 (Counts, Row%) 테이블
ct_risk, rt_risk = row_pct_crosstab(sub["cond"], sub["downtime_risk"])

# ------------------------------------------------------------
# (2x2 heatmap)

# 2행 x 2열 서브플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 히트맵 컬러맵 설정
#     - anomaly는 파란 계열, risk는 초록 계열로 통일
CMAP_ANOM = "Blues"
CMAP_RISK = "Greens"

# 히트맵 공통 옵션(Counts용)
# - annot=True: 셀 안에 숫자 표시
# - fmt="d": 정수 포맷
# - square=True: 셀을 정사각형으로
# - linewidths/linecolor: 그리드 라인
# - cbar=True: 컬러바 표시
heat_cnt_common = dict(
    annot=True, fmt="d", square=True,
    linewidths=0.5, linecolor="white", cbar=True
)

# 히트맵 공통 옵션(Row%용)
# - fmt=".1f": 소수점 1자리로 표시
# - vmin=0, vmax=100: % 범위를 0~100으로 고정(그래프 간 비교 용이)
heat_pct_common = dict(
    annot=True, fmt=".1f", vmin=0, vmax=100, square=True,
    linewidths=0.5, linecolor="white", cbar=True
)

# (0,0) anomaly count 히트맵
sns.heatmap(ct_anom, ax=axes[0, 0], cmap=CMAP_ANOM, **heat_cnt_common)
axes[0, 0].set_title("Counts: cond × anomaly_flag")
axes[0, 0].set_xlabel("anomaly_flag (0/1)")
axes[0, 0].set_ylabel("cond (0/1)")

# (0,1) anomaly row% 히트맵
sns.heatmap(rt_anom, ax=axes[0, 1], cmap=CMAP_ANOM, **heat_pct_common)
axes[0, 1].set_title("Row %: P(anomaly_flag | cond) [%]")
axes[0, 1].set_xlabel("anomaly_flag (0/1)")
axes[0, 1].set_ylabel("cond (0/1)")

# (1,0) risk count 히트맵
sns.heatmap(ct_risk, ax=axes[1, 0], cmap=CMAP_RISK, **heat_cnt_common)
axes[1, 0].set_title("Counts: cond × downtime_risk")
axes[1, 0].set_xlabel("downtime_risk (0/1)")
axes[1, 0].set_ylabel("cond (0/1)")

# (1,1) risk row% 히트맵
sns.heatmap(rt_risk, ax=axes[1, 1], cmap=CMAP_RISK, **heat_pct_common)
axes[1, 1].set_title("Row %: P(downtime_risk | cond) [%]")
axes[1, 1].set_xlabel("downtime_risk (0/1)")
axes[1, 1].set_ylabel("cond (0/1)")


# 레이아웃 정리 후 출력
plt.tight_layout()  # 범례/라벨이 잘리지 않도록 여백 자동 조정
plt.show()          # 그래프 표시