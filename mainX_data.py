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
# ======================================
# 유지보수와 관계없는 데이터값들의 관계표현
# ======================================


features = ["humidity", "pressure", "energy_consumption"]

# maintenance_required 값(0/1)을 사람이 보기 좋은 라벨로 바꿔서 범례에 쓰기 위한 매핑
maintenance_name = {
    0: "maintenance_required (0)",
    1: "maintenance_required (1)"
}

# 원본 df를 건드리지 않기 위해 복사본으로 작업
df = df.copy()

# 컬럼명 앞/뒤 공백 때문에 KeyError가 나는 상황을 예방 (예: "humidity " 같은 경우)
df.columns = df.columns.str.strip()

# 경험적 누적분포함수(ECDF)를 계산하는 함수
# - 입력: 숫자 데이터(Series/array)
# - 출력: (정렬된 x값, 누적비율 y값)
# - 빈 배열이면 (None, None)을 반환하여 이후 plot에서 건너뛰게 함
def ecdf(x):
    x = pd.Series(x).dropna().to_numpy()  # 결측 제거 후 numpy 배열로 변환
    x = np.sort(x)                        # ECDF는 x가 오름차순이어야 선이 자연스럽게 연결됨
    if x.size == 0:
        return None, None
    y = np.arange(1, x.size + 1) / x.size # 1/n, 2/n, ..., n/n 형태의 누적비율
    return x, y

# maintenance_required에 실제로 존재하는 상태값(예: [0, 1])만 추출
# - 결측치 제거
# - 정렬하여 항상 같은 순서로 그려지게 함(범례/색상 비교가 안정적)
unique_statuses = sorted(df["maintenance_required"].dropna().unique())

# -----------------------------
# Figure(페이지) 구성
# -----------------------------
# 2행 x 3열:
# - 첫 번째 행(axes[0, j])  : 히스토그램(빈도수)
# - 두 번째 행(axes[1, j]) : ECDF(CDF)
# sharex="col": 같은 열에서 위/아래 그래프가 x축 범위를 공유
#               (히스토그램과 CDF를 같은 x 스케일로 비교 가능)
# sharey="row": 같은 행에서 y축 범위를 공유
#               (히스토그램끼리, CDF끼리 y축 스케일이 통일되어 비교가 쉬움)
fig, axes = plt.subplots(
    2, 3,
    figsize=(18, 9),
    sharex="col",
    sharey="row"
)

# -----------------------------
# 각 feature(습도/압력/에너지)마다 한 열(column)을 사용
# -----------------------------
for j, feature in enumerate(features):
    ax_hist = axes[0, j]  # 위쪽: 히스토그램 축
    ax_cdf  = axes[1, j]  # 아래쪽: CDF 축

    # =============================
    # Histogram
    # =============================
    for s in unique_statuses:
        # 현재 상태(s)에 해당하는 feature 값만 추출
        # dropna(): 히스토그램은 NaN을 그릴 수 없으므로 제외
        sub = df.loc[df["maintenance_required"] == s, feature].dropna()

        # 해당 상태 데이터가 없다면(전부 NaN이거나 행이 없는 경우) 스킵
        if sub.empty:
            continue

        # hist 주요 옵션 설명
        # - bins=30: 구간 개수(더 크면 더 촘촘하지만 노이즈가 늘 수 있음)
        # - density=False: y축을 '빈도수(count)'로 표시 (True면 확률밀도)
        # - alpha=0.35: 두 상태를 겹쳐 그릴 때 서로 보이도록 반투명 처리
        # - edgecolor="black": 막대 경계를 검정으로 줘서 구간이 또렷하게 보이게 함
        # - label=...: 범례에 표시될 텍스트(0/1을 사람이 보기 좋게)
        ax_hist.hist(
            sub,
            bins=30,
            density=False,
            alpha=0.35,
            edgecolor="black",
            label=maintenance_name.get(s, str(s))
        )

    # 히스토그램 축 꾸미기
    # - title: 어떤 그래프인지 + 어떤 feature인지 표시
    # - xlabel/ylabel: 축 의미 명시
    # - grid: 눈금선으로 값 비교가 쉬워짐
    # - legend: 상태별 색상/라벨 구분
    ax_hist.set_title(f"Histogram (Count): {feature}")
    ax_hist.set_xlabel(feature)
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.3)
    ax_hist.legend()

    # =============================
    # CDF(ECDF) 정리
    # =============================
    # 목적: 같은 feature에 대해 상태별로 "누적분포"를 비교
    #       예) CDF가 더 왼쪽에 있으면 전반적으로 값이 더 작다는 의미
    for s in unique_statuses:
        # 현재 상태(s)의 원본 값(결측 포함)을 가져오고, ecdf 내부에서 dropna 처리
        sub = df.loc[df["maintenance_required"] == s, feature]

        # ECDF 계산: x(정렬된 값), y(누적확률)
        x, y = ecdf(sub)

        # 데이터가 전혀 없으면(None 반환) 스킵
        if x is None:
            continue

        # 선 그래프로 ECDF를 그림
        # - x축: feature 값
        # - y축: 해당 값 이하의 비율(누적)
        ax_cdf.plot(
            x, y,
            label=maintenance_name.get(s, str(s))
        )

    # CDF 축 꾸미기
    # - CDF는 정의상 0~1 범위이므로 ylim 고정
    ax_cdf.set_title(f"CDF: {feature}")
    ax_cdf.set_xlabel(feature)
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_ylim(0, 1)
    ax_cdf.grid(alpha=0.3)
    ax_cdf.legend()

# 레이아웃 정리 후 출력
plt.tight_layout()  # 범례/라벨이 잘리지 않도록 여백 자동 조정
plt.show()          # 그래프 표시