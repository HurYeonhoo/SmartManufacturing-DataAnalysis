# setup

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
# =================================================================================
# 정상상태 (machine=0,1), 조건을 제외한 상황에서
# 유지보수 요구 (maintenance = 1)인 경우가 고장상태(machine_status=2)인지 파악하는 코드
# =================================================================================

# 제외(필터링)할 조건들을 하나의 불리언 마스크로 묶음
# - machine_status가 0 또는 1이면 제외
# - temperature가 90 이상이면 제외
# - vibration이 80 이상이면 제외
# - predicted_remaining_life가 20 이하이면 제외
exclude_condition = (
    (df["machine_status"].isin([0, 1])) |
    (df["temperature"] >= 90) |
    (df["vibration"] >= 80) |
    (df["predicted_remaining_life"] <= 20)
)

# 제외 조건에 해당하지 않는 행만 남김 (~ 는 boolean 반전)
# copy()로 뷰/원본 참조 이슈(SettingWithCopy 등)를 피하고 안전하게 작업
remaining = df.loc[~exclude_condition].copy()

# remaining 데이터에서 maintenance_required가 1인 행을 찾는 마스크
mask_req1 = (remaining["maintenance_required"] == 1)

# 전체/남은 데이터 크기와 remaining 내 maintenance_required==1 개수 계산
total = len(df)                     # 원본 전체 행 수
rem_n = len(remaining)              # 필터링 후 remaining 행 수
cnt = int(mask_req1.sum())          # True(=1) 개수 = 조건 만족 행 수

# 보기 좋게 구분선과 함께 요약 출력
print("\n" + "="*70)
print("검증: 제외조건 적용 후 remaining에서 maintenance_required==1 존재 여부")
print("-"*70)
print(f"전체 rows: {total:,}")
print(f"remaining rows: {rem_n:,} (전체 대비 {rem_n/total*100:.2f}%)")
# rem_n이 0일 때 0으로 나누는 에러를 피하기 위해 max(rem_n, 1) 사용
print(f"remaining 중 maintenance_required==1: {cnt:,} (remaining 대비 {cnt/max(rem_n,1)*100:.2f}%)")
print("-"*70)

# remaining 안에 maintenance_required==1 이 있는지 최종 결론 출력
if cnt == 0:
    print("== 결론 ==")
    print("remaining 데이터에 maintenance_required == 1 인 값이 없습니다.")
else:
    print("== 결론 ==")
    print("remaining 데이터에 maintenance_required == 1 인 값이 존재합니다. 아래 샘플을 확인하세요.\n")

    # 보여줄 컬럼 후보 중 remaining에 실제로 존재하는 컬럼만 골라서 표시
    cols_to_show = [c for c in [
        "machine_id", "timestamp",
        "machine_status", "temperature", "vibration",
        "predicted_remaining_life", "maintenance_required"
    ] if c in remaining.columns]

    print(remaining.loc[mask_req1, cols_to_show].head(20))

print("="*70 + "\n")
