# =================================================================================
# 파일명: 03_Sensor_Deep_Dive_Presentation.py
# 설명: 프레젠테이션(PPT) 보고를 위해 고화질의 개별 통계 차트를 생성하며,
#       한글 폰트 지원 및 상세한 범례, 축 설명을 포함하여 가독성을 높였습니다.
# 생성 폴더: 03_Sensor_Deep_Dive_Presentation_image
# 주요 차트: 01 고장 원인 비중(Explode Pie), 02 센서 상관관계 히트맵,
#           03 센서 데이터 밀도 분포(KDE), 04 기계별 가동 중단 위험도 평가
# =================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# [1] 한글 폰트 설정 및 마이너스 기호 깨짐 방지
plt.rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False
# Seaborn의 기본 폰트도 Malgun Gothic으로 설정하여 일관성 유지
sns.set_theme(style="whitegrid", font='Malgun Gothic')

# [2] 발표용 고화질 이미지를 저장할 폴더 생성
save_dir = '03_Sensor_Deep_Dive_Presentation_image'
os.makedirs(save_dir, exist_ok=True)

# [3] 데이터 로드 및 전처리 (날짜 형식 변환, 이진 변수 타입 지정)
df = pd.read_csv('smart_manufacturing_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
binary_cols = ['anomaly_flag', 'downtime_risk', 'maintenance_required']
df[binary_cols] = df[binary_cols].astype('int8')

# --- [4] 고도화된 발표용 개별 차트 생성 및 저장 (누락 없이 모두 포함) ---

# 차트 1: 고장 원인별 발생 비중 (정상 데이터를 제외하고 실제 고장 사례만 분석)
plt.figure(figsize=(10, 8))
fail_df = df[df['failure_type'] != 'Normal']
fail_counts = fail_df['failure_type'].value_counts()
plt.pie(fail_counts, labels=fail_counts.index, autopct='%1.1f%%', startangle=140, 
        explode=[0.05]*len(fail_counts), colors=sns.color_palette('pastel'))
plt.title('1. 설비 고장 원인별 발생 비중 (정상 제외)', fontsize=16, fontweight='bold')
plt.legend(title="고장 상세 유형", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10) # 범례 추가
plt.savefig(f'{save_dir}/01_발표용_고장유형_비중.png', dpi=300, bbox_inches='tight')
plt.close()

# 차트 2: 센서 데이터 및 운영 지표 상관계수 히트맵
plt.figure(figsize=(12, 10))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt='.2f', center=0, annot_kws={"size": 9}) # 어노테이션 폰트 사이즈 조정
plt.title('2. 센서 데이터 및 운영 지표 상관계수 분석', fontsize=16, fontweight='bold')
plt.xlabel('비교 대상 변수군', fontsize=12) # 축 이름 추가
plt.ylabel('기준 변수군', fontsize=12) # 축 이름 추가
plt.savefig(f'{save_dir}/02_발표용_센서_상관관계_히트맵.png', dpi=300, bbox_inches='tight')
plt.close()

# 차트 3: 주요 센서 데이터 분포 및 밀도 분석 (온도, 진동)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(df['temperature'], kde=True, ax=axes[0], color='orange', label='온도 데이터 분포')
axes[0].set_title('3-1. 온도(Temperature) 수집 데이터 분포', fontsize=13)
axes[0].set_xlabel('온도 ($^\circ$C)', fontsize=11) # 축 이름 추가
axes[0].set_ylabel('데이터 수집 빈도', fontsize=11) # 축 이름 추가
axes[0].legend(fontsize=10) # 범례 추가

sns.histplot(df['vibration'], kde=True, ax=axes[1], color='blue', label='진동 데이터 분포')
axes[1].set_title('3-2. 진동(Vibration) 수집 데이터 분포', fontsize=13)
axes[1].set_xlabel('진동 (mm/s)', fontsize=11) # 축 이름 추가
axes[1].set_ylabel('데이터 수집 빈도', fontsize=11) # 축 이름 추가
axes[1].legend(fontsize=10) # 범례 추가
plt.tight_layout() # 서브플롯 간 여백 자동 조절
plt.savefig(f'{save_dir}/03_발표용_센서_분포_분석.png', dpi=300)
plt.close()

# 차트 4: 기계별 가동 중단 위험도 평가 (전체 기계 ID 0~49)
plt.figure(figsize=(15, 6))
risk_by_machine = df.groupby('machine_id')['downtime_risk'].mean().sort_values(ascending=False)
risk_by_machine.plot(kind='bar', color='red', alpha=0.7, label='기계별 평균 위험 수치')
plt.title('4. 전체 기계 가동 중단 위험도 평가 (ID 0~49)', fontsize=16, fontweight='bold')
plt.xlabel('기계 식별 번호 (Machine ID)', fontsize=12) # 축 이름 추가
plt.ylabel('가동 중단 발생 위험 확률 (0.0~1.0)', fontsize=12) # 축 이름 추가
plt.legend(fontsize=10) # 범례 추가
plt.savefig(f'{save_dir}/04_발표용_기계별_위험도_현황.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ [작업 완료] 발표용 고해상도 개별 차트 4종이 '{save_dir}'에 저장되었습니다.")