# =================================================================================
# 파일명: 04_Executive_Decision_Support_System.py
# 설명: 경영진의 신속한 의사결정을 돕기 위해 리드타임 분석, 기계별 위험도 랭킹,
#       상태별 온도 변동폭(Violin Plot) 및 시계열 트렌드를 심층 분석합니다.
# 생성 폴더: 04_Executive_Decision_Support_System_image
# 주요 결과물: 05 리드타임 분포, 06 온도 바이올린 플롯, 07 위험기계 시계열 트렌드, 08 통합 경영 대시보드
# =================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# [1] 한글 깨짐 방지 및 스타일 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

# [2] 저장 폴더 생성
save_dir = '04_Executive_Decision_Support_System_image'
os.makedirs(save_dir, exist_ok=True)

# [3] 데이터 로드 및 시계열 처리
df = pd.read_csv('smart_manufacturing_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- [4] 경영진용 심층 분석 및 시각화 (모든 로직 포함) ---

# 05. 리드타임 분포 분석 (위험 감지 후 실제 정비까지 걸린 시간)
def calc_lead_time(df):
    lead_times = []
    for mid, group in df.groupby('machine_id'):
        risk_times = group[group['downtime_risk'] == 1]['timestamp']
        maint_times = group[group['maintenance_required'] == 1]['timestamp']
        for mt in maint_times:
            prev_risk = risk_times[risk_times < mt]
            if len(prev_risk) > 0:
                lead_time = (mt - prev_risk.iloc[-1]).total_seconds() / 60
                lead_times.append(lead_time)
    return np.array(lead_times)

lead_times = calc_lead_time(df)
plt.figure(figsize=(10, 6))
plt.hist(lead_times, bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.title('05. 정비 리드타임 분포 (위험 감지 ~ 실제 정비)', fontsize=14, fontweight='bold')
plt.xlabel('리드타임 (분)', fontsize=12)
plt.ylabel('발생 빈도', fontsize=12)
plt.savefig(f'{save_dir}/05_Lead_Time_분포.png', dpi=300)
plt.close()

# 06. 기계 가동 상태별 온도 바이올린 플롯 (밀도와 분포를 동시에 확인)
plt.figure(figsize=(12, 8))
sns.violinplot(data=df, x='machine_status', y='temperature', hue='maintenance_required', split=True, palette='Set2')
plt.title('06. 기계 운영 상태 및 정비 여부에 따른 온도 변동성', fontsize=14, fontweight='bold')
plt.savefig(f'{save_dir}/06_상태별_온도_바이올린.png', dpi=300)
plt.close()

# 07. TOP 5 고위험 기계의 시계열 온도 트렌드 분석
risk_rank = df.groupby('machine_id')['downtime_risk'].mean().sort_values(ascending=False)
top5_risk_ids = risk_rank.head(5).index

plt.figure(figsize=(16, 10))
for mid in top5_risk_ids:
    # 시간 단위로 평균 온도를 리샘플링하여 추세 파악
    m_data = df[df['machine_id'] == mid].resample('1H', on='timestamp')['temperature'].mean()
    plt.plot(m_data.index, m_data.values, label=f'기계 #{mid}', marker='o', markersize=4, alpha=0.8)

plt.title('07. 가동 중단 위험 TOP 5 기계의 온도 변화 추이', fontsize=15, fontweight='bold')
plt.xlabel('분석 시간', fontsize=12)
plt.ylabel('평균 온도 ($^\circ$C)', fontsize=12)
plt.legend(title="위험 기계 ID", loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(f'{save_dir}/07_TOP5_위험기계_트렌드.png', dpi=300)
plt.close()

# 08. 최종 경영 의사결정 요약 대시보드
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('08. 경영진 의사결정 지원용 요약 리포트', fontsize=20, fontweight='bold')

# (A) 정비 여부별 진동 수준 비교
sns.boxplot(data=df, x='maintenance_required', y='vibration', ax=axes[0,0], palette='pastel')
axes[0,0].set_title('A. 정비 대상 기계의 진동 특성', fontsize=13)

# (B) 기계별 위험도 TOP 10 (바 차트)
risk_rank.head(10).plot(kind='bar', ax=axes[0,1], color='red', alpha=0.6)
axes[0,1].set_title('B. 가동 중단 위험 상위 10개 기계', fontsize=13)

# (C) 고장 원인 비중
df['failure_type'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1,0], colors=sns.color_palette('pastel'))
axes[1,0].set_title('C. 전체 설비 고장 원인 통계', fontsize=13)

# (D) 핵심 변수(온도, 진동, 위험도) 상관관계
sns.heatmap(df[['temperature','vibration','downtime_risk','energy_consumption']].corr(), annot=True, ax=axes[1,1], cmap='coolwarm')
axes[1,1].set_title('D. 핵심 생산 지표 상관계수', fontsize=13)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'{save_dir}/08_최종_의사결정_대시보드.png', dpi=300)
plt.close()

print(f"✅ [작업 완료] 경영 인사이트 분석 리포트 및 이미지 4종이 '{save_dir}' 폴더에 모두 저장되었습니다.")