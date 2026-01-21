import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. 데이터 불러오기 및 폴더 설정
df = pd.read_csv('smart_manufacturing_data.csv')

# [핵심] 날짜 데이터를 문자열에서 날짜 객체로 변환 (데이터 신뢰도 확보)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 저장할 폴더 생성
save_dir = '분석결과_리포트'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 스타일 설정
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# 분석용 데이터 가공 (최신 상태 추출 및 그룹화)
latest_data = df.sort_values('timestamp').groupby('machine_id').last()

def classify_status(rul):
    if rul <= 15: return 'Critical'
    elif rul <= 45: return 'Warning'
    else: return 'Normal'
df['status_group'] = df['predicted_remaining_life'].apply(classify_status)

# --- (1) 01_센서상관관계_지도.png ---
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".2f")
plt.title('01. Sensor Correlation Map', fontweight='bold')
plt.savefig(f'{save_dir}/01_센서상관관계_지도.png', dpi=300, bbox_inches='tight')
plt.close()

# --- (2) 02_유지보수대상_온도분포.png ---
plt.figure(figsize=(8, 6))
sns.boxplot(x='maintenance_required', y='temperature', data=df, palette='Set2')
plt.title('02. Temperature Distribution', fontweight='bold')
plt.savefig(f'{save_dir}/02_유지보수대상_온도분포.png', dpi=300)
plt.close()

# --- (3) 03_고장원인_비중분석.png ---
plt.figure(figsize=(8, 6))
fail_counts = df[df['failure_type'] != 'Normal']['failure_type'].value_counts()
plt.pie(fail_counts, labels=fail_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('03. Failure Type Breakdown', fontweight='bold')
plt.savefig(f'{save_dir}/03_고장원인_비중분석.png', dpi=300)
plt.close()

# --- (4) 04_기계상태_운영현황.png ---
plt.figure(figsize=(8, 6))
sns.countplot(x='machine_status', hue='maintenance_required', data=df, palette='viridis')
plt.title('04. Machine Status Summary', fontweight='bold')
plt.savefig(f'{save_dir}/04_기계상태_운영현황.png', dpi=300)
plt.close()

# --- (5) 05_위험그룹별_진동차이.png ---
plt.figure(figsize=(10, 7))
sns.boxplot(x='status_group', y='vibration', data=df, order=['Normal', 'Warning', 'Critical'], 
            palette={'Normal':'#2ca02c', 'Warning':'#ff7f0e', 'Critical':'#d62728'})
plt.title('05. Vibration Level by Risk Group', fontweight='bold', color='darkred')
plt.savefig(f'{save_dir}/05_위험그룹별_진동차이.png', dpi=300)
plt.close()

# --- (6) 06_최종_정비전략_지침.png ---
plt.figure(figsize=(10, 6))
plt.axis('off')
critical_id = latest_data.nsmallest(1, 'predicted_remaining_life').index[0]
est_downtime = latest_data.nsmallest(5, 'predicted_remaining_life')['downtime_risk'].sum() * 24

summary_text = (
    " [ PREDICTIVE MAINTENANCE ACTION PLAN ]\n\n"
    f"▶ PRIMARY TARGET: Machine #{critical_id}\n"
    f"▶ EST. PRODUCTION LOSS: ~{est_downtime:.1f} Hours\n\n"
    " [ KEY TASKS ]\n"
    " 1. High vibration detected in Critical group (Ref #05)\n"
    " 2. Inspect bearing/motor of the target machine\n"
    " 3. Update maintenance schedule for 'Warning' group\n"
    " 4. Full list saved in '긴급점검리스트.csv'"
)
plt.text(0.1, 0.5, summary_text, fontsize=14, family='monospace', bbox={'facecolor':'white', 'edgecolor':'black', 'pad':20})
plt.savefig(f'{save_dir}/06_최종_정비전략_지침.png', dpi=300, bbox_inches='tight')
plt.close()

# (부록) 정비팀용 상세 리스트 저장
latest_data.nsmallest(10, 'predicted_remaining_life').to_csv('긴급점검리스트.csv', encoding='utf-8-sig')

print(f"✅ '{save_dir}' 폴더 내에 6개의 한글 파일명 이미지가 생성되었습니다.")
print(f"✅ '긴급점검리스트.csv' 파일이 생성되었습니다.")