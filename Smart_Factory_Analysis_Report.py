# =================================================================================
# 파일명: Smart_Factory_Analysis_Report.py
# 설명: PPT 슬라이드별 이미지 결과에 대응하는 통합 분석 코드
# =================================================================================

import os
import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 1. 데이터 로드 및 환경 설정 (setup.py 참고)
path = kagglehub.dataset_download("ziya07/smart-manufacturing-iot-cloud-monitoring-dataset")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

print(f"데이터 로드 완료: {df.shape}")

# ---------------------------------------------------------------------------------
# [PPT 7 슬라이드] 상관관계 분석 
# 주석: slide_07_image_04.png (Sensor Correlation Map)
# ---------------------------------------------------------------------------------
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".2f")
plt.title('Sensor Correlation Map (slide_07_image_04)', fontsize=14)
plt.savefig('slide_07_result.png')
plt.close()

# ---------------------------------------------------------------------------------
# [PPT 8~9 슬라이드] 환경 및 주요 센서 분포 분석
# 주석: slide_08_image_05.png / slide_09_image_06.png (Histogram & CDF)
# ---------------------------------------------------------------------------------
def save_dist_plot(features, filename, title):
    fig, axes = plt.subplots(2, len(features), figsize=(15, 10))
    for i, col in enumerate(features):
        # Histogram
        sns.histplot(data=df, x=col, hue='maintenance_required', ax=axes[0, i], kde=True)
        # CDF
        sns.ecdfplot(data=df, x=col, hue='maintenance_required', ax=axes[1, i])
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 환경 변수 분석 (습도, 압력, 에너지)
save_dist_plot(['humidity', 'pressure', 'energy_consumption'], 
               'slide_08_result.png', 'Environmental Sensors Analysis')

# 주요 센서 분석 (온도, 진동 등)
save_dist_plot(['temperature', 'vibration', 'predicted_remaining_life'], 
               'slide_09_result.png', 'Core Sensors Analysis')

# ---------------------------------------------------------------------------------
# [PPT 10~11 슬라이드] 상태별 유지보수 비율 분석 (mainO_data_rate.py 참고)
# 주석: slide_10_image_08, 09 / slide_11_image_10, 11
# ---------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. machine_status별 비율 (slide_10_image_08)
pd.crosstab(df['machine_status'], df['maintenance_required'], normalize='index').plot(
    kind='bar', stacked=True, ax=axes[0,0], color=['#3498db', '#e74c3c'])
axes[0,0].set_title('Status vs Maintenance (slide_10_image_08)')

# 2. anomaly_flag별 비율 (slide_11_image_10)
pd.crosstab(df['anomaly_flag'], df['maintenance_required'], normalize='index').plot(
    kind='bar', stacked=True, ax=axes[0,1], color=['#3498db', '#e74c3c'])
axes[0,1].set_title('Anomaly vs Maintenance (slide_11_image_10)')

# 3. downtime_risk별 비율 (slide_11_image_11)
# 데이터에 해당 컬럼이 없을 경우를 대비해 처리
if 'downtime_risk' in df.columns:
    pd.crosstab(df['downtime_risk'], df['maintenance_required'], normalize='index').plot(
        kind='bar', stacked=True, ax=axes[1,0], color=['#3498db', '#e74c3c'])
    axes[1,0].set_title('Downtime Risk vs Maintenance (slide_11_image_11)')

# 4. RUL 분포 (slide_13_image_14)
sns.histplot(data=df, x='predicted_remaining_life', hue='maintenance_required', 
             ax=axes[1,1], palette='coolwarm')
axes[1,1].set_title('RUL Distribution (slide_13_image_14)')

plt.tight_layout()
plt.savefig('slide_10_13_combined_result.png')
plt.close()

# ---------------------------------------------------------------------------------
# [PPT 12 슬라이드] 기계 가동 상태별 상세 분석 (mainX_data.py 참고)
# 주석: slide_12_image_12.png, slide_12_image_13.png
# ---------------------------------------------------------------------------------
# 기계 상태가 Failure(2)인 경우와 정상인 경우의 센서 데이터 차이 시각화
plt.figure(figsize=(10, 6))
sns.boxplot(x='machine_status', y='temperature', data=df, palette='Set3')
plt.title('Temperature by Machine Status (slide_12_image_12)')
plt.savefig('slide_12_result.png')
plt.close()

print("✅ 모든 슬라이드 대응 이미지 생성이 완료되었습니다.")