# =================================================================================
# 파일명: 01_Smart_Factory_Quick_Dashboard.py
# 설명: 스마트 팩토리의 주요 IoT 센서 데이터를 2x3 그리드 형태의 한 장의 이미지로 요약
# 생성 폴더: 01_Smart_Factory_Quick_Dashboard_image
# 주요 차트: 상관관계 히트맵, 온도 분포, 고장 비중, 상태별 현황, 진동 분석, 에너지 소모량
# =================================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# 한글 깨짐 방지 및 스타일 설정
plt.rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

# 저장 폴더 생성
save_dir = '01_Smart_Factory_Quick_Dashboard_image'
os.makedirs(save_dir, exist_ok=True)

# 데이터 로드
df = pd.read_csv('smart_manufacturing_data.csv')

# 대시보드 생성 (2행 3열)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.suptitle('스마트 제조 IoT 데이터 분석 통합 대시보드', fontsize=18, fontweight='bold')

# (1) 상관관계 히트맵
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='RdYlGn', ax=axes[0,0], fmt=".2f", annot_kws={"size": 9})
axes[0,0].set_title('1. 센서 데이터 상관관계 지도', fontsize=12)

# (2) 정비 여부별 온도 분포
sns.boxplot(x='maintenance_required', y='temperature', data=df, ax=axes[0,1], palette='Set2')
axes[0,1].set_title('2. 정비 필요 유무별 온도 분포', fontsize=12)

# (3) 고장 유형별 비중
failure_counts = df['failure_type'].value_counts()
axes[0,2].pie(failure_counts, labels=failure_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
axes[0,2].set_title('3. 전체 고장 유형 발생 비중', fontsize=12)

# (4) 운영 상태별 정비 건수
sns.countplot(x='machine_status', hue='maintenance_required', data=df, ax=axes[1,0])
axes[1,0].set_title('4. 기계 가동 상태별 정비 현황', fontsize=12)

# (5) 온도와 진동의 관계
sns.scatterplot(x='temperature', y='vibration', hue='maintenance_required', data=df, ax=axes[1,1], alpha=0.5)
axes[1,1].set_title('5. 온도 대비 진동 수치 분석', fontsize=12)

# (6) 고장 유형별 에너지 소모량
sns.barplot(x='failure_type', y='energy_consumption', data=df, ax=axes[1,2], palette='viridis')
axes[1,2].set_title('6. 고장 유형별 평균 에너지 소모', fontsize=12)

# 파일 저장
output_path = f'{save_dir}/종합_분석_대시보드.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ [작업 완료] 통합 대시보드 이미지가 '{output_path}'에 저장되었습니다.")