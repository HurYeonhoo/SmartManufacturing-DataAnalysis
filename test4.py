import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 환경 설정 (한글 폰트 및 마이너스 기호 설정)
plt.rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False

def run_comprehensive_analysis(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return

    # 2. 데이터 로드 및 전처리
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    binary_cols = ['anomaly_flag', 'downtime_risk', 'maintenance_required']
    df[binary_cols] = df[binary_cols].astype('int8')

    print("🚀 발표용 고도화 시각화를 시작합니다. (범례 및 축 이름 추가 완료)")

    # --- [차트 1] 고장 유형별 비중 (파이 차트) ---
    plt.figure(figsize=(10, 8))
    fail_df = df[df['failure_type'] != 'Normal']
    fail_counts = fail_df['failure_type'].value_counts()
    
    # 파이 차트의 각 조각 의미를 범례에 추가
    plt.pie(fail_counts, labels=fail_counts.index, autopct='%1.1f%%', startangle=140, 
            explode=[0.05]*len(fail_counts), colors=sns.color_palette('pastel'))
    plt.title('1. 고장 원인별 발생 비중 (정상 제외)', fontsize=15)
    plt.legend(title="고장 상세 유형", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.savefig('01_결과_고장유형_비중.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- [차트 2] 센서 상관관계 맵 (히트맵) ---
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt='.2f', center=0)
    
    plt.title('2. 센서 데이터 및 운영 지표 상관계수 분석', fontsize=15)
    plt.xlabel('비교 대상 변수군', fontsize=12)
    plt.ylabel('기준 변수군', fontsize=12)
    # 범례 대용: 우측 컬러바가 상관계수(1.0 ~ -0.4)를 나타냄
    plt.savefig('02_결과_센서_상관관계_히트맵.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- [차트 3] 주요 센서 분포 및 임계치 (PPT 재현) ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(df['temperature'], kde=True, ax=axes[0], color='orange', label='온도 데이터 분포')
    axes[0].set_title('3-1. 온도(Temperature) 수집 분포', fontsize=13)
    axes[0].set_xlabel('온도 ($^\circ$C)', fontsize=11)
    axes[0].set_ylabel('데이터 수집 빈도', fontsize=11)
    axes[0].legend()

    sns.histplot(df['vibration'], kde=True, ax=axes[1], color='blue', label='진동 데이터 분포')
    axes[1].set_title('3-2. 진동(Vibration) 수집 분포', fontsize=13)
    axes[1].set_xlabel('진동 (mm/s)', fontsize=11)
    axes[1].set_ylabel('데이터 수집 빈도', fontsize=11)
    axes[1].legend()

    plt.savefig('03_결과_센서_분포_분석.png', dpi=300)
    plt.show()

    # --- [차트 4] 기계별 가동 중단 위험도 (TOP 50) ---
    plt.figure(figsize=(15, 6))
    risk_by_machine = df.groupby('machine_id')['downtime_risk'].mean().sort_values(ascending=False)
    risk_by_machine.plot(kind='bar', color='red', alpha=0.7, label='기계별 평균 위험 수치')
    
    plt.title('4. 전 기계(ID 0~49) 가동 중단 위험도 평가', fontsize=15)
    plt.xlabel('기계 식별 번호 (Machine ID)', fontsize=12)
    plt.ylabel('가동 중단 발생 위험 확률 (0.0~1.0)', fontsize=12)
    plt.legend()
    plt.savefig('04_결과_기계별_위험도_현황.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("💾 4개의 고도화된 분석 차트가 저장되었습니다.")

if __name__ == "__main__":
    # 실제 파일명 'smart_manufacturing_data.csv'로 실행
    target_file = 'smart_manufacturing_data.csv' 
    run_comprehensive_analysis(target_file)