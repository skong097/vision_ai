"""
전체 파이프라인 실행 - Master Script
Author: Home Safe Solution Team
Date: 2026-01-28

실행 순서:
1. Skeleton 추출
2. Feature Engineering
3. Auto Labeling (Binary & 3-Class)
4. Dataset 생성
5. Random Forest 학습 (Binary & 3-Class)
"""

import sys
import subprocess
from pathlib import Path
import time


class PipelineMaster:
    """전체 파이프라인 관리"""
    
    def __init__(self, base_dir='/home/gjkong/dev_ws/yolo/myproj'):
        self.base_dir = Path(base_dir)
        self.scripts_dir = self.base_dir / 'scripts'
        
        # 필요한 디렉토리
        self.dirs = {
            'data': self.base_dir / 'data',
            'accel': self.base_dir / 'accel',
            'skeleton': self.base_dir / 'skeleton',
            'features': self.base_dir / 'features',
            'labeled': self.base_dir / 'labeled',
            'dataset': self.base_dir / 'dataset',
            'models': self.base_dir / 'models',
        }
    
    def check_prerequisites(self):
        """사전 조건 확인"""
        
        print("=" * 60)
        print("🔍 사전 조건 확인")
        print("=" * 60)
        
        # 원본 데이터 확인
        video_files = list(self.dirs['data'].glob("fall-*-cam0.mp4"))
        accel_files = list(self.dirs['accel'].glob("fall-*-acc.csv"))
        
        print(f"\n✅ 동영상 파일: {len(video_files)}개")
        print(f"✅ 가속도 파일: {len(accel_files)}개")
        
        if len(video_files) == 0 or len(accel_files) == 0:
            print("\n❌ 오류: 원본 데이터가 없습니다!")
            print(f"   동영상 경로: {self.dirs['data']}")
            print(f"   가속도 경로: {self.dirs['accel']}")
            return False
        
        # 스크립트 확인
        required_scripts = [
            'extract_skeleton.py',
            'feature_engineering.py',
            'auto_labeling.py',
            'create_dataset.py',
            'train_random_forest.py'
        ]
        
        missing_scripts = []
        for script in required_scripts:
            if not (self.scripts_dir / script).exists():
                missing_scripts.append(script)
        
        if missing_scripts:
            print(f"\n⚠️  경고: 다음 스크립트가 없습니다:")
            for script in missing_scripts:
                print(f"   - {script}")
            print(f"\n   스크립트 경로: {self.scripts_dir}")
            return False
        
        print(f"\n✅ 모든 스크립트 확인 완료")
        return True
    
    def run_step(self, step_num, step_name, script_name):
        """단일 스텝 실행"""
        
        print("\n" + "=" * 60)
        print(f"Step {step_num}: {step_name}")
        print("=" * 60)
        
        script_path = self.scripts_dir / script_name
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.scripts_dir),
                capture_output=True,
                text=True
            )
            
            # 출력 표시
            if result.stdout:
                print(result.stdout)
            
            if result.returncode != 0:
                print(f"\n❌ 오류 발생!")
                if result.stderr:
                    print(result.stderr)
                return False
            
            elapsed = time.time() - start_time
            print(f"\n✅ Step {step_num} 완료! (소요 시간: {elapsed:.1f}초)")
            return True
            
        except Exception as e:
            print(f"\n❌ 오류: {e}")
            return False
    
    def run_pipeline(self, skip_skeleton=False):
        """전체 파이프라인 실행"""
        
        print("\n" + "🚀" * 30)
        print("HOME SAFE SOLUTION - 전체 파이프라인 실행")
        print("🚀" * 30 + "\n")
        
        # 사전 조건 확인
        if not self.check_prerequisites():
            print("\n❌ 사전 조건을 만족하지 못했습니다. 종료합니다.")
            return
        
        start_time = time.time()
        
        # Step 1: Skeleton 추출
        if not skip_skeleton:
            if not self.run_step(1, "Skeleton 추출", "extract_skeleton.py"):
                return
        else:
            print("\n⏭️  Step 1 (Skeleton 추출) 스킵됨")
        
        # Step 2: Feature Engineering
        if not self.run_step(2, "Feature Engineering", "feature_engineering.py"):
            return
        
        # Step 3: Auto Labeling
        if not self.run_step(3, "Auto Labeling", "auto_labeling.py"):
            return
        
        # Step 4: Dataset 생성
        if not self.run_step(4, "Dataset 생성", "create_dataset.py"):
            return
        
        # Step 5: Random Forest 학습
        if not self.run_step(5, "Random Forest 학습", "train_random_forest.py"):
            return
        
        # 완료
        total_time = time.time() - start_time
        
        print("\n" + "🎉" * 30)
        print("전체 파이프라인 완료!")
        print(f"총 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print("🎉" * 30 + "\n")
        
        # 결과 요약
        self.print_summary()
    
    def print_summary(self):
        """결과 요약"""
        
        print("=" * 60)
        print("📊 최종 결과 요약")
        print("=" * 60)
        
        print(f"\n📁 생성된 데이터:")
        print(f"   Skeleton:  {self.dirs['skeleton']}")
        print(f"   Features:  {self.dirs['features']}")
        print(f"   Labeled:   {self.dirs['labeled']}")
        print(f"   Dataset:   {self.dirs['dataset']}")
        
        print(f"\n🤖 학습된 모델:")
        print(f"   Binary:    {self.dirs['models'] / 'binary' / 'random_forest_model.pkl'}")
        print(f"   3-Class:   {self.dirs['models'] / '3class' / 'random_forest_model.pkl'}")
        
        print(f"\n📊 시각화 결과:")
        print(f"   Labeling:  {self.dirs['labeled'] / 'visualizations'}")
        print(f"   Binary:    {self.dirs['models'] / 'binary' / 'visualizations'}")
        print(f"   3-Class:   {self.dirs['models'] / '3class' / 'visualizations'}")
        
        print("\n" + "=" * 60)


def main():
    """메인 함수"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Home Safe Solution - Full Pipeline')
    parser.add_argument('--skip-skeleton', action='store_true',
                      help='Skeleton 추출 단계 스킵 (이미 완료된 경우)')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    master = PipelineMaster()
    master.run_pipeline(skip_skeleton=args.skip_skeleton)


if __name__ == "__main__":
    main()