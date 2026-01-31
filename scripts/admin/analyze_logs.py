"""
낙상 감지 로그 분석 도구
Author: Home Safe Solution Team
Date: 2026-01-28
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import Counter


class LogAnalyzer:
    """로그 파일 분석 및 이슈 진단"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.issues = []
        self.warnings = []
        self.stats = {}
    
    def analyze_log_file(self, log_file):
        """로그 파일 분석"""
        print(f"\n{'='*60}")
        print(f"📄 로그 분석: {log_file.name}")
        print(f"{'='*60}\n")
        
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 통계
        total_lines = len(lines)
        error_count = 0
        warning_count = 0
        info_count = 0
        
        # 패턴 분석
        errors = []
        warnings = []
        predictions = []
        fall_events = []
        
        for line in lines:
            if '[ERROR]' in line:
                error_count += 1
                errors.append(line.strip())
            elif '[WARNING]' in line:
                warning_count += 1
                warnings.append(line.strip())
            elif '[INFO]' in line:
                info_count += 1
                
                # 예측 정보 추출
                if 'Frame' in line and '확률' in line:
                    predictions.append(line.strip())
                
                # 낙상 이벤트 추출
                if '낙상 감지' in line:
                    fall_events.append(line.strip())
        
        # 결과 출력
        print(f"📊 기본 통계")
        print(f"  총 로그 라인: {total_lines}")
        print(f"  INFO:    {info_count} ({info_count/total_lines*100:.1f}%)")
        print(f"  WARNING: {warning_count} ({warning_count/total_lines*100:.1f}%)")
        print(f"  ERROR:   {error_count} ({error_count/total_lines*100:.1f}%)")
        
        # 에러 분석
        if errors:
            print(f"\n❌ 에러 발견: {len(errors)}개")
            for i, error in enumerate(errors[:5], 1):
                print(f"  {i}. {error}")
            if len(errors) > 5:
                print(f"  ... 외 {len(errors)-5}개")
        else:
            print(f"\n✅ 에러 없음!")
        
        # 경고 분석
        if warnings:
            print(f"\n⚠️  경고 발견: {len(warnings)}개")
            for i, warning in enumerate(warnings[:5], 1):
                print(f"  {i}. {warning}")
            if len(warnings) > 5:
                print(f"  ... 외 {len(warnings)-5}개")
        else:
            print(f"\n✅ 경고 없음!")
        
        # 예측 분석
        if predictions:
            print(f"\n🎯 예측 로그: {len(predictions)}개")
            
            # 예측 분포 분석
            prediction_types = []
            for pred in predictions:
                if 'Normal' in pred:
                    prediction_types.append('Normal')
                elif 'Falling' in pred:
                    prediction_types.append('Falling')
                elif 'Fallen' in pred:
                    prediction_types.append('Fallen')
                elif 'Fall' in pred:
                    prediction_types.append('Fall')
            
            counter = Counter(prediction_types)
            print(f"  예측 분포:")
            for pred_type, count in counter.items():
                print(f"    {pred_type}: {count}회")
            
            # 최근 5개 예측 표시
            print(f"\n  최근 예측:")
            for pred in predictions[-5:]:
                print(f"    {pred}")
        
        # 낙상 이벤트 분석
        if fall_events:
            print(f"\n🚨 낙상 이벤트: {len(fall_events)}개")
            for i, event in enumerate(fall_events[:3], 1):
                print(f"  {i}. {event}")
            if len(fall_events) > 3:
                print(f"  ... 외 {len(fall_events)-3}개")
        else:
            print(f"\n✅ 낙상 이벤트 없음")
        
        # 이슈 진단
        self.diagnose_issues(error_count, warning_count, errors, warnings)
        
        return {
            'total_lines': total_lines,
            'errors': error_count,
            'warnings': warning_count,
            'predictions': len(predictions),
            'fall_events': len(fall_events)
        }
    
    def analyze_statistics(self, stats_file):
        """통계 JSON 파일 분석"""
        print(f"\n{'='*60}")
        print(f"📊 통계 분석: {stats_file.name}")
        print(f"{'='*60}\n")
        
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        print(f"총 프레임: {stats['total_frames']}")
        print(f"평균 FPS: {stats['avg_fps']:.1f}")
        
        print(f"\n예측 분포:")
        total_predictions = sum(stats['prediction_stats'].values())
        for label, count in stats['prediction_stats'].items():
            percentage = count / total_predictions * 100 if total_predictions > 0 else 0
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        fall_events = stats.get('fall_events', [])
        print(f"\n낙상 이벤트: {len(fall_events)}개")
        
        if fall_events:
            print(f"\n상세 이벤트:")
            for i, event in enumerate(fall_events[:3], 1):
                print(f"  {i}. Frame {event['frame']}: {event['prediction']} "
                      f"(확률: {event['probability']*100:.1f}%)")
                print(f"     Hip Height: {event['hip_height']:.1f}, "
                      f"Spine Angle: {event['spine_angle']:.1f}°")
            if len(fall_events) > 3:
                print(f"  ... 외 {len(fall_events)-3}개")
    
    def diagnose_issues(self, error_count, warning_count, errors, warnings):
        """이슈 진단"""
        print(f"\n{'='*60}")
        print(f"🔍 이슈 진단")
        print(f"{'='*60}\n")
        
        issues_found = []
        
        # 1. 에러 체크
        if error_count > 0:
            issues_found.append(f"❌ 에러 발견: {error_count}개")
            
            # 구체적 에러 분석
            for error in errors:
                if 'feature names' in error.lower():
                    issues_found.append("  - Feature 순서 불일치 문제")
                elif 'model' in error.lower():
                    issues_found.append("  - 모델 로드 문제")
                elif 'keypoint' in error.lower():
                    issues_found.append("  - Keypoint 감지 문제")
                elif 'camera' in error.lower() or 'webcam' in error.lower():
                    issues_found.append("  - 웹캠 연결 문제")
        
        # 2. 경고 체크
        if warning_count > 10:
            issues_found.append(f"⚠️  과도한 경고: {warning_count}개")
            
            for warning in warnings:
                if 'keypoint' in warning.lower():
                    issues_found.append("  - Keypoint 감지 불안정")
                elif 'frame' in warning.lower():
                    issues_found.append("  - 프레임 읽기 불안정")
        
        # 결과 출력
        if issues_found:
            print("🔴 발견된 이슈:")
            for issue in issues_found:
                print(f"  {issue}")
            
            print(f"\n💡 권장 조치:")
            if any('feature' in issue.lower() for issue in issues_found):
                print("  1. feature_columns.txt 파일이 존재하는지 확인")
                print("  2. 모델을 재학습하여 feature 순서 일치시키기")
            
            if any('keypoint' in issue.lower() for issue in issues_found):
                print("  1. 조명 확인 (밝은 곳에서 테스트)")
                print("  2. 카메라와의 거리 조정 (1~3m 권장)")
                print("  3. 전신이 화면에 들어오도록 조정")
            
            if any('webcam' in issue.lower() or 'camera' in issue.lower() for issue in issues_found):
                print("  1. 웹캠 연결 확인")
                print("  2. 다른 카메라 ID 시도 (--camera 1)")
                print("  3. 권한 확인")
        else:
            print("✅ 이슈 없음! 시스템이 정상 작동 중입니다.")
    
    def analyze_all(self):
        """모든 로그 파일 분석"""
        if not self.log_dir.exists():
            print(f"❌ 로그 디렉토리를 찾을 수 없습니다: {self.log_dir}")
            return
        
        # 로그 파일 찾기
        log_files = sorted(self.log_dir.glob("fall_detection_*.log"))
        stats_files = sorted(self.log_dir.glob("statistics_*.json"))
        
        if not log_files and not stats_files:
            print(f"❌ 로그 파일을 찾을 수 없습니다: {self.log_dir}")
            return
        
        print(f"\n{'='*60}")
        print(f"📁 로그 분석 시작")
        print(f"{'='*60}")
        print(f"로그 디렉토리: {self.log_dir}")
        print(f"로그 파일: {len(log_files)}개")
        print(f"통계 파일: {len(stats_files)}개")
        
        # 가장 최근 로그 분석
        if log_files:
            latest_log = log_files[-1]
            self.analyze_log_file(latest_log)
        
        # 가장 최근 통계 분석
        if stats_files:
            latest_stats = stats_files[-1]
            self.analyze_statistics(latest_stats)
        
        print(f"\n{'='*60}")
        print(f"✅ 분석 완료!")
        print(f"{'='*60}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='낙상 감지 로그 분석')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='로그 디렉토리 경로')
    parser.add_argument('--file', type=str, default=None,
                       help='특정 로그 파일 분석')
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(log_dir=args.log_dir)
    
    if args.file:
        # 특정 파일 분석
        log_file = Path(args.file)
        if log_file.exists():
            analyzer.analyze_log_file(log_file)
        else:
            print(f"❌ 파일을 찾을 수 없습니다: {log_file}")
    else:
        # 모든 로그 분석
        analyzer.analyze_all()


if __name__ == "__main__":
    main()
