"""
실시간 낙상 감지 정확도 측정 시스템
monitoring_page.py에 통합

사용자가 실제 상태를 입력하면서 정확도를 측정
"""

import time
from datetime import datetime
from collections import deque
import json
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from path_config import PATHS


class AccuracyMonitor:
    """실시간 정확도 측정 및 기록"""
    
    def __init__(self, save_dir='./accuracy_logs'):
        """
        Args:
            save_dir: 정확도 로그 저장 디렉토리
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 혼동 행렬 (Confusion Matrix)
        self.confusion_matrix = {
            'Normal': {'Normal': 0, 'Falling': 0, 'Fallen': 0},
            'Falling': {'Normal': 0, 'Falling': 0, 'Fallen': 0},
            'Fallen': {'Normal': 0, 'Falling': 0, 'Fallen': 0}
        }
        
        # 현재 세션 데이터
        self.current_ground_truth = None  # 사용자가 지정한 실제 상태
        self.predictions_buffer = deque(maxlen=30)  # 최근 30개 예측
        
        # 통계
        self.total_samples = 0
        self.correct_predictions = 0
        
        # 시간 기록
        self.start_time = time.time()
        self.last_save_time = time.time()
        self.save_interval = 60  # 1분마다 저장
        
        # 세션 ID
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1분 단위 기록
        self.minute_records = []
        self.current_minute_data = {
            'total': 0,
            'correct': 0,
            'confusion': {
                'Normal': {'Normal': 0, 'Falling': 0, 'Fallen': 0},
                'Falling': {'Normal': 0, 'Falling': 0, 'Fallen': 0},
                'Fallen': {'Normal': 0, 'Falling': 0, 'Fallen': 0}
            }
        }
    
    def set_ground_truth(self, state):
        """
        실제 상태 설정
        
        Args:
            state: 'Normal', 'Falling', 'Fallen' 중 하나
        """
        if state not in ['Normal', 'Falling', 'Fallen']:
            return False
        
        self.current_ground_truth = state
        return True
    
    def record_prediction(self, predicted_state, confidence):
        """
        예측 결과 기록
        
        Args:
            predicted_state: 예측된 상태 ('Normal', 'Falling', 'Fallen')
            confidence: 신뢰도 (0-1)
        """
        # Ground truth가 설정되지 않았으면 기록 안함
        if self.current_ground_truth is None:
            return
        
        # 혼동 행렬 업데이트
        self.confusion_matrix[self.current_ground_truth][predicted_state] += 1
        self.current_minute_data['confusion'][self.current_ground_truth][predicted_state] += 1
        
        # 통계 업데이트
        self.total_samples += 1
        self.current_minute_data['total'] += 1
        
        if self.current_ground_truth == predicted_state:
            self.correct_predictions += 1
            self.current_minute_data['correct'] += 1
        
        # 버퍼에 추가
        self.predictions_buffer.append({
            'timestamp': time.time(),
            'ground_truth': self.current_ground_truth,
            'predicted': predicted_state,
            'confidence': confidence,
            'correct': self.current_ground_truth == predicted_state
        })
        
        # 1분마다 저장
        if time.time() - self.last_save_time >= self.save_interval:
            self.save_minute_record()
    
    def save_minute_record(self):
        """1분 단위 기록 저장"""
        if self.current_minute_data['total'] == 0:
            return
        
        # 현재 분 데이터
        minute_accuracy = (self.current_minute_data['correct'] / 
                          self.current_minute_data['total'] * 100)
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_minutes': int((time.time() - self.start_time) / 60),
            'samples': self.current_minute_data['total'],
            'correct': self.current_minute_data['correct'],
            'accuracy': round(minute_accuracy, 2),
            'confusion_matrix': self.current_minute_data['confusion']
        }
        
        self.minute_records.append(record)
        
        # 파일로 저장
        log_file = self.save_dir / f'accuracy_{self.session_id}.json'
        with open(log_file, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'total_samples': self.total_samples,
                'overall_accuracy': self.get_accuracy(),
                'minute_records': self.minute_records,
                'overall_confusion_matrix': self.confusion_matrix
            }, f, indent=2)
        
        print(f"\n[ACCURACY] 1분 기록 저장: {minute_accuracy:.1f}% ({self.current_minute_data['correct']}/{self.current_minute_data['total']})")
        
        # 현재 분 데이터 초기화
        self.current_minute_data = {
            'total': 0,
            'correct': 0,
            'confusion': {
                'Normal': {'Normal': 0, 'Falling': 0, 'Fallen': 0},
                'Falling': {'Normal': 0, 'Falling': 0, 'Fallen': 0},
                'Fallen': {'Normal': 0, 'Falling': 0, 'Fallen': 0}
            }
        }
        
        self.last_save_time = time.time()
    
    def get_accuracy(self):
        """전체 정확도 반환"""
        if self.total_samples == 0:
            return 0.0
        return round(self.correct_predictions / self.total_samples * 100, 2)
    
    def get_class_accuracy(self, class_name):
        """클래스별 정확도"""
        total = sum(self.confusion_matrix[class_name].values())
        if total == 0:
            return 0.0
        correct = self.confusion_matrix[class_name][class_name]
        return round(correct / total * 100, 2)
    
    def get_stats(self):
        """현재 통계 반환"""
        return {
            'total_samples': self.total_samples,
            'correct': self.correct_predictions,
            'overall_accuracy': self.get_accuracy(),
            'class_accuracy': {
                'Normal': self.get_class_accuracy('Normal'),
                'Falling': self.get_class_accuracy('Falling'),
                'Fallen': self.get_class_accuracy('Fallen')
            },
            'confusion_matrix': self.confusion_matrix,
            'elapsed_time': time.time() - self.start_time
        }
    
    def print_stats(self):
        """통계 출력"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 실시간 낙상 감지 정확도")
        print("="*60)
        print(f"⏰ 경과 시간: {stats['elapsed_time']/60:.1f}분")
        print(f"📈 총 샘플: {stats['total_samples']}개")
        print(f"✅ 정확한 예측: {stats['correct']}개")
        print(f"🎯 전체 정확도: {stats['overall_accuracy']:.1f}%")
        print()
        print("클래스별 정확도:")
        print(f"  - Normal:  {stats['class_accuracy']['Normal']:.1f}%")
        print(f"  - Falling: {stats['class_accuracy']['Falling']:.1f}%")
        print(f"  - Fallen:  {stats['class_accuracy']['Fallen']:.1f}%")
        print()
        print("혼동 행렬 (Confusion Matrix):")
        print("           Predicted")
        print("           Normal  Falling  Fallen")
        for true_label in ['Normal', 'Falling', 'Fallen']:
            print(f"Actual {true_label:7s}", end="")
            for pred_label in ['Normal', 'Falling', 'Fallen']:
                count = self.confusion_matrix[true_label][pred_label]
                print(f"{count:7d}", end=" ")
            print()
        print("="*60 + "\n")
    
    def finalize(self):
        """세션 종료 시 최종 저장"""
        # 마지막 분 데이터 저장
        if self.current_minute_data['total'] > 0:
            self.save_minute_record()
        
        # 최종 통계 출력
        print("\n" + "🏁 최종 정확도 보고서")
        self.print_stats()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# monitoring_page.py에 통합하는 방법
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
1. __init__() 메소드에 추가:

    # 정확도 모니터 초기화
    # log_dir = '/home/gjkong/dev_ws/yolo/myproj/accuracy_logs'
    log_dir = str(PATHS.ACCURACY_LOG_DIR)
    self.accuracy_monitor = AccuracyMonitor(save_dir=log_dir)
    
    # 현재 Ground Truth 상태
    self.current_state_label = None  # GUI 라벨


2. create_info_panel()에 Ground Truth 선택 UI 추가:

    # Ground Truth 설정 그룹
    gt_group = QGroupBox("Ground Truth (실제 상태)")
    gt_layout = QVBoxLayout()
    
    gt_label = QLabel("현재 실제 상태를 선택하세요:")
    gt_layout.addWidget(gt_label)
    
    # 라디오 버튼
    from PyQt6.QtWidgets import QRadioButton, QButtonGroup
    
    self.gt_button_group = QButtonGroup()
    
    self.gt_normal = QRadioButton("Normal (정상)")
    self.gt_falling = QRadioButton("Falling (낙상 중)")
    self.gt_fallen = QRadioButton("Fallen (낙상 완료)")
    
    self.gt_button_group.addButton(self.gt_normal, 0)
    self.gt_button_group.addButton(self.gt_falling, 1)
    self.gt_button_group.addButton(self.gt_fallen, 2)
    
    # 기본값: Normal
    self.gt_normal.setChecked(True)
    self.accuracy_monitor.set_ground_truth('Normal')
    
    # 변경 시 이벤트
    self.gt_normal.toggled.connect(lambda: self.on_gt_changed('Normal'))
    self.gt_falling.toggled.connect(lambda: self.on_gt_changed('Falling'))
    self.gt_fallen.toggled.connect(lambda: self.on_gt_changed('Fallen'))
    
    gt_layout.addWidget(self.gt_normal)
    gt_layout.addWidget(self.gt_falling)
    gt_layout.addWidget(self.gt_fallen)
    
    # 현재 상태 표시
    self.current_state_label = QLabel("현재: Normal")
    self.current_state_label.setStyleSheet("color: #27ae60; font-weight: bold;")
    gt_layout.addWidget(self.current_state_label)
    
    gt_group.setLayout(gt_layout)
    layout.addWidget(gt_group)
    
    # 정확도 표시 그룹
    accuracy_group = QGroupBox("실시간 정확도")
    accuracy_layout = QVBoxLayout()
    
    self.accuracy_label = QLabel("정확도: --")
    self.accuracy_label.setFont(QFont('맑은 고딕', 14, QFont.Weight.Bold))
    accuracy_layout.addWidget(self.accuracy_label)
    
    self.samples_label = QLabel("샘플: 0개")
    accuracy_layout.addWidget(self.samples_label)
    
    accuracy_group.setLayout(accuracy_layout)
    layout.addWidget(accuracy_group)


3. on_gt_changed() 메소드 추가:

    def on_gt_changed(self, state):
        '''Ground Truth 변경'''
        self.accuracy_monitor.set_ground_truth(state)
        self.current_state_label.setText(f"현재: {state}")
        
        # 색상 변경
        color_map = {
            'Normal': '#27ae60',
            'Falling': '#f39c12',
            'Fallen': '#e74c3c'
        }
        self.current_state_label.setStyleSheet(
            f"color: {color_map[state]}; font-weight: bold;"
        )
        
        self.add_log(f"[GT] Ground Truth 설정: {state}")


4. update_frame()에서 예측 기록:

    # 필터 적용 후
    filtered_prediction, filtered_proba, filter_msg = apply_sitting_filter(...)
    
    # ⭐ 정확도 모니터에 기록
    class_name = self.class_names[filtered_prediction]
    self.accuracy_monitor.record_prediction(
        predicted_state=class_name,
        confidence=filtered_proba[filtered_prediction]
    )
    
    # 정확도 UI 업데이트
    self.update_accuracy_display()


5. update_accuracy_display() 메소드 추가:

    def update_accuracy_display(self):
        '''정확도 표시 업데이트'''
        try:
            stats = self.accuracy_monitor.get_stats()
            
            # 정확도
            accuracy = stats['overall_accuracy']
            self.accuracy_label.setText(f"정확도: {accuracy:.1f}%")
            
            # 샘플 수
            self.samples_label.setText(
                f"샘플: {stats['correct']}/{stats['total_samples']}"
            )
            
            # 색상 (정확도에 따라)
            if accuracy >= 90:
                color = "#27ae60"  # 녹색
            elif accuracy >= 70:
                color = "#f39c12"  # 주황
            else:
                color = "#e74c3c"  # 빨강
            
            self.accuracy_label.setStyleSheet(
                f"color: {color}; font-weight: bold;"
            )
        except:
            pass


6. stop_monitoring()에 추가:

    def stop_monitoring(self):
        # ... 기존 코드 ...
        
        # ⭐ 최종 정확도 저장
        if hasattr(self, 'accuracy_monitor'):
            self.accuracy_monitor.finalize()


7. 통계 보기 버튼 추가 (선택사항):

    self.btn_stats = QPushButton('📊 Accuracy Stats')
    self.btn_stats.clicked.connect(self.show_accuracy_stats)
    
    def show_accuracy_stats(self):
        '''정확도 통계 표시'''
        stats = self.accuracy_monitor.get_stats()
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("정확도 통계")
        msg.setText(f"전체 정확도: {stats['overall_accuracy']:.1f}%")
        
        info = f'''
샘플 수: {stats['total_samples']}개
정확한 예측: {stats['correct']}개

클래스별 정확도:
• Normal:  {stats['class_accuracy']['Normal']:.1f}%
• Falling: {stats['class_accuracy']['Falling']:.1f}%
• Fallen:  {stats['class_accuracy']['Fallen']:.1f}%

경과 시간: {stats['elapsed_time']/60:.1f}분
        '''
        
        msg.setInformativeText(info)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 독립 실행 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # 테스트
    monitor = AccuracyMonitor(save_dir='./test_accuracy_logs')
    
    # Ground Truth 설정
    monitor.set_ground_truth('Normal')
    
    # 예측 시뮬레이션
    import random
    
    print("📊 정확도 모니터 테스트 시작...")
    print("30초 동안 예측을 시뮬레이션합니다.\n")
    
    for i in range(100):
        # 90% 정확도로 시뮬레이션
        if random.random() < 0.9:
            predicted = 'Normal'
        else:
            predicted = random.choice(['Falling', 'Fallen'])
        
        confidence = random.uniform(0.7, 0.99)
        
        monitor.record_prediction(predicted, confidence)
        
        if (i + 1) % 10 == 0:
            stats = monitor.get_stats()
            print(f"[{i+1}/100] 정확도: {stats['overall_accuracy']:.1f}%")
        
        time.sleep(0.3)  # 0.3초마다
    
    # 최종 통계
    monitor.finalize()