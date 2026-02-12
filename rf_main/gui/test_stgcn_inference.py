#!/usr/bin/env python3
"""
ST-GCN 추론 테스트 스크립트
- 동영상 파일에서 YOLO Pose로 키포인트 추출
- ST-GCN으로 낙상 감지 추론
- 결과 시각화 (OpenCV)

사용법:
    python test_stgcn_inference.py --video /path/to/video.mp4
    python test_stgcn_inference.py --video /path/to/video.mp4 --no-display
    python test_stgcn_inference.py --camera 0
"""

import sys
import os
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# YOLO 및 ST-GCN 모듈
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[Warning] ultralytics not installed. Run: pip install ultralytics")
    YOLO_AVAILABLE = False

from stgcn_inference import STGCNInference, STGCNInferenceWithSmoothing


# ========== 설정 ==========

# 기본 경로 (환경에 맞게 수정)
DEFAULT_MODEL_PATH = '/home/gjkong/dev_ws/st_gcn/checkpoints/best_model_binary.pth'
DEFAULT_YOLO_PATH = '/home/gjkong/dev_ws/yolo/myproj/yolo11s-pose.pt'
DEFAULT_VIDEO_PATH = '/home/gjkong/dev_ws/st_gcn/data_integrated/fall-01-cam0.mp4'

# 시각화 색상
COLORS = {
    'Normal': (0, 255, 0),    # Green
    'Fall': (0, 0, 255),      # Red
    'Buffering': (255, 165, 0), # Orange
}

# 스켈레톤 연결 (COCO 17 keypoints)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 상체
    (5, 11), (6, 12), (11, 12),  # 몸통
    (11, 13), (13, 15), (12, 14), (14, 16),  # 다리
]


# ========== 헬퍼 함수 ==========

def draw_skeleton(frame, keypoints, color=(0, 255, 0), thickness=2):
    """스켈레톤 그리기"""
    if keypoints is None:
        return frame
    
    # 키포인트 그리기
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    
    # 연결선 그리기
    for (i, j) in SKELETON_CONNECTIONS:
        if keypoints[i][2] > 0.5 and keypoints[j][2] > 0.5:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(frame, pt1, pt2, color, thickness)
    
    return frame


def draw_status(frame, label, confidence, buffer_status, fps):
    """상태 정보 표시"""
    h, w = frame.shape[:2]
    
    # 배경 박스
    cv2.rectangle(frame, (10, 10), (350, 130), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (350, 130), (255, 255, 255), 2)
    
    # 모델 이름
    cv2.putText(frame, "ST-GCN Fall Detection", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 버퍼 상태
    cv2.putText(frame, f"Buffer: {buffer_status}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # 감지 결과
    if label:
        color = COLORS.get(label, (255, 255, 255))
        cv2.putText(frame, f"Status: {label}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        cv2.putText(frame, "Status: Buffering...", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['Buffering'], 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return frame


def draw_progress_bar(frame, progress, y_pos=150):
    """버퍼 진행률 바"""
    h, w = frame.shape[:2]
    bar_width = 330
    bar_height = 20
    x_start = 10
    
    # 배경
    cv2.rectangle(frame, (x_start, y_pos), 
                  (x_start + bar_width, y_pos + bar_height), (50, 50, 50), -1)
    
    # 진행률
    filled_width = int(bar_width * progress)
    color = (0, 255, 0) if progress >= 1.0 else (255, 165, 0)
    cv2.rectangle(frame, (x_start, y_pos), 
                  (x_start + filled_width, y_pos + bar_height), color, -1)
    
    # 테두리
    cv2.rectangle(frame, (x_start, y_pos), 
                  (x_start + bar_width, y_pos + bar_height), (255, 255, 255), 1)
    
    # 텍스트
    text = f"{int(progress * 100)}%" if progress < 1.0 else "Ready"
    cv2.putText(frame, text, (x_start + bar_width // 2 - 20, y_pos + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


# ========== 메인 테스트 함수 ==========

def run_test(video_source, model_path, yolo_path, display=True, save_output=False):
    """
    ST-GCN 추론 테스트 실행
    
    Args:
        video_source: 동영상 파일 경로 또는 카메라 인덱스
        model_path: ST-GCN 모델 경로
        yolo_path: YOLO Pose 모델 경로
        display: 화면 표시 여부
        save_output: 결과 동영상 저장 여부
    """
    print("=" * 60)
    print("ST-GCN Real-time Inference Test")
    print("=" * 60)
    
    # ===== 1. 모델 로드 =====
    print("\n[1/3] Loading models...")
    
    # YOLO Pose
    if not YOLO_AVAILABLE:
        print("❌ YOLO not available")
        return
    
    if not os.path.exists(yolo_path):
        print(f"❌ YOLO model not found: {yolo_path}")
        # 기본 YOLO pose 모델 사용
        print("   Trying default yolo11s-pose...")
        yolo_path = 'yolo11s-pose.pt'
    
    try:
        yolo_model = YOLO(yolo_path)
        print(f"✅ YOLO Pose loaded: {yolo_path}")
    except Exception as e:
        print(f"❌ YOLO load failed: {e}")
        return
    
    # ST-GCN
    if not os.path.exists(model_path):
        print(f"❌ ST-GCN model not found: {model_path}")
        return
    
    try:
        stgcn_model = STGCNInferenceWithSmoothing(
            model_path=model_path,
            smoothing_window=3
        )
        print(f"✅ ST-GCN loaded: {model_path}")
    except Exception as e:
        print(f"❌ ST-GCN load failed: {e}")
        return
    
    # ===== 2. 비디오 소스 열기 =====
    print("\n[2/3] Opening video source...")
    
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source)
        source_name = f"Camera {video_source}"
    else:
        if not os.path.exists(video_source):
            print(f"❌ Video file not found: {video_source}")
            return
        cap = cv2.VideoCapture(video_source)
        source_name = os.path.basename(video_source)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video source: {video_source}")
        return
    
    # 비디오 정보
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video opened: {source_name}")
    print(f"   Resolution: {frame_width}x{frame_height}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Total frames: {total_frames}")
    
    # ST-GCN 프레임 크기 설정
    stgcn_model.set_frame_size(frame_width, frame_height)
    
    # 출력 비디오 설정
    out_writer = None
    if save_output:
        output_path = f"stgcn_output_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                      (frame_width, frame_height))
        print(f"   Output: {output_path}")
    
    # ===== 3. 추론 루프 =====
    print("\n[3/3] Running inference...")
    print("      Press 'q' to quit, 'r' to reset buffer")
    print("-" * 60)
    
    frame_count = 0
    detection_count = {'Normal': 0, 'Fall': 0}
    start_time = time.time()
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n[Info] Video ended")
            break
        
        frame_count += 1
        
        # YOLO Pose 추론
        results = yolo_model(frame, verbose=False)
        
        keypoints = None
        if len(results) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints
            if kpts.data.shape[0] > 0:
                # 첫 번째 사람의 키포인트
                keypoints = kpts.data[0].cpu().numpy()  # (17, 3)
        
        # ST-GCN 추론
        result = stgcn_model.update(keypoints if keypoints is not None 
                                    else np.zeros((17, 3)))
        
        # 결과 처리
        label, confidence = None, 0.0
        if result is not None:
            label, confidence = result
            detection_count[label] += 1
        
        # 버퍼 상태
        current, required, buffer_status = stgcn_model.get_buffer_status()
        progress = stgcn_model.get_buffer_progress()
        
        # FPS 계산
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start = time.time()
        
        # 시각화
        if display or save_output:
            # 스켈레톤 그리기
            skeleton_color = COLORS.get(label, COLORS['Buffering']) if label else COLORS['Buffering']
            frame = draw_skeleton(frame, keypoints, color=skeleton_color)
            
            # 상태 정보
            frame = draw_status(frame, label, confidence, buffer_status, current_fps)
            
            # 진행률 바
            frame = draw_progress_bar(frame, progress)
        
        # 화면 표시
        if display:
            cv2.imshow('ST-GCN Fall Detection Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[Info] User quit")
                break
            elif key == ord('r'):
                stgcn_model.reset_buffer()
                print("[Info] Buffer reset")
        
        # 저장
        if out_writer:
            out_writer.write(frame)
        
        # 진행 상황 출력 (100프레임마다)
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"   Frame {frame_count}/{total_frames} | "
                  f"FPS: {current_fps:.1f} | "
                  f"Normal: {detection_count['Normal']} | "
                  f"Fall: {detection_count['Fall']}")
    
    # ===== 정리 =====
    cap.release()
    if out_writer:
        out_writer.release()
    if display:
        cv2.destroyAllWindows()
    
    # 결과 요약
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average FPS: {frame_count / elapsed:.2f}")
    print(f"Detections:")
    print(f"  - Normal: {detection_count['Normal']}")
    print(f"  - Fall: {detection_count['Fall']}")
    
    if detection_count['Fall'] > 0:
        fall_ratio = detection_count['Fall'] / (detection_count['Normal'] + detection_count['Fall'])
        print(f"  - Fall ratio: {fall_ratio:.2%}")
    
    print("=" * 60)


# ========== 메인 ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ST-GCN Inference Test')
    
    # 입력 소스
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--video', type=str, default=DEFAULT_VIDEO_PATH,
                       help='Video file path')
    group.add_argument('--camera', type=int, default=None,
                       help='Camera index (e.g., 0)')
    
    # 모델 경로
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='ST-GCN model path')
    parser.add_argument('--yolo', type=str, default=DEFAULT_YOLO_PATH,
                        help='YOLO Pose model path')
    
    # 옵션
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display (headless mode)')
    parser.add_argument('--save', action='store_true',
                        help='Save output video')
    
    args = parser.parse_args()
    
    # 비디오 소스 결정
    video_source = args.camera if args.camera is not None else args.video
    
    # 테스트 실행
    run_test(
        video_source=video_source,
        model_path=args.model,
        yolo_path=args.yolo,
        display=not args.no_display,
        save_output=args.save
    )
