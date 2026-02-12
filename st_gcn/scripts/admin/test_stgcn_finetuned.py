#!/usr/bin/env python3
"""
ST-GCN Fine-tuned 모델 실시간 테스트
YOLO Pose + ST-GCN Fine-tuned 통합 테스트
"""

import cv2
import numpy as np
import torch
import time
import argparse
from pathlib import Path

# YOLO
from ultralytics import YOLO

# ST-GCN Fine-tuned
from stgcn_inference_finetuned import STGCNInference, STGCNInferenceWithSmoothing


def draw_skeleton(frame, keypoints, color=(0, 255, 0), thickness=2):
    """스켈레톤 그리기"""
    # COCO 17 keypoints 연결
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 11), (6, 12), (11, 12),  # torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # legs
    ]
    
    # 키포인트 그리기
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    
    # 스켈레톤 연결 그리기
    for i, j in skeleton:
        if keypoints[i][2] > 0.5 and keypoints[j][2] > 0.5:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(frame, pt1, pt2, color, thickness)
    
    return frame


def normalize_keypoints(keypoints, frame_width, frame_height):
    """키포인트 정규화 (0-1 범위)"""
    normalized = keypoints.copy()
    normalized[:, 0] = keypoints[:, 0] / frame_width
    normalized[:, 1] = keypoints[:, 1] / frame_height
    # confidence는 그대로
    return normalized


def test_video(video_path: str, 
               stgcn_model_path: str,
               yolo_model_path: str = '/home/gjkong/dev_ws/yolo/myproj/yolo11s-pose.pt',
               output_path: str = None,
               show: bool = True):
    """
    비디오 테스트
    
    Args:
        video_path: 입력 비디오 경로
        stgcn_model_path: ST-GCN 모델 경로
        yolo_model_path: YOLO Pose 모델 경로
        output_path: 출력 비디오 경로 (선택)
        show: 화면 표시 여부
    """
    print("\n" + "="*60)
    print("ST-GCN Fine-tuned Real-time Inference Test")
    print("="*60)
    
    # 1. 모델 로드
    print("\n[1/3] Loading models...")
    
    # YOLO Pose
    yolo = YOLO(yolo_model_path)
    print(f"✅ YOLO Pose loaded: {yolo_model_path}")
    
    # ST-GCN Fine-tuned
    try:
        stgcn = STGCNInferenceWithSmoothing(model_path=stgcn_model_path, smoothing_window=5)
        print(f"✅ ST-GCN Fine-tuned loaded: {stgcn_model_path}")
    except Exception as e:
        print(f"❌ ST-GCN load failed: {e}")
        return
    
    # 2. 비디오 열기
    print("\n[2/3] Opening video source...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video opened: {video_path}")
    print(f"   Resolution: {frame_width}x{frame_height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    
    # 출력 비디오
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 3. 추론
    print("\n[3/3] Running inference...")
    
    frame_count = 0
    normal_count = 0
    fall_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # YOLO Pose 추론
        results = yolo(frame, verbose=False)
        
        # 결과 처리
        label = "Buffering..."
        confidence = 0.0
        color = (128, 128, 128)  # 회색
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                # 첫 번째 사람의 키포인트
                keypoints = result.keypoints.data[0].cpu().numpy()  # (17, 3)
                
                # 정규화
                keypoints_norm = normalize_keypoints(keypoints, frame_width, frame_height)
                
                # ST-GCN 업데이트
                pred = stgcn.update(keypoints_norm)
                
                if pred:
                    label, confidence = pred
                    if label == 'Fall':
                        fall_count += 1
                        color = (0, 0, 255)  # 빨강
                    else:
                        normal_count += 1
                        color = (0, 255, 0)  # 초록
                else:
                    # 버퍼링 중
                    current, required, status = stgcn.get_buffer_status()
                    label = f"Buffering {current}/{required}"
                
                # 스켈레톤 그리기
                frame = draw_skeleton(frame, keypoints, color)
        
        # 정보 표시
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {label}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if confidence > 0:
            cv2.putText(frame, f"Confidence: {confidence:.1%}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Normal: {normal_count} | Fall: {fall_count}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 진행 상황 출력
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            print(f"\r   Frame {frame_count}/{total_frames} | "
                  f"FPS: {current_fps:.1f} | "
                  f"Normal: {normal_count} | Fall: {fall_count}", end='')
        
        # 화면 표시
        if show:
            cv2.imshow('ST-GCN Fine-tuned Test', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)  # 일시정지
        
        # 출력 비디오
        if writer:
            writer.write(frame)
    
    # 정리
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # 결과 요약
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    
    print("\n\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Detections:")
    print(f"  - Normal: {normal_count}")
    print(f"  - Fall: {fall_count}")
    if normal_count + fall_count > 0:
        fall_ratio = fall_count / (normal_count + fall_count) * 100
        print(f"  - Fall ratio: {fall_ratio:.2f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='ST-GCN Fine-tuned Test')
    parser.add_argument('--video', type=str, 
                        default='/home/gjkong/dev_ws/st_gcn/data_integrated/fall-01-cam0.mp4',
                        help='Input video path')
    parser.add_argument('--model', type=str,
                        default='/home/gjkong/dev_ws/st_gcn/checkpoints_finetuned/best_model_finetuned.pth',
                        help='ST-GCN model path')
    parser.add_argument('--yolo', type=str,
                        default='/home/gjkong/dev_ws/yolo/myproj/yolo11s-pose.pt',
                        help='YOLO Pose model path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (optional)')
    parser.add_argument('--no-show', action='store_true',
                        help='Disable display')
    
    args = parser.parse_args()
    
    test_video(
        video_path=args.video,
        stgcn_model_path=args.model,
        yolo_model_path=args.yolo,
        output_path=args.output,
        show=not args.no_show
    )


if __name__ == '__main__':
    main()
