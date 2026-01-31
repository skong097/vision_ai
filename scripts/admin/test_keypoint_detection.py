"""
Keypoint 감지 디버깅 - 문제 진단용
"""

import cv2
from ultralytics import YOLO

# YOLO Pose 모델 로드
model = YOLO('yolov8n-pose.pt')

# 웹캠 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=" * 60)
print("🔍 Keypoint 감지 테스트")
print("=" * 60)
print("💡 'q' 키로 종료")
print("=" * 60)

frame_count = 0
detected_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO Pose 추론
    results = model(frame, verbose=False)
    
    frame_count += 1
    
    # Keypoint 감지 여부 확인
    if len(results) > 0 and results[0].keypoints is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()
        
        if len(keypoints) > 0:
            detected_count += 1
            kp = keypoints[0]
            
            # Confidence 높은 keypoint 개수 세기
            valid_kp_count = sum(1 for i in range(17) if kp[i, 2] > 0.3)
            
            # 화면에 표시
            cv2.putText(frame, f"✅ Keypoints Detected: {valid_kp_count}/17", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Keypoint 그리기
            for i in range(17):
                if kp[i, 2] > 0.3:
                    x, y = int(kp[i, 0]), int(kp[i, 1])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # 통계 출력 (10프레임마다)
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}: ✅ {valid_kp_count}/17 keypoints (avg conf: {kp[:, 2].mean():.2f})")
        else:
            cv2.putText(frame, "❌ NO KEYPOINTS DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}: ❌ NO keypoints detected")
    else:
        cv2.putText(frame, "❌ NO PERSON DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: ❌ NO person detected")
    
    # 감지 비율 표시
    detection_rate = (detected_count / frame_count) * 100 if frame_count > 0 else 0
    cv2.putText(frame, f"Detection Rate: {detection_rate:.1f}%", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Keypoint Detection Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("📊 최종 통계")
print("=" * 60)
print(f"총 프레임: {frame_count}")
print(f"감지 성공: {detected_count} ({detection_rate:.1f}%)")
print(f"감지 실패: {frame_count - detected_count} ({100-detection_rate:.1f}%)")

if detection_rate < 50:
    print("\n🚨 문제 진단:")
    print("   Keypoint 감지율이 50% 미만입니다!")
    print("\n💡 해결 방법:")
    print("   1. 밝은 곳으로 이동")
    print("   2. 카메라와의 거리 조정 (1~3m)")
    print("   3. 전신이 화면에 들어오도록 조정")
    print("   4. 단순한 배경 선택")
elif detection_rate < 80:
    print("\n⚠️  경고:")
    print("   Keypoint 감지가 불안정합니다.")
    print("   조명과 거리를 조정해보세요.")
else:
    print("\n✅ Keypoint 감지 정상!")
