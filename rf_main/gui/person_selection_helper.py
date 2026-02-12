"""
여러 사람 중 모니터링 대상자 선택 헬퍼 함수
"""

def select_target_person(results, method='largest'):
    """
    여러 사람 중 모니터링 대상자 선택
    
    Args:
        results: YOLO 결과 객체
        method: 선택 방법
            - 'largest': 가장 큰 Bounding Box (기본, 추천)
            - 'center': 화면 중앙에 가장 가까운 사람
            - 'combined': 크기 + 중앙 거리 조합
            - 'confidence': 신뢰도 순
    
    Returns:
        int: 선택된 사람의 인덱스 (없으면 None)
    """
    if len(results) == 0:
        return None
    
    # Keypoints와 Boxes 가져오기
    keypoints = results[0].keypoints
    boxes = results[0].boxes
    
    if keypoints is None or boxes is None:
        return None
    
    keypoints_data = keypoints.data.cpu().numpy()
    boxes_data = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    
    if len(keypoints_data) == 0 or len(boxes_data) == 0:
        return None
    
    num_people = len(keypoints_data)
    
    # 한 명만 있으면 바로 반환
    if num_people == 1:
        return 0
    
    # ===== 방법 1: 가장 큰 Bounding Box (면적) =====
    if method == 'largest':
        areas = []
        for box in boxes_data:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height
            areas.append(area)
        
        # 가장 큰 면적의 인덱스 반환
        return areas.index(max(areas))
    
    # ===== 방법 2: 화면 중앙 거리 =====
    elif method == 'center':
        # 프레임 중앙 (가정: 640x480)
        frame_center_x = 640 / 2
        frame_center_y = 480 / 2
        
        distances = []
        for box in boxes_data:
            x1, y1, x2, y2 = box
            # Box 중심점
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            
            # 유클리드 거리
            distance = ((box_center_x - frame_center_x)**2 + 
                       (box_center_y - frame_center_y)**2)**0.5
            distances.append(distance)
        
        # 가장 가까운 사람의 인덱스 반환
        return distances.index(min(distances))
    
    # ===== 방법 3: 크기 + 중앙 거리 조합 (가중치) =====
    elif method == 'combined':
        frame_center_x = 640 / 2
        frame_center_y = 480 / 2
        
        scores = []
        areas = []
        distances = []
        
        # 1단계: 면적과 거리 계산
        for box in boxes_data:
            x1, y1, x2, y2 = box
            
            # 면적
            width = x2 - x1
            height = y2 - y1
            area = width * height
            areas.append(area)
            
            # 중앙 거리
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            distance = ((box_center_x - frame_center_x)**2 + 
                       (box_center_y - frame_center_y)**2)**0.5
            distances.append(distance)
        
        # 2단계: 정규화 (0-1)
        max_area = max(areas)
        max_distance = max(distances)
        
        # 3단계: 가중치 계산
        for i in range(len(boxes_data)):
            area_normalized = areas[i] / max_area
            distance_normalized = distances[i] / max_distance
            
            # 큰 면적 = 좋음 (1에 가까울수록)
            # 작은 거리 = 좋음 (0에 가까울수록)
            # 가중치: 면적 60% + 중앙 거리 40%
            score = (area_normalized * 0.6) + ((1 - distance_normalized) * 0.4)
            scores.append(score)
        
        # 가장 높은 점수의 인덱스 반환
        return scores.index(max(scores))
    
    # ===== 방법 4: 신뢰도 순 =====
    elif method == 'confidence':
        confidences = boxes.conf.cpu().numpy()
        return confidences.argmax()
    
    # 기본: 첫 번째 사람
    return 0


# ===== 간단한 사용 예시 =====
"""
# YOLO 추론
results = self.yolo_model(frame, verbose=False)

# 모니터링 대상자 선택
target_idx = select_target_person(results, method='largest')

if target_idx is not None:
    keypoints = results[0].keypoints.data.cpu().numpy()
    target_keypoints = keypoints[target_idx]
    
    # 필터링 적용
    keypoints_filtered = self.keypoint_filter.apply(target_keypoints)
    
    # 낙상 감지 진행...
"""
