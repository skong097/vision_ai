    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ⭐ ST-GCN 관련 메소드 ⭐
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def init_stgcn_model(self):
        """ST-GCN 모델 초기화"""
        if not STGCN_AVAILABLE:
            self.safe_add_log("[ERROR] ST-GCN 모듈을 찾을 수 없습니다.")
            self.model_type = 'random_forest'
            return False
        
        try:
            self.stgcn_model = STGCNInference(
                model_path='/home/gjkong/dev_ws/st_gcn/checkpoints/best_model_binary.pth'
            )
            
            # 프레임 크기 설정
            if self.cap:
                frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.stgcn_model.set_frame_size(frame_width, frame_height)
            
            self.keypoints_buffer = []
            self.stgcn_ready = False
            self.safe_add_log(f"[ST-GCN] 모델 로드 완료 (버퍼: {self.stgcn_buffer_size}프레임)")
            return True
            
        except Exception as e:
            self.safe_add_log(f"[ERROR] ST-GCN 모델 로드 실패: {e}")
            self.model_type = 'random_forest'
            return False
    
    def process_stgcn_inference(self, keypoints, frame):
        """
        ST-GCN 모델로 낙상 감지 추론
        
        Args:
            keypoints: 필터링된 키포인트 (17, 3)
            frame: 현재 프레임 (시각화용)
        """
        if self.stgcn_model is None:
            return
        
        # 버퍼에 키포인트 추가
        self.keypoints_buffer.append(keypoints.copy())
        
        # 버퍼 크기 유지 (슬라이딩 윈도우)
        if len(self.keypoints_buffer) > self.stgcn_buffer_size:
            self.keypoints_buffer.pop(0)
        
        # 버퍼 진행률
        buffer_progress = len(self.keypoints_buffer) / self.stgcn_buffer_size
        buffer_percent = int(buffer_progress * 100)
        
        # 추론 수행
        if len(self.keypoints_buffer) >= self.stgcn_buffer_size:
            self.stgcn_ready = True
            
            try:
                label, confidence = self.stgcn_model.predict(self.keypoints_buffer)
                
                # 결과 처리
                if label == 'Fall':
                    # 낙상 감지
                    self.fall_status = '낙상'
                    self.fall_confidence = confidence
                    
                    # 로그 (30프레임마다)
                    if self.frame_count % 30 == 0:
                        self.safe_add_log(f"[ST-GCN] 🚨 낙상 감지! (신뢰도: {confidence:.1%})")
                    
                    # 정확도 트래커 업데이트
                    self.accuracy_tracker.add_prediction(
                        predicted='낙상',
                        ground_truth=self.ground_truth
                    )
                    
                    # DB 저장 (10프레임마다)
                    if self.frame_count % 10 == 0:
                        self.save_event_to_db('낙상', confidence)
                    
                else:
                    # 정상
                    self.fall_status = '정상'
                    self.fall_confidence = confidence
                    
                    # 정확도 트래커 업데이트
                    self.accuracy_tracker.add_prediction(
                        predicted='정상',
                        ground_truth=self.ground_truth
                    )
                
                # 상태 라벨 업데이트
                self.update_stgcn_status_label(label, confidence, buffer_percent)
                
            except Exception as e:
                if self.frame_count % 60 == 0:
                    self.safe_add_log(f"[ST-GCN] 추론 오류: {e}")
        else:
            # 버퍼링 중
            self.stgcn_ready = False
            self.fall_status = '버퍼링'
            self.update_stgcn_status_label('버퍼링', 0.0, buffer_percent)
    
    def update_stgcn_status_label(self, status: str, confidence: float, buffer_percent: int):
        """ST-GCN 상태 표시 업데이트"""
        if status == '낙상' or status == 'Fall':
            color = '#f44336'  # Red
            status_text = f"🚨 낙상 감지 ({confidence:.1%})"
        elif status == '정상' or status == 'Normal':
            color = '#4caf50'  # Green
            status_text = f"✅ 정상 ({confidence:.1%})"
        else:  # 버퍼링
            color = '#ff9800'  # Orange
            status_text = f"⏳ ST-GCN 버퍼링... {buffer_percent}%"
        
        # status_label이 있으면 업데이트
        if hasattr(self, 'status_label'):
            self.status_label.setText(status_text)
            self.status_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px;")
    
    def reset_stgcn_buffer(self):
        """ST-GCN 버퍼 초기화"""
        self.keypoints_buffer = []
        self.stgcn_ready = False
        if self.stgcn_model:
            self.stgcn_model.reset_buffer()
        self.safe_add_log("[ST-GCN] 버퍼 초기화됨")
