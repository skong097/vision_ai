-- ============================================================
-- Home Safe Solution - 데이터베이스 스키마
-- Author: Home Safe Solution Team
-- Date: 2026-01-28
-- ============================================================

-- 데이터베이스 생성
CREATE DATABASE IF NOT EXISTS home_safe CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE home_safe;

-- ============================================================
-- 1. 사용자 테이블
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,  -- bcrypt hash
    rtsp_url VARCHAR(500),  -- RTSP 접속 정보
    name VARCHAR(100) NOT NULL,
    gender ENUM('남성', '여성', '기타') NOT NULL,
    blood_type ENUM('A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'),
    address VARCHAR(500),
    birth_date DATE,
    emergency_contact VARCHAR(20),
    user_type ENUM('관리자', '일반유저') NOT NULL DEFAULT '일반유저',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_username (username),
    INDEX idx_user_type (user_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 기본 관리자 계정 생성 (비밀번호: admin123)
INSERT INTO users (username, password_hash, name, gender, user_type) 
VALUES ('admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYILNv.gLKO', '관리자', '남성', '관리자')
ON DUPLICATE KEY UPDATE user_id=user_id;

INSERT INTO users (username, password_hash, name, gender, user_type) 
VALUES ('homesafe', 'homesafe2026', '관리자', '남성', '관리자')
ON DUPLICATE KEY UPDATE user_id=user_id;

INSERT INTO users (username, password_hash, name, gender, user_type) 
VALUES ('gjkong', 'gjkong', '관리자', '남성', '관리자')
ON DUPLICATE KEY UPDATE user_id=user_id;


-- ============================================================
-- 2. 이벤트 타입 테이블
-- ============================================================
CREATE TABLE IF NOT EXISTS event_types (
    event_type_id INT AUTO_INCREMENT PRIMARY KEY,
    type_name VARCHAR(50) UNIQUE NOT NULL,  -- 낙상, 쓰러짐, 화재, 침수, 외부인침입, 안전영역이탈
    severity ENUM('정상', '주의', '경고', '위험') NOT NULL DEFAULT '주의',
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_type_name (type_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 기본 이벤트 타입 데이터
INSERT INTO event_types (type_name, severity, description) VALUES
('정상', '정상', '정상 상태'),
('낙상', '위험', '넘어지는 행동 감지'),
('쓰러짐', '위험', '바닥에 쓰러진 상태'),
('화재', '위험', '화재 감지'),
('침수', '경고', '침수 감지'),
('외부인침입', '경고', '승인되지 않은 사람 감지'),
('안전영역이탈', '주의', '안전 영역 이탈 감지')
ON DUPLICATE KEY UPDATE event_type_id=event_type_id;

-- ============================================================
-- 3. 이벤트 로그 테이블
-- ============================================================
CREATE TABLE IF NOT EXISTS event_logs (
    event_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    event_type_id INT NOT NULL,
    event_status ENUM('발생', '조치중', '완료') NOT NULL DEFAULT '발생',
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP NULL,
    duration_seconds INT,  -- 이벤트 지속 시간 (초)
    
    -- 낙상 감지 관련 정보
    confidence FLOAT,  -- 예측 확률
    hip_height FLOAT,
    spine_angle FLOAT,
    hip_velocity FLOAT,
    
    -- 영상 정보
    video_path VARCHAR(500),  -- 이벤트 동영상 경로
    thumbnail_path VARCHAR(500),  -- 썸네일 이미지 경로
    
    -- 조치 정보
    action_taken ENUM('없음', '1차_메시지발송', '2차_긴급호출') DEFAULT '없음',
    action_result TEXT,  -- 조치 결과 상세
    
    -- 메타데이터
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (event_type_id) REFERENCES event_types(event_type_id),
    INDEX idx_user_event (user_id, occurred_at),
    INDEX idx_event_type (event_type_id),
    INDEX idx_occurred_at (occurred_at),
    INDEX idx_status (event_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 4. 자동신고 로그 테이블
-- ============================================================
CREATE TABLE IF NOT EXISTS auto_report_logs (
    report_id INT AUTO_INCREMENT PRIMARY KEY,
    event_id INT NOT NULL,
    report_target ENUM('119', '112', '비상연락처') NOT NULL,
    report_type ENUM('1차_메시지', '2차_긴급호출') NOT NULL,
    
    -- 신고 내용
    report_content TEXT,  -- 신고 메시지 내용
    video_sent BOOLEAN DEFAULT FALSE,  -- 동영상 전송 여부
    
    -- 발송 정보
    recipient VARCHAR(100),  -- 수신자 (전화번호 또는 이메일)
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    delivery_status ENUM('대기', '발송중', '성공', '실패') DEFAULT '대기',
    delivery_result TEXT,  -- 발송 결과 상세
    
    FOREIGN KEY (event_id) REFERENCES event_logs(event_id) ON DELETE CASCADE,
    INDEX idx_event (event_id),
    INDEX idx_target (report_target),
    INDEX idx_sent_at (sent_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 5. 시스템 설정 테이블
-- ============================================================
CREATE TABLE IF NOT EXISTS system_settings (
    setting_id INT AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT,
    setting_type ENUM('string', 'int', 'float', 'bool', 'json') DEFAULT 'string',
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_key (setting_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 기본 설정값
INSERT INTO system_settings (setting_key, setting_value, setting_type, description) VALUES
('auto_report_enabled', 'true', 'bool', '자동 신고 활성화 여부'),
('first_action_delay', '30', 'int', '1차 조치 대기 시간 (초)'),
('second_action_delay', '180', 'int', '2차 조치 대기 시간 (초)'),
('video_before_seconds', '5', 'int', '이벤트 발생 전 녹화 시간 (초)'),
('video_after_seconds', '10', 'int', '이벤트 발생 후 녹화 시간 (초)'),
('recording_path', '/home/gjkong/dev_ws/yolo/myproj/recordings/', 'string', '녹화 파일 저장 경로'),
('event_video_path', '/home/gjkong/dev_ws/yolo/myproj/events/', 'string', '이벤트 동영상 저장 경로'),
('model_type', '3class', 'string', '사용할 모델 타입 (binary/3class)'),
('confidence_threshold', '0.7', 'float', '낙상 감지 신뢰도 임계값')
ON DUPLICATE KEY UPDATE setting_id=setting_id;

-- ============================================================
-- 6. 사용자 로그인 히스토리 테이블
-- ============================================================
CREATE TABLE IF NOT EXISTS login_history (
    login_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    logout_at TIMESTAMP NULL,
    ip_address VARCHAR(45),
    user_agent VARCHAR(500),
    login_status ENUM('성공', '실패') DEFAULT '성공',
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_user_login (user_id, login_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 뷰 생성: 이벤트 상세 정보
-- ============================================================
CREATE OR REPLACE VIEW v_event_details AS
SELECT 
    el.event_id,
    u.username,
    u.name AS user_name,
    et.type_name AS event_type,
    et.severity,
    el.event_status,
    el.occurred_at,
    el.resolved_at,
    el.duration_seconds,
    el.confidence,
    el.hip_height,
    el.spine_angle,
    el.video_path,
    el.action_taken,
    el.action_result,
    u.address,
    u.emergency_contact,
    u.gender,
    u.blood_type,
    TIMESTAMPDIFF(YEAR, u.birth_date, CURDATE()) AS age
FROM event_logs el
JOIN users u ON el.user_id = u.user_id
JOIN event_types et ON el.event_type_id = et.event_type_id;

-- ============================================================
-- 인덱스 최적화
-- ============================================================
-- 이벤트 검색 최적화
CREATE INDEX idx_event_search ON event_logs(user_id, event_type_id, occurred_at);

-- 자동신고 검색 최적화
CREATE INDEX idx_report_search ON auto_report_logs(event_id, report_target, sent_at);

-- ============================================================
-- 통계 쿼리 예제
-- ============================================================

-- 사용자별 이벤트 통계
/*
SELECT 
    u.name,
    et.type_name,
    COUNT(*) as event_count,
    AVG(el.duration_seconds) as avg_duration
FROM event_logs el
JOIN users u ON el.user_id = u.user_id
JOIN event_types et ON el.event_type_id = et.event_type_id
WHERE el.occurred_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY u.user_id, et.event_type_id
ORDER BY event_count DESC;
*/

-- 시간대별 이벤트 발생 현황
/*
SELECT 
    HOUR(occurred_at) as hour,
    et.type_name,
    COUNT(*) as event_count
FROM event_logs el
JOIN event_types et ON el.event_type_id = et.event_type_id
WHERE el.occurred_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY HOUR(occurred_at), et.event_type_id
ORDER BY hour, event_count DESC;
*/

-- ============================================================
-- 완료
-- ============================================================
SELECT 'Database schema created successfully!' AS Status;