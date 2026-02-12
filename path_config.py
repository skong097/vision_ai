"""
📁 path_config.py — 프로젝트 경로 중앙 관리

이 파일의 위치: ~/dev_root/path_config.py

모든 .py 파일에서 절대 경로 대신 이 모듈을 import하여 사용:
    from path_config import PATHS
    model_path = PATHS.RF_MODEL
    yolo_path = PATHS.YOLO_MODEL

PC를 옮기거나 디렉토리 구조가 바뀌면 이 파일만 수정하면 됩니다.
"""
import os
from pathlib import Path


# ====================================================================
#  프로젝트 루트 (이 파일의 위치 = dev_root)
# ====================================================================

DEV_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

RF_MAIN = DEV_ROOT / "rf_main"
ST_GCN = DEV_ROOT / "st_gcn"


# ====================================================================
#  경로 클래스
# ====================================================================

class _Paths:
    """프로젝트 내 모든 경로를 속성으로 접근"""
    
    def __init__(self):
        self.DEV_ROOT = DEV_ROOT
        self.RF_MAIN = RF_MAIN
        self.ST_GCN = ST_GCN
        
        # ---- YOLO 모델 ----
        self.YOLO_MODEL = RF_MAIN / "models" / "yolo11s-pose.pt"
        self.YOLO_MODEL_N = RF_MAIN / "models" / "yolo11n-pose.pt"
        
        # ---- RF 모델 ----
        self.RF_MODEL = RF_MAIN / "models_integrated" / "binary_v3" / "random_forest_model.pkl"
        self.RF_FEATURE_COLS = RF_MAIN / "models_integrated" / "binary_v3" / "feature_columns.txt"
        self.RF_MODEL_3CLASS = RF_MAIN / "models_integrated" / "3class" / "random_forest_model.pkl"
        self.RF_FEATURE_3CLASS = RF_MAIN / "models_integrated" / "3class" / "feature_columns.txt"
        self.RF_MODEL_BINARY_OLD = RF_MAIN / "models" / "binary" / "random_forest_model.pkl"
        self.RF_MODEL_3CLASS_OLD = RF_MAIN / "models" / "3class" / "random_forest_model.pkl"
        
        # ---- ST-GCN 모델 ----
        self.STGCN_V2 = ST_GCN / "checkpoints_v2" / "best_model.pth"
        self.STGCN_FINETUNED = ST_GCN / "checkpoints_finetuned" / "best_model_finetuned.pth"
        self.STGCN_ORIGINAL = ST_GCN / "checkpoints" / "best_model_binary.pth" if (ST_GCN / "checkpoints").exists() else None
        self.STGCN_PRETRAINED = ST_GCN / "pretrained" / "stgcn_ntu60_hrnet.pth"
        
        # ---- ST-GCN 데이터 ----
        self.STGCN_DATA_BINARY = ST_GCN / "data" / "binary"
        self.STGCN_DATA_V2 = ST_GCN / "data" / "binary_v2"
        
        # ---- RF 데이터/스크립트 디렉토리 ----
        self.NEW_DATA_DIR = RF_MAIN / "new_data"
        self.NEW_DATA_NORMAL = RF_MAIN / "new_data" / "normal"
        self.NEW_DATA_FALLEN = RF_MAIN / "new_data" / "fallen"
        self.FEATURES_DIR = RF_MAIN / "new_data" / "features"
        
        # ---- 기존 데이터 디렉토리 ----
        self.VIDEO_DIR = RF_MAIN / "data"
        self.ACCEL_DIR = RF_MAIN / "accel"
        self.SKELETON_DIR = RF_MAIN / "skeleton"
        self.FEATURES_OLD_DIR = RF_MAIN / "features"
        self.LABELED_DIR = RF_MAIN / "labeled"
        self.DATASET_DIR = RF_MAIN / "dataset"
        self.MODELS_DIR = RF_MAIN / "models"
        
        # ---- 리포트 ----
        self.COMPARE_REPORT_DIR = RF_MAIN / "scripts" / "admin" / "Model_Compare_Report"
        self.ACCURACY_LOG_DIR = RF_MAIN / "accuracy_logs"
        
        # ---- GUI ----
        self.GUI_DIR = RF_MAIN / "gui"
    
    def __repr__(self):
        lines = ["=== Project Paths ==="]
        for k, v in vars(self).items():
            if not k.startswith('_'):
                exists = "✅" if v and Path(v).exists() else "❌"
                lines.append(f"  {exists} {k}: {v}")
        return "\n".join(lines)


PATHS = _Paths()


# ====================================================================
#  유틸리티 함수
# ====================================================================

def get_str(path_attr) -> str:
    """Path → str 변환 (기존 코드에서 문자열 경로 필요 시)"""
    return str(path_attr)


# ====================================================================
#  독립 실행 시 경로 검증
# ====================================================================

if __name__ == "__main__":
    print(PATHS)
    print()
    
    # 핵심 파일 존재 확인
    critical = {
        "YOLO 모델": PATHS.YOLO_MODEL,
        "RF 모델 (v3)": PATHS.RF_MODEL,
        "ST-GCN v2": PATHS.STGCN_V2,
        "ST-GCN Pretrained": PATHS.STGCN_PRETRAINED,
    }
    
    print("🔍 핵심 파일 존재 확인:")
    all_ok = True
    for name, path in critical.items():
        exists = path and path.exists()
        mark = "✅" if exists else "❌ MISSING"
        print(f"  {mark} {name}: {path}")
        if not exists:
            all_ok = False
    
    print()
    if all_ok:
        print("✅ 모든 핵심 파일 확인 완료!")
    else:
        print("⚠️  누락 파일이 있습니다. 경로를 확인해 주세요.")
