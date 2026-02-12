#!/usr/bin/env python3
"""
============================================================
Home Safe Solution - Pipeline 테스트 스크립트
============================================================

이 스크립트를 프로젝트 루트에서 실행하여 파이프라인이
정상 동작하는지 확인합니다.

실행 방법:
    cd /home/gjkong/dev_ws/yolo/myproj
    python test_pipeline.py

테스트 항목:
    1. 모듈 import 테스트
    2. Config 생성/저장/로드 테스트
    3. 데이터 수집 엔진 테스트 (dry-run)
    4. 전처리 엔진 테스트 (YOLO 로드)
    5. 학습 엔진 테스트 (모델 초기화)
    6. GUI 테스트 (PyQt6)
============================================================
"""

import sys
import os
from pathlib import Path

# 색상 출력
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def ok(msg):
    print(f"  {Colors.GREEN}✅ {msg}{Colors.END}")

def fail(msg):
    print(f"  {Colors.RED}❌ {msg}{Colors.END}")

def warn(msg):
    print(f"  {Colors.YELLOW}⚠️  {msg}{Colors.END}")

def info(msg):
    print(f"  {Colors.BLUE}ℹ️  {msg}{Colors.END}")

def header(msg):
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  {msg}{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")


def test_imports():
    """1. 모듈 import 테스트"""
    header("1. 모듈 Import 테스트")
    
    modules = [
        ("pipeline.config", "PipelineConfig"),
        ("pipeline.data_ingest", "DataIngestEngine"),
        ("pipeline.preprocessor", "PreprocessEngine"),
        ("pipeline.trainer", "RFTrainer, STGCNTrainer"),
        ("pipeline.orchestrator", "TrainingPipelineOrchestrator"),
        ("pipeline._stgcn_model", "STGCNFineTunedModel"),
    ]
    
    success = 0
    for module, classes in modules:
        try:
            exec(f"from {module} import *")
            ok(f"{module} → {classes}")
            success += 1
        except ImportError as e:
            fail(f"{module}: {e}")
        except Exception as e:
            warn(f"{module}: {e}")
    
    print(f"\n  결과: {success}/{len(modules)} 모듈 로드 성공")
    return success == len(modules)


def test_config():
    """2. Config 테스트"""
    header("2. Config 생성/저장/로드 테스트")
    
    try:
        from pipeline.config import PipelineConfig, get_default_config
        
        # 기본 설정 생성
        config = get_default_config()
        ok(f"기본 설정 생성: {config.name}")
        
        # 설정 수정
        config.stgcn_train.epochs = 100
        config.preprocess.sequence_length = 90
        ok(f"설정 수정: epochs={config.stgcn_train.epochs}, seq_len={config.preprocess.sequence_length}")
        
        # 유효성 검사
        errors = config.validate()
        if errors:
            warn(f"설정 검증 경고: {errors}")
        else:
            ok("설정 유효성 검사 통과")
        
        # 저장 테스트
        test_path = "/tmp/test_pipeline_config.json"
        config.save(test_path)
        ok(f"설정 저장: {test_path}")
        
        # 로드 테스트
        loaded = PipelineConfig.load(test_path)
        assert loaded.stgcn_train.epochs == 100
        ok(f"설정 로드 확인: epochs={loaded.stgcn_train.epochs}")
        
        # 요약 출력
        print("\n" + config.summary())
        
        return True
        
    except Exception as e:
        fail(f"Config 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_ingest():
    """3. 데이터 수집 엔진 테스트"""
    header("3. 데이터 수집 엔진 테스트 (Dry-run)")
    
    try:
        from pipeline.config import DataIngestConfig
        from pipeline.data_ingest import DataIngestEngine
        
        config = DataIngestConfig()
        config.raw_video_dir = "/tmp/test_raw_videos"
        
        engine = DataIngestEngine(config)
        ok("DataIngestEngine 초기화")
        
        # 소스 추가 테스트
        engine.add_youtube("https://youtube.com/watch?v=test123", "fall")
        engine.add_url("https://example.com/video.mp4", "normal")
        engine.add_local("/path/to/test.mp4", "fall")
        ok(f"소스 추가: {len(engine.sources)}개")
        
        # 요약 출력
        print(f"\n{engine.get_summary()}")
        
        return True
        
    except Exception as e:
        fail(f"데이터 수집 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessor():
    """4. 전처리 엔진 테스트"""
    header("4. 전처리 엔진 테스트")
    
    try:
        from pipeline.config import PreprocessConfig
        from pipeline.preprocessor import PreprocessEngine
        
        config = PreprocessConfig()
        engine = PreprocessEngine(config)
        ok("PreprocessEngine 초기화")
        
        # YOLO 모델 로드 테스트
        try:
            engine._load_yolo()
            ok(f"YOLO Pose 모델 로드: {config.yolo_model}")
        except ImportError:
            warn("ultralytics 패키지 없음 - pip install ultralytics 필요")
        except Exception as e:
            warn(f"YOLO 로드 실패 (모델 파일 필요): {e}")
        
        return True
        
    except Exception as e:
        fail(f"전처리 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer():
    """5. 학습 엔진 테스트"""
    header("5. 학습 엔진 테스트")
    
    try:
        from pipeline.config import RFTrainConfig, STGCNTrainConfig
        from pipeline.trainer import RFTrainer, STGCNTrainer
        
        # RF Trainer
        rf_config = RFTrainConfig()
        rf_trainer = RFTrainer(rf_config)
        ok("RFTrainer 초기화")
        
        # ST-GCN Trainer
        stgcn_config = STGCNTrainConfig()
        stgcn_trainer = STGCNTrainer(stgcn_config)
        ok("STGCNTrainer 초기화")
        
        # ST-GCN 모델 빌드 테스트
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            info(f"PyTorch 디바이스: {device}")
            
            from pipeline._stgcn_model import STGCNFineTunedModel
            model = STGCNFineTunedModel(num_classes=2)
            ok(f"STGCNFineTunedModel 생성: {sum(p.numel() for p in model.parameters())} params")
            
            # 더미 입력 테스트
            dummy = torch.randn(2, 3, 60, 17, 1)
            with torch.no_grad():
                out = model(dummy)
            ok(f"Forward pass 테스트: input={dummy.shape} → output={out.shape}")
            
        except ImportError:
            warn("PyTorch 없음 - pip install torch 필요")
        
        return True
        
    except Exception as e:
        fail(f"학습 엔진 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator():
    """6. 오케스트레이터 테스트"""
    header("6. 오케스트레이터 테스트")
    
    try:
        from pipeline.config import PipelineConfig
        from pipeline.orchestrator import TrainingPipelineOrchestrator, PipelineState
        
        config = PipelineConfig(name="test_run")
        orchestrator = TrainingPipelineOrchestrator(config)
        ok("TrainingPipelineOrchestrator 초기화")
        
        # 콜백 설정
        logs = []
        orchestrator.on_log = lambda m: logs.append(m)
        ok("콜백 설정 완료")
        
        # 상태 확인
        state = orchestrator.state
        ok(f"초기 상태: stage={state.current_stage}, running={state.is_running}")
        
        return True
        
    except Exception as e:
        fail(f"오케스트레이터 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gui():
    """7. GUI 테스트 (PyQt6)"""
    header("7. GUI 테스트 (PyQt6)")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        ok("PyQt6 import 성공")
        
        # QApplication 생성 (headless)
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        ok("QApplication 생성")
        
        # TrainingPage import (gui/ 폴더에서)
        sys.path.insert(0, str(Path(__file__).parent / "gui"))
        from training_page import TrainingPage
        ok("TrainingPage import 성공")
        
        # 위젯 생성
        page = TrainingPage()
        ok("TrainingPage 인스턴스 생성")
        
        # 설정 동기화 테스트
        page._sync_config_from_gui()
        ok("GUI → Config 동기화 테스트")
        
        info("GUI 테스트 완료 (화면 표시 생략)")
        return True
        
    except ImportError as e:
        warn(f"PyQt6 없음: {e}")
        info("pip install PyQt6 로 설치하세요")
        return False
    except Exception as e:
        fail(f"GUI 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """의존성 패키지 확인"""
    header("0. 의존성 패키지 확인")
    
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("torch", "torch"),
        ("cv2", "opencv-python"),
        ("ultralytics", "ultralytics"),
        ("yt_dlp", "yt-dlp"),
        ("requests", "requests"),
        ("joblib", "joblib"),
        ("PyQt6", "PyQt6"),
    ]
    
    installed = 0
    for module, pip_name in packages:
        try:
            __import__(module)
            ok(f"{pip_name}")
            installed += 1
        except ImportError:
            warn(f"{pip_name} - 설치 필요: pip install {pip_name}")
    
    print(f"\n  결과: {installed}/{len(packages)} 패키지 설치됨")
    return installed >= 5  # 최소 5개 이상


def main():
    """메인 테스트 실행"""
    print(f"\n{Colors.BOLD}{'#'*60}{Colors.END}")
    print(f"{Colors.BOLD}#  Home Safe Solution - Pipeline 테스트{Colors.END}")
    print(f"{Colors.BOLD}{'#'*60}{Colors.END}")
    
    print(f"\n현재 경로: {os.getcwd()}")
    print(f"Python: {sys.version}")
    
    results = {}
    
    # 테스트 실행
    results["dependencies"] = test_dependencies()
    results["imports"] = test_imports()
    results["config"] = test_config()
    results["data_ingest"] = test_data_ingest()
    results["preprocessor"] = test_preprocessor()
    results["trainer"] = test_trainer()
    results["orchestrator"] = test_orchestrator()
    results["gui"] = test_gui()
    
    # 결과 요약
    header("테스트 결과 요약")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {name:20s} : {status}")
    
    print(f"\n{Colors.BOLD}  총 결과: {passed}/{total} 테스트 통과{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}  🎉 모든 테스트 통과! 파이프라인 사용 준비 완료{Colors.END}")
    elif passed >= total - 2:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}  ⚠️  일부 테스트 실패 - 누락된 패키지 설치 후 재시도{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}  ❌ 여러 테스트 실패 - 설치 가이드 확인 필요{Colors.END}")
    
    print(f"\n{'='*60}")
    print("설치 명령어:")
    print("  pip install numpy pandas scikit-learn torch opencv-python")
    print("  pip install ultralytics yt-dlp requests joblib PyQt6")
    print(f"{'='*60}\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
