#!/usr/bin/env python3
"""
============================================================
Home Safe Solution - Training Pipeline GUI Page (PyQt6)
============================================================
학습 파이프라인을 설정하고 실행하는 PyQt6 GUI 페이지

main.py에서 통합:
    from training_page import TrainingPage
    training_page = TrainingPage()
============================================================
"""

import sys
import os
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QLineEdit, QTextEdit,
    QProgressBar, QGroupBox, QFormLayout, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QTabWidget, QScrollArea, QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# 파이프라인 모듈 경로 추가 (gui/ 폴더에서 실행되므로 상위 폴더 참조)
sys.path.insert(0, str(Path(__file__).parent.parent))  # myproj/ 를 path에 추가

try:
    from pipeline.config import PipelineConfig, DATASET_DIR, REPORT_DIR, SUPPORTED_VIDEO_FORMATS
    from pipeline.orchestrator import TrainingPipelineOrchestrator
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠ pipeline 모듈 import 실패: {e}")
    print("  pipeline/ 디렉토리가 myproj/ 아래에 있는지 확인하세요.")
    PIPELINE_AVAILABLE = False
    PipelineConfig = None
    DATASET_DIR = Path(".")
    REPORT_DIR = Path(".")
    SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ============================================================
# PyQt6용 PipelineWorker
# ============================================================
class PipelineWorker(QThread):
    """GUI용 비동기 Worker Thread (PyQt6)"""
    stage_changed = pyqtSignal(int, str)
    progress = pyqtSignal(int, int, str)
    metric = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.orchestrator = None

    def run(self):
        if not PIPELINE_AVAILABLE:
            self.error_signal.emit("pipeline 모듈을 찾을 수 없습니다.")
            return
            
        self.orchestrator = TrainingPipelineOrchestrator(self.config)
        self.orchestrator.on_stage_changed = lambda s, n: self.stage_changed.emit(s, n)
        self.orchestrator.on_progress = lambda c, t, m: self.progress.emit(c, t, m)
        self.orchestrator.on_metric = lambda d: self.metric.emit(d)
        self.orchestrator.on_log = lambda m: self.log_message.emit(m)
        self.orchestrator.on_finished = lambda r: self.finished_signal.emit(r)
        self.orchestrator.on_error = lambda e: self.error_signal.emit(e)
        self.orchestrator.run()

    def cancel(self):
        if self.orchestrator:
            self.orchestrator.cancel()


class TrainingPage(QWidget):
    """학습 파이프라인 메인 페이지 (PyQt6)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 설정 초기화
        if PIPELINE_AVAILABLE and PipelineConfig:
            self.config = PipelineConfig(name="gui_session")
        else:
            self.config = None
            
        self.worker = None
        self.sources = []  # 데이터 소스 목록
        
        self._init_ui()
        self._sync_gui_from_config()

    def _init_ui(self):
        """UI 초기화"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ========== 우측 메인 영역 (먼저 생성) ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # 패널 스택 (사이드바보다 먼저 생성해야 함)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._create_data_panel())       # 0
        self.stack.addWidget(self._create_preprocess_panel()) # 1
        self.stack.addWidget(self._create_training_panel())   # 2
        self.stack.addWidget(self._create_monitor_panel())    # 3
        self.stack.addWidget(self._create_results_panel())    # 4

        # 스플리터: 패널 + 로그
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.stack)
        splitter.addWidget(self._create_log_panel())
        splitter.setSizes([500, 200])

        right_layout.addWidget(splitter)

        # ========== 좌측 사이드바 (stack 생성 후에 생성) ==========
        sidebar = self._create_sidebar()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("""
            QWidget { background-color: #f8fafc; }
            QListWidget { border: none; font-size: 13px; }
            QListWidget::item { padding: 10px 8px; border-radius: 4px; margin: 2px 4px; }
            QListWidget::item:selected { background-color: #e2e8f0; color: #1e293b; }
            QListWidget::item:hover { background-color: #f1f5f9; }
        """)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(right_widget, stretch=1)

    # ================================================================
    # 사이드바
    # ================================================================

    def _create_sidebar(self) -> QWidget:
        """좌측 네비게이션 + 실행 버튼"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 15, 8, 15)
        layout.setSpacing(8)

        # 제목
        title = QLabel("🎓 Training Pipeline")
        title.setFont(QFont("", 12, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #334155; padding: 5px;")
        layout.addWidget(title)

        layout.addSpacing(10)

        # 네비게이션 리스트
        self.nav_list = QListWidget()
        nav_items = [
            ("📥  데이터 소스", 0),
            ("⚙️  전처리 설정", 1),
            ("🎯  학습 설정", 2),
            ("📈  학습 모니터", 3),
            ("📊  결과 뷰어", 4),
        ]
        for text, idx in nav_items:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.nav_list.addItem(item)

        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        self.nav_list.setCurrentRow(0)
        layout.addWidget(self.nav_list)

        layout.addSpacing(20)

        # 실행 버튼들
        self.btn_run_all = QPushButton("▶  전체 실행")
        self.btn_run_all.setStyleSheet("""
            QPushButton {
                background-color: #22c55e; color: white;
                font-weight: bold; font-size: 13px;
                padding: 10px; border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #16a34a; }
            QPushButton:disabled { background-color: #9ca3af; }
        """)
        self.btn_run_all.clicked.connect(self._on_run_all)
        layout.addWidget(self.btn_run_all)

        self.btn_run_from = QPushButton("▶  선택 스테이지부터")
        self.btn_run_from.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6; color: white;
                font-size: 12px; padding: 8px; border-radius: 5px; border: none;
            }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:disabled { background-color: #9ca3af; }
        """)
        self.btn_run_from.clicked.connect(self._on_run_from_stage)
        layout.addWidget(self.btn_run_from)

        layout.addSpacing(5)

        self.btn_cancel = QPushButton("⏹  중단")
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #ef4444; color: white;
                font-size: 12px; padding: 8px; border-radius: 5px; border: none;
            }
            QPushButton:hover { background-color: #dc2626; }
            QPushButton:disabled { background-color: #9ca3af; }
        """)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_cancel.setEnabled(False)
        layout.addWidget(self.btn_cancel)

        layout.addStretch()

        # 설정 저장/불러오기
        btn_save = QPushButton("💾  설정 저장")
        btn_save.clicked.connect(self._on_save_config)
        layout.addWidget(btn_save)

        btn_load = QPushButton("📂  설정 불러오기")
        btn_load.clicked.connect(self._on_load_config)
        layout.addWidget(btn_load)

        return widget

    def _on_nav_changed(self, row: int):
        """네비게이션 변경"""
        self.stack.setCurrentIndex(row)

    # ================================================================
    # Panel 0: 데이터 소스
    # ================================================================

    def _create_data_panel(self) -> QWidget:
        """데이터 소스 관리 패널"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # 헤더
        header = QLabel("📥 데이터 소스 관리")
        header.setFont(QFont("", 16, QFont.Weight.Bold))
        header.setStyleSheet("color: #1e293b;")
        layout.addWidget(header)

        desc = QLabel("YouTube, URL, 로컬 파일에서 학습 데이터를 수집합니다.")
        desc.setStyleSheet("color: #64748b;")
        layout.addWidget(desc)

        # 소스 추가 그룹
        add_group = QGroupBox("➕ 소스 추가")
        add_layout = QFormLayout(add_group)
        add_layout.setSpacing(10)

        self.cmb_source_type = QComboBox()
        self.cmb_source_type.addItems(["YouTube URL", "인터넷 URL", "로컬 파일", "로컬 폴더"])
        add_layout.addRow("소스 타입:", self.cmb_source_type)

        path_row = QHBoxLayout()
        self.txt_source_path = QLineEdit()
        self.txt_source_path.setPlaceholderText("URL 또는 파일/폴더 경로 입력...")
        path_row.addWidget(self.txt_source_path)
        btn_browse = QPushButton("📁 찾아보기")
        btn_browse.setFixedWidth(100)
        btn_browse.clicked.connect(self._on_browse_source)
        path_row.addWidget(btn_browse)
        add_layout.addRow("경로:", path_row)

        self.cmb_label = QComboBox()
        self.cmb_label.addItems(["fall", "normal"])
        add_layout.addRow("라벨:", self.cmb_label)

        btn_add = QPushButton("➕ 소스 추가")
        btn_add.setStyleSheet("background-color: #3b82f6; color: white; padding: 6px; border-radius: 4px;")
        btn_add.clicked.connect(self._on_add_source)
        add_layout.addRow("", btn_add)

        layout.addWidget(add_group)

        # 소스 테이블
        self.tbl_sources = QTableWidget(0, 5)
        self.tbl_sources.setHorizontalHeaderLabels(["#", "타입", "경로", "라벨", "상태"])
        self.tbl_sources.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tbl_sources.setMinimumHeight(200)
        self.tbl_sources.setAlternatingRowColors(True)
        layout.addWidget(self.tbl_sources)

        # 버튼 행
        btn_row = QHBoxLayout()
        btn_folder = QPushButton("📂 폴더 일괄 추가 (fall/normal 구조)")
        btn_folder.clicked.connect(self._on_add_folder_batch)
        btn_row.addWidget(btn_folder)

        btn_remove = QPushButton("🗑 선택 삭제")
        btn_remove.clicked.connect(self._on_remove_source)
        btn_row.addWidget(btn_remove)

        btn_clear = QPushButton("🗑 전체 삭제")
        btn_clear.clicked.connect(self._on_clear_sources)
        btn_row.addWidget(btn_clear)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # 요약
        self.lbl_data_summary = QLabel("📊 총 소스: 0개 (Fall: 0, Normal: 0)")
        self.lbl_data_summary.setStyleSheet("font-size: 13px; color: #475569; padding: 5px; background-color: #f1f5f9; border-radius: 4px;")
        layout.addWidget(self.lbl_data_summary)

        layout.addStretch()
        scroll.setWidget(widget)
        return scroll

    def _on_browse_source(self):
        """파일/폴더 찾아보기"""
        source_type = self.cmb_source_type.currentText()
        if "폴더" in source_type:
            path = QFileDialog.getExistingDirectory(self, "폴더 선택")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "파일 선택", "",
                "Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
            )
        if path:
            self.txt_source_path.setText(path)

    def _on_add_source(self):
        """소스 추가"""
        path = self.txt_source_path.text().strip()
        if not path:
            QMessageBox.warning(self, "경고", "경로를 입력하세요.")
            return

        source_type_map = {"YouTube URL": "youtube", "인터넷 URL": "url", "로컬 파일": "local", "로컬 폴더": "folder"}
        source_type = source_type_map.get(self.cmb_source_type.currentText(), "local")
        label = self.cmb_label.currentText()

        self.sources.append({"type": source_type, "path": path, "label": label, "status": "pending"})
        self._refresh_source_table()
        self.txt_source_path.clear()

    def _on_add_folder_batch(self):
        """fall/normal 구조 폴더 일괄 추가"""
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택 (fall/, normal/ 하위 구조)")
        if not folder:
            return

        folder_path = Path(folder)
        added = 0

        for label_name in ["fall", "normal"]:
            sub_dir = folder_path / label_name
            if sub_dir.exists():
                for f in sub_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                        self.sources.append({
                            "type": "local",
                            "path": str(f),
                            "label": label_name,
                            "status": "pending",
                        })
                        added += 1

        if added > 0:
            self._refresh_source_table()
            QMessageBox.information(self, "완료", f"{added}개 비디오가 추가되었습니다.")
        else:
            QMessageBox.warning(self, "경고", "fall/, normal/ 폴더에서 비디오를 찾을 수 없습니다.")

    def _on_remove_source(self):
        """선택된 소스 삭제"""
        rows = set(item.row() for item in self.tbl_sources.selectedItems())
        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self.sources):
                del self.sources[row]
        self._refresh_source_table()

    def _on_clear_sources(self):
        """모든 소스 삭제"""
        if self.sources:
            reply = QMessageBox.question(self, "확인", "모든 소스를 삭제할까요?")
            if reply == QMessageBox.StandardButton.Yes:
                self.sources.clear()
                self._refresh_source_table()

    def _refresh_source_table(self):
        """소스 테이블 갱신"""
        self.tbl_sources.setRowCount(len(self.sources))
        fall_count = 0
        normal_count = 0

        for i, src in enumerate(self.sources):
            self.tbl_sources.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.tbl_sources.setItem(i, 1, QTableWidgetItem(src["type"]))
            self.tbl_sources.setItem(i, 2, QTableWidgetItem(src["path"]))
            self.tbl_sources.setItem(i, 3, QTableWidgetItem(src["label"]))
            self.tbl_sources.setItem(i, 4, QTableWidgetItem(src["status"]))

            if src["label"] == "fall":
                fall_count += 1
            else:
                normal_count += 1

        self.lbl_data_summary.setText(
            f"📊 총 소스: {len(self.sources)}개 (Fall: {fall_count}, Normal: {normal_count})"
        )

    # ================================================================
    # Panel 1: 전처리 설정
    # ================================================================

    def _create_preprocess_panel(self) -> QWidget:
        """전처리 설정 패널"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        header = QLabel("⚙️ 전처리 설정")
        header.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        # 비디오 설정
        vid_group = QGroupBox("📹 비디오 정규화")
        vid_layout = QFormLayout(vid_group)

        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(10, 60)
        self.spn_fps.setValue(30)
        vid_layout.addRow("타겟 FPS:", self.spn_fps)

        self.spn_conf = QDoubleSpinBox()
        self.spn_conf.setRange(0.1, 0.9)
        self.spn_conf.setSingleStep(0.05)
        self.spn_conf.setValue(0.5)
        vid_layout.addRow("YOLO Confidence:", self.spn_conf)

        self.cmb_target_method = QComboBox()
        self.cmb_target_method.addItems(["largest", "center", "combined"])
        vid_layout.addRow("대상자 선택:", self.cmb_target_method)

        layout.addWidget(vid_group)

        # ST-GCN 시퀀스 설정
        seq_group = QGroupBox("🔢 ST-GCN 시퀀스")
        seq_layout = QFormLayout(seq_group)

        self.spn_seq_len = QSpinBox()
        self.spn_seq_len.setRange(20, 180)
        self.spn_seq_len.setValue(60)
        seq_layout.addRow("시퀀스 길이 (프레임):", self.spn_seq_len)

        self.spn_stride = QSpinBox()
        self.spn_stride.setRange(5, 90)
        self.spn_stride.setValue(30)
        seq_layout.addRow("Stride:", self.spn_stride)

        self.cmb_normalize = QComboBox()
        self.cmb_normalize.addItems(["center", "minmax", "none"])
        seq_layout.addRow("정규화 방법:", self.cmb_normalize)

        layout.addWidget(seq_group)

        # 데이터 분할
        split_group = QGroupBox("📊 데이터 분할")
        split_layout = QFormLayout(split_group)

        self.spn_train_ratio = QDoubleSpinBox()
        self.spn_train_ratio.setRange(0.5, 0.9)
        self.spn_train_ratio.setSingleStep(0.05)
        self.spn_train_ratio.setValue(0.70)
        split_layout.addRow("Train 비율:", self.spn_train_ratio)

        self.spn_val_ratio = QDoubleSpinBox()
        self.spn_val_ratio.setRange(0.05, 0.3)
        self.spn_val_ratio.setSingleStep(0.05)
        self.spn_val_ratio.setValue(0.15)
        split_layout.addRow("Val 비율:", self.spn_val_ratio)

        self.spn_test_ratio = QDoubleSpinBox()
        self.spn_test_ratio.setRange(0.05, 0.3)
        self.spn_test_ratio.setSingleStep(0.05)
        self.spn_test_ratio.setValue(0.15)
        split_layout.addRow("Test 비율:", self.spn_test_ratio)

        layout.addWidget(split_group)

        layout.addStretch()
        scroll.setWidget(widget)
        return scroll

    # ================================================================
    # Panel 2: 학습 설정
    # ================================================================

    def _create_training_panel(self) -> QWidget:
        """학습 설정 패널"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        header = QLabel("🎯 학습 설정")
        header.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        tabs = QTabWidget()

        # ---- RF 탭 ----
        rf_widget = QWidget()
        rf_layout = QFormLayout(rf_widget)
        rf_layout.setSpacing(12)

        self.chk_rf_enabled = QCheckBox("Random Forest 학습 활성화")
        self.chk_rf_enabled.setChecked(True)
        rf_layout.addRow(self.chk_rf_enabled)

        self.spn_n_estimators = QSpinBox()
        self.spn_n_estimators.setRange(10, 1000)
        self.spn_n_estimators.setValue(100)
        rf_layout.addRow("n_estimators:", self.spn_n_estimators)

        self.chk_rf_tune = QCheckBox("하이퍼파라미터 튜닝")
        rf_layout.addRow(self.chk_rf_tune)

        self.cmb_rf_tune_method = QComboBox()
        self.cmb_rf_tune_method.addItems(["grid", "random"])
        rf_layout.addRow("튜닝 방법:", self.cmb_rf_tune_method)

        self.cmb_rf_scoring = QComboBox()
        self.cmb_rf_scoring.addItems(["f1", "accuracy", "recall", "precision"])
        rf_layout.addRow("평가 지표:", self.cmb_rf_scoring)

        tabs.addTab(rf_widget, "🌲 Random Forest")

        # ---- ST-GCN 탭 ----
        stgcn_widget = QWidget()
        stgcn_layout = QFormLayout(stgcn_widget)
        stgcn_layout.setSpacing(12)

        self.chk_stgcn_enabled = QCheckBox("ST-GCN 학습 활성화")
        self.chk_stgcn_enabled.setChecked(True)
        stgcn_layout.addRow(self.chk_stgcn_enabled)

        self.spn_epochs = QSpinBox()
        self.spn_epochs.setRange(1, 500)
        self.spn_epochs.setValue(50)
        stgcn_layout.addRow("Epochs:", self.spn_epochs)

        self.spn_batch_size = QSpinBox()
        self.spn_batch_size.setRange(4, 128)
        self.spn_batch_size.setValue(16)
        stgcn_layout.addRow("Batch Size:", self.spn_batch_size)

        self.spn_backbone_lr = QDoubleSpinBox()
        self.spn_backbone_lr.setDecimals(6)
        self.spn_backbone_lr.setRange(0.000001, 0.01)
        self.spn_backbone_lr.setValue(0.00001)
        stgcn_layout.addRow("Backbone LR:", self.spn_backbone_lr)

        self.spn_head_lr = QDoubleSpinBox()
        self.spn_head_lr.setDecimals(5)
        self.spn_head_lr.setRange(0.0001, 0.1)
        self.spn_head_lr.setValue(0.001)
        stgcn_layout.addRow("Head LR:", self.spn_head_lr)

        self.cmb_scheduler = QComboBox()
        self.cmb_scheduler.addItems(["cosine", "step", "plateau"])
        stgcn_layout.addRow("스케줄러:", self.cmb_scheduler)

        self.chk_early_stop = QCheckBox("Early Stopping")
        self.chk_early_stop.setChecked(True)
        stgcn_layout.addRow(self.chk_early_stop)

        self.spn_patience = QSpinBox()
        self.spn_patience.setRange(3, 50)
        self.spn_patience.setValue(10)
        stgcn_layout.addRow("Patience:", self.spn_patience)

        self.chk_use_pretrained = QCheckBox("Pre-trained 가중치 사용")
        self.chk_use_pretrained.setChecked(True)
        stgcn_layout.addRow(self.chk_use_pretrained)

        tabs.addTab(stgcn_widget, "🚀 ST-GCN")

        # ---- 비교 탭 ----
        compare_widget = QWidget()
        compare_layout = QFormLayout(compare_widget)

        self.chk_auto_compare = QCheckBox("학습 완료 후 자동 비교 실행")
        self.chk_auto_compare.setChecked(True)
        compare_layout.addRow(self.chk_auto_compare)

        self.spn_inference_repeat = QSpinBox()
        self.spn_inference_repeat.setRange(10, 200)
        self.spn_inference_repeat.setValue(50)
        compare_layout.addRow("추론 속도 측정 횟수:", self.spn_inference_repeat)

        tabs.addTab(compare_widget, "📊 자동 비교")

        layout.addWidget(tabs)
        layout.addStretch()
        scroll.setWidget(widget)
        return scroll

    # ================================================================
    # Panel 3: 학습 모니터
    # ================================================================

    def _create_monitor_panel(self) -> QWidget:
        """학습 진행 모니터 패널"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        header = QLabel("📈 학습 모니터")
        header.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        # 현재 단계
        stage_group = QGroupBox("🔄 현재 진행")
        stage_layout = QFormLayout(stage_group)

        self.lbl_stage = QLabel("대기 중")
        self.lbl_stage.setFont(QFont("", 13, QFont.Weight.Bold))
        self.lbl_stage.setStyleSheet("color: #3b82f6;")
        stage_layout.addRow("현재 단계:", self.lbl_stage)

        self.progress_overall = QProgressBar()
        self.progress_overall.setRange(0, 100)
        self.progress_overall.setValue(0)
        self.progress_overall.setTextVisible(True)
        self.progress_overall.setStyleSheet("""
            QProgressBar { border: 1px solid #cbd5e1; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background-color: #22c55e; }
        """)
        stage_layout.addRow("전체 진행:", self.progress_overall)

        self.lbl_elapsed = QLabel("경과 시간: 0초")
        stage_layout.addRow(self.lbl_elapsed)

        layout.addWidget(stage_group)

        # RF 상태
        rf_group = QGroupBox("🌲 Random Forest")
        rf_layout = QFormLayout(rf_group)
        self.lbl_rf_status = QLabel("대기")
        self.lbl_rf_f1 = QLabel("-")
        self.lbl_rf_params = QLabel("-")
        rf_layout.addRow("상태:", self.lbl_rf_status)
        rf_layout.addRow("Best F1:", self.lbl_rf_f1)
        rf_layout.addRow("최적 파라미터:", self.lbl_rf_params)
        layout.addWidget(rf_group)

        # ST-GCN 상태
        stgcn_group = QGroupBox("🚀 ST-GCN Fine-tuning")
        stgcn_layout = QFormLayout(stgcn_group)

        self.lbl_stgcn_status = QLabel("대기")
        stgcn_layout.addRow("상태:", self.lbl_stgcn_status)

        self.lbl_stgcn_epoch = QLabel("0 / 0")
        stgcn_layout.addRow("Epoch:", self.lbl_stgcn_epoch)

        self.lbl_stgcn_train = QLabel("Loss: - / Acc: -")
        stgcn_layout.addRow("Train:", self.lbl_stgcn_train)

        self.lbl_stgcn_val = QLabel("Loss: - / Acc: -")
        stgcn_layout.addRow("Val:", self.lbl_stgcn_val)

        self.lbl_stgcn_best = QLabel("Best Val Acc: -")
        self.lbl_stgcn_best.setStyleSheet("color: #16a34a; font-weight: bold;")
        stgcn_layout.addRow(self.lbl_stgcn_best)

        self.lbl_stgcn_patience = QLabel("Early Stop: -")
        stgcn_layout.addRow(self.lbl_stgcn_patience)

        self.progress_epoch = QProgressBar()
        self.progress_epoch.setRange(0, 100)
        self.progress_epoch.setStyleSheet("""
            QProgressBar { border: 1px solid #cbd5e1; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background-color: #3b82f6; }
        """)
        stgcn_layout.addRow("Epoch 진행:", self.progress_epoch)

        layout.addWidget(stgcn_group)

        layout.addStretch()
        return widget

    # ================================================================
    # Panel 4: 결과 뷰어
    # ================================================================

    def _create_results_panel(self) -> QWidget:
        """결과 리포트 뷰어"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        header = QLabel("📊 결과 뷰어")
        header.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        # 리포트 선택
        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("리포트 폴더:"))
        self.cmb_reports = QComboBox()
        self.cmb_reports.setMinimumWidth(300)
        self.cmb_reports.currentIndexChanged.connect(self._on_report_selected)
        select_row.addWidget(self.cmb_reports)

        btn_refresh = QPushButton("🔄 새로고침")
        btn_refresh.clicked.connect(self._refresh_reports)
        select_row.addWidget(btn_refresh)

        btn_open_folder = QPushButton("📂 폴더 열기")
        btn_open_folder.clicked.connect(self._on_open_report_folder)
        select_row.addWidget(btn_open_folder)

        select_row.addStretch()
        layout.addLayout(select_row)

        # 리포트 내용
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setFont(QFont("Consolas", 10))
        self.txt_report.setStyleSheet("background-color: #f8fafc; border: 1px solid #e2e8f0;")
        layout.addWidget(self.txt_report)

        self._refresh_reports()
        return widget

    def _refresh_reports(self):
        """리포트 목록 새로고침"""
        self.cmb_reports.clear()
        report_base = Path(REPORT_DIR) if REPORT_DIR else Path(".")

        if report_base.exists():
            dirs = sorted(
                [d for d in report_base.iterdir() if d.is_dir()],
                key=lambda x: x.name,
                reverse=True
            )
            for d in dirs[:20]:  # 최근 20개만
                self.cmb_reports.addItem(d.name, str(d))

        if self.cmb_reports.count() == 0:
            self.cmb_reports.addItem("(리포트 없음)")

    def _on_report_selected(self, index: int):
        """리포트 선택됨"""
        path = self.cmb_reports.currentData()
        if not path:
            return

        report_file = Path(path) / "MODEL_COMPARISON_REPORT.md"
        if report_file.exists():
            self.txt_report.setPlainText(report_file.read_text(encoding="utf-8"))
        else:
            self.txt_report.setPlainText(f"리포트 파일 없음: {report_file}")

    def _on_open_report_folder(self):
        """리포트 폴더 열기"""
        path = self.cmb_reports.currentData()
        if path and Path(path).exists():
            import subprocess
            import platform
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])

    # ================================================================
    # 로그 패널
    # ================================================================

    def _create_log_panel(self) -> QWidget:
        """하단 로그 패널"""
        group = QGroupBox("📋 로그")
        layout = QVBoxLayout(group)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 9))
        self.txt_log.setStyleSheet("background-color: #1e293b; color: #e2e8f0;")
        self.txt_log.setMaximumHeight(180)
        layout.addWidget(self.txt_log)

        return group

    def _log(self, message: str):
        """로그 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{timestamp}] {message}")
        # 자동 스크롤
        scrollbar = self.txt_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ================================================================
    # 파이프라인 실행
    # ================================================================

    def _on_run_all(self):
        """전체 파이프라인 실행"""
        if not PIPELINE_AVAILABLE:
            QMessageBox.critical(self, "오류", "pipeline 모듈을 찾을 수 없습니다.\npipeline/ 폴더가 프로젝트에 있는지 확인하세요.")
            return
            
        self._sync_config_from_gui()
        self.config.start_stage = 1
        self.config.end_stage = 4
        self._start_pipeline()

    def _on_run_from_stage(self):
        """선택된 스테이지부터 실행"""
        if not PIPELINE_AVAILABLE:
            QMessageBox.critical(self, "오류", "pipeline 모듈을 찾을 수 없습니다.")
            return
            
        row = self.nav_list.currentRow()
        stage_map = {0: 1, 1: 2, 2: 3, 3: 3, 4: 4}
        start = stage_map.get(row, 1)

        self._sync_config_from_gui()
        self.config.start_stage = start
        self.config.end_stage = 4
        self._start_pipeline()

    def _start_pipeline(self):
        """파이프라인 워커 시작"""
        # 설정 검증
        errors = self.config.validate()
        if errors:
            QMessageBox.warning(self, "설정 오류", "\n".join(errors))
            return

        # UI 상태 변경
        self.btn_run_all.setEnabled(False)
        self.btn_run_from.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        # 모니터 패널로 이동
        self.nav_list.setCurrentRow(3)

        # 초기화
        self._reset_monitor()
        self._log(f"파이프라인 시작: Stage {self.config.start_stage} → {self.config.end_stage}")

        # 워커 생성 및 시작
        self.worker = PipelineWorker(self.config, self)
        self.worker.stage_changed.connect(self._on_stage_changed)
        self.worker.progress.connect(self._on_progress)
        self.worker.metric.connect(self._on_metric)
        self.worker.log_message.connect(self._log)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()

    def _on_cancel(self):
        """파이프라인 취소"""
        if self.worker:
            self.worker.cancel()
            self._log("⏹ 취소 요청됨...")

    def _reset_monitor(self):
        """모니터 패널 초기화"""
        self.lbl_stage.setText("시작 중...")
        self.progress_overall.setValue(0)
        self.lbl_elapsed.setText("경과 시간: 0초")

        self.lbl_rf_status.setText("대기")
        self.lbl_rf_f1.setText("-")
        self.lbl_rf_params.setText("-")

        self.lbl_stgcn_status.setText("대기")
        self.lbl_stgcn_epoch.setText("0 / 0")
        self.lbl_stgcn_train.setText("Loss: - / Acc: -")
        self.lbl_stgcn_val.setText("Loss: - / Acc: -")
        self.lbl_stgcn_best.setText("Best Val Acc: -")
        self.lbl_stgcn_patience.setText("Early Stop: -")
        self.progress_epoch.setValue(0)

    def _on_stage_changed(self, stage: int, name: str):
        """스테이지 변경 콜백"""
        self.lbl_stage.setText(f"Stage {stage}: {name}")
        stage_progress = {1: 10, 2: 30, 3: 70, 4: 90}
        self.progress_overall.setValue(stage_progress.get(stage, 0))

    def _on_progress(self, current: int, total: int, message: str):
        """진행률 콜백"""
        if total > 0:
            pct = int(current / total * 100)
            self.progress_epoch.setValue(pct)

    def _on_metric(self, data: dict):
        """메트릭 콜백"""
        model = data.get("model", "")
        event = data.get("event", "")

        if model == "RF":
            if event == "done":
                self.lbl_rf_status.setText("✅ 완료")
                self.lbl_rf_f1.setText(f"{data.get('metric', 0):.4f}")
                params = data.get("params", {})
                self.lbl_rf_params.setText(str(params)[:50])

        elif model == "ST-GCN":
            if event == "epoch":
                epoch = data.get("epoch", 0)
                epochs = self.config.stgcn_train.epochs if self.config else 50
                self.lbl_stgcn_status.setText("🔄 학습 중")
                self.lbl_stgcn_epoch.setText(f"{epoch} / {epochs}")
                self.lbl_stgcn_train.setText(
                    f"Loss: {data.get('train_loss', 0):.4f} / Acc: {data.get('train_acc', 0):.1f}%"
                )
                self.lbl_stgcn_val.setText(
                    f"Loss: {data.get('val_loss', 0):.4f} / Acc: {data.get('val_acc', 0):.1f}%"
                )
                self.lbl_stgcn_best.setText(
                    f"Best Val Acc: {data.get('best_val_acc', 0):.2f}% (Epoch {data.get('best_epoch', 0)})"
                )
                self.lbl_stgcn_patience.setText(f"Early Stop: {data.get('patience', '-')}")
                self.progress_epoch.setValue(int(epoch / epochs * 100))

    def _on_finished(self, results: dict):
        """파이프라인 완료 콜백"""
        self.btn_run_all.setEnabled(True)
        self.btn_run_from.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        self.lbl_stage.setText("✅ 완료")
        self.progress_overall.setValue(100)
        self.lbl_stgcn_status.setText("✅ 완료")

        self._log("=" * 50)
        self._log("파이프라인 완료!")
        self._log("=" * 50)

        # 결과 뷰어로 전환
        self._refresh_reports()
        self.nav_list.setCurrentRow(4)

        QMessageBox.information(self, "완료", "학습 파이프라인이 완료되었습니다!")

    def _on_error(self, error_msg: str):
        """에러 콜백"""
        self.btn_run_all.setEnabled(True)
        self.btn_run_from.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        self.lbl_stage.setText("❌ 오류")
        self._log(f"❌ 오류: {error_msg}")

        QMessageBox.critical(self, "오류", f"파이프라인 오류:\n{error_msg}")

    # ================================================================
    # 설정 동기화
    # ================================================================

    def _sync_config_from_gui(self):
        """GUI → Config 동기화"""
        if not self.config:
            return

        # 데이터 소스
        self.config.data_ingest.sources = self.sources.copy()

        # 전처리
        self.config.preprocess.target_fps = self.spn_fps.value()
        self.config.preprocess.confidence_threshold = self.spn_conf.value()
        self.config.preprocess.select_target_method = self.cmb_target_method.currentText()
        self.config.preprocess.sequence_length = self.spn_seq_len.value()
        self.config.preprocess.sequence_stride = self.spn_stride.value()
        self.config.preprocess.normalize_method = self.cmb_normalize.currentText()
        self.config.preprocess.train_ratio = self.spn_train_ratio.value()
        self.config.preprocess.val_ratio = self.spn_val_ratio.value()
        self.config.preprocess.test_ratio = self.spn_test_ratio.value()

        # RF 학습
        self.config.rf_train.enabled = self.chk_rf_enabled.isChecked()
        self.config.rf_train.n_estimators = self.spn_n_estimators.value()
        self.config.rf_train.tuning_enabled = self.chk_rf_tune.isChecked()
        self.config.rf_train.tuning_method = self.cmb_rf_tune_method.currentText()
        self.config.rf_train.scoring = self.cmb_rf_scoring.currentText()

        # ST-GCN 학습
        self.config.stgcn_train.enabled = self.chk_stgcn_enabled.isChecked()
        self.config.stgcn_train.epochs = self.spn_epochs.value()
        self.config.stgcn_train.batch_size = self.spn_batch_size.value()
        self.config.stgcn_train.backbone_lr = self.spn_backbone_lr.value()
        self.config.stgcn_train.head_lr = self.spn_head_lr.value()
        self.config.stgcn_train.scheduler = self.cmb_scheduler.currentText()
        self.config.stgcn_train.early_stopping = self.chk_early_stop.isChecked()
        self.config.stgcn_train.patience = self.spn_patience.value()
        self.config.stgcn_train.use_pretrained = self.chk_use_pretrained.isChecked()

        # 자동 비교
        self.config.auto_compare.enabled = self.chk_auto_compare.isChecked()
        self.config.auto_compare.inference_repeat = self.spn_inference_repeat.value()

    def _sync_gui_from_config(self):
        """Config → GUI 동기화"""
        if not self.config:
            return

        # 전처리
        self.spn_fps.setValue(self.config.preprocess.target_fps)
        self.spn_conf.setValue(self.config.preprocess.confidence_threshold)
        idx = self.cmb_target_method.findText(self.config.preprocess.select_target_method)
        if idx >= 0:
            self.cmb_target_method.setCurrentIndex(idx)
        self.spn_seq_len.setValue(self.config.preprocess.sequence_length)
        self.spn_stride.setValue(self.config.preprocess.sequence_stride)
        idx = self.cmb_normalize.findText(self.config.preprocess.normalize_method)
        if idx >= 0:
            self.cmb_normalize.setCurrentIndex(idx)
        self.spn_train_ratio.setValue(self.config.preprocess.train_ratio)
        self.spn_val_ratio.setValue(self.config.preprocess.val_ratio)
        self.spn_test_ratio.setValue(self.config.preprocess.test_ratio)

        # RF
        self.chk_rf_enabled.setChecked(self.config.rf_train.enabled)
        self.spn_n_estimators.setValue(self.config.rf_train.n_estimators)
        self.chk_rf_tune.setChecked(self.config.rf_train.tuning_enabled)

        # ST-GCN
        self.chk_stgcn_enabled.setChecked(self.config.stgcn_train.enabled)
        self.spn_epochs.setValue(self.config.stgcn_train.epochs)
        self.spn_batch_size.setValue(self.config.stgcn_train.batch_size)
        self.spn_backbone_lr.setValue(self.config.stgcn_train.backbone_lr)
        self.spn_head_lr.setValue(self.config.stgcn_train.head_lr)
        self.chk_early_stop.setChecked(self.config.stgcn_train.early_stopping)
        self.spn_patience.setValue(self.config.stgcn_train.patience)
        self.chk_use_pretrained.setChecked(self.config.stgcn_train.use_pretrained)

        # 비교
        self.chk_auto_compare.setChecked(self.config.auto_compare.enabled)
        self.spn_inference_repeat.setValue(self.config.auto_compare.inference_repeat)

    # ================================================================
    # 설정 저장/불러오기
    # ================================================================

    def _on_save_config(self):
        """설정 JSON 저장"""
        if not self.config:
            QMessageBox.warning(self, "경고", "설정을 저장할 수 없습니다.")
            return
            
        self._sync_config_from_gui()
        path, _ = QFileDialog.getSaveFileName(
            self, "설정 저장", "pipeline_config.json", "JSON Files (*.json)"
        )
        if path:
            self.config.save(path)
            self._log(f"설정 저장: {path}")
            QMessageBox.information(self, "저장 완료", f"설정이 저장되었습니다:\n{path}")

    def _on_load_config(self):
        """설정 JSON 불러오기"""
        if not PIPELINE_AVAILABLE:
            QMessageBox.warning(self, "경고", "pipeline 모듈이 없어 설정을 불러올 수 없습니다.")
            return
            
        path, _ = QFileDialog.getOpenFileName(
            self, "설정 불러오기", "", "JSON Files (*.json)"
        )
        if path:
            self.config = PipelineConfig.load(path)
            self._sync_gui_from_config()
            self._log(f"설정 로드: {path}")
            QMessageBox.information(self, "로드 완료", f"설정을 불러왔습니다:\n{path}")

    def cleanup(self):
        """리소스 정리 - MainWindow에서 호출"""
        if self.worker and self.worker.isRunning():
            self._log("파이프라인 중단 중...")
            self.worker.cancel()
            self.worker.wait(5000)  # 최대 5초 대기
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait(1000)


# ============================================================
# 단독 실행 테스트
# ============================================================
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = TrainingPage()
    window.setWindowTitle("Training Pipeline - Test")
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())
