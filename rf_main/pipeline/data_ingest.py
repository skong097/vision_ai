#!/usr/bin/env python3
"""
============================================================
Home Safe Solution - Data Ingest Module (Stage 1)
============================================================
외부 동영상 데이터를 수집하여 학습 데이터로 준비합니다.

지원 소스:
  - YouTube URL (yt-dlp 사용)
  - HTTP/HTTPS URL (requests 사용)
  - 로컬 파일/폴더

라벨링 전략:
  - folder: fall/, normal/ 폴더 구조
  - filename: fall-01.mp4, normal-03.mp4 (접두어 기반)
  - csv: manifest.csv 파일 참조
  - manual: 소스 등록 시 명시적 라벨 지정

사용법:
    from pipeline.data_ingest import DataIngestEngine
    from pipeline.config import DataIngestConfig
    
    config = DataIngestConfig()
    engine = DataIngestEngine(config)
    
    # 소스 추가
    engine.add_youtube("https://youtube.com/watch?v=...", "fall")
    engine.add_local("/path/to/video.mp4", "normal")
    engine.add_folder("/path/to/videos/", label_strategy="folder")
    
    # 다운로드 실행
    result = engine.process_all()
    print(result)

CLI 사용:
    python data_ingest.py --folder ./videos --strategy folder
    python data_ingest.py --youtube "https://..." --label fall
============================================================
"""

import os
import re
import csv
import shutil
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Callable
from datetime import datetime

# config.py에서 설정 가져오기
try:
    from pipeline.config import DataIngestConfig, SUPPORTED_VIDEO_FORMATS
except ImportError:
    # 단독 실행 시
    from config import DataIngestConfig, SUPPORTED_VIDEO_FORMATS

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 데이터 소스 클래스
# ============================================================
@dataclass
class DataSource:
    """
    개별 데이터 소스 정보
    
    Attributes:
        source_type: 소스 타입 (youtube / url / local)
        path: URL 또는 파일 경로
        label: 라벨 (fall / normal)
        status: 처리 상태 (pending / downloading / done / error)
        output_path: 다운로드/복사 후 저장 경로
        error_msg: 에러 발생 시 메시지
        file_size_mb: 파일 크기 (MB)
        duration_sec: 비디오 길이 (초)
    """
    source_type: str         # youtube / url / local
    path: str                # URL 또는 파일 경로
    label: str               # fall / normal
    status: str = "pending"  # pending / downloading / done / error
    output_path: str = ""    # 다운로드 후 저장 경로
    error_msg: str = ""
    file_size_mb: float = 0.0
    duration_sec: float = 0.0
    
    def __str__(self):
        return f"[{self.source_type}] {self.path[:50]}... → {self.label} ({self.status})"


# ============================================================
# 데이터 수집 엔진
# ============================================================
class DataIngestEngine:
    """
    데이터 수집 엔진
    
    YouTube, URL, 로컬 파일에서 비디오를 수집하여
    fall/, normal/ 구조의 데이터셋 디렉토리를 구성합니다.
    
    Example:
        config = DataIngestConfig()
        engine = DataIngestEngine(config)
        
        # 다양한 소스 추가
        engine.add_youtube("https://youtube.com/watch?v=xxx", "fall")
        engine.add_url("https://example.com/video.mp4", "normal")
        engine.add_local("/path/to/video.mp4", "fall")
        engine.add_folder("/path/to/videos/", "folder")
        
        # 처리 실행
        result = engine.process_all()
    """
    
    def __init__(
        self,
        config: DataIngestConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """
        Args:
            config: 데이터 수집 설정
            progress_callback: 진행 콜백 함수 (current, total, message)
        """
        self.config = config
        self.progress_callback = progress_callback
        self.sources: List[DataSource] = []
        self.raw_dir = Path(config.raw_video_dir)
        
        # 출력 디렉토리 생성
        (self.raw_dir / "fall").mkdir(parents=True, exist_ok=True)
        (self.raw_dir / "normal").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"데이터 저장 경로: {self.raw_dir}")
    
    # ================================================================
    # 소스 등록 메서드
    # ================================================================
    
    def add_source(self, source_type: str, path: str, label: str) -> DataSource:
        """
        데이터 소스 추가 (범용)
        
        Args:
            source_type: "youtube" / "url" / "local"
            path: URL 또는 파일 경로
            label: "fall" / "normal"
            
        Returns:
            생성된 DataSource 객체
        """
        # 라벨 검증
        label = label.lower().strip()
        if label not in ("fall", "normal"):
            logger.warning(f"알 수 없는 라벨 '{label}' → 'normal'로 설정")
            label = "normal"
        
        source = DataSource(
            source_type=source_type.lower(),
            path=path.strip(),
            label=label,
        )
        self.sources.append(source)
        logger.info(f"소스 추가: {source}")
        return source
    
    def add_youtube(self, url: str, label: str) -> DataSource:
        """YouTube URL 추가"""
        return self.add_source("youtube", url, label)
    
    def add_url(self, url: str, label: str) -> DataSource:
        """HTTP/HTTPS URL 추가"""
        return self.add_source("url", url, label)
    
    def add_local(self, path: str, label: str) -> DataSource:
        """로컬 파일 추가"""
        return self.add_source("local", path, label)
    
    def add_folder(self, folder_path: str, label_strategy: str = "folder") -> int:
        """
        폴더 내 비디오 일괄 등록
        
        Args:
            folder_path: 비디오가 있는 폴더 경로
            label_strategy: 라벨 결정 방식
                - "folder": fall/, normal/ 하위 폴더 구조
                - "filename": 파일명 접두어 (fall-01.mp4, normal-02.mp4)
                
        Returns:
            등록된 소스 수
        """
        folder = Path(folder_path)
        if not folder.exists():
            logger.error(f"폴더 없음: {folder_path}")
            return 0
        
        added_count = 0
        
        if label_strategy == "folder":
            # fall/, normal/ 하위 폴더 구조
            for label_name in ["fall", "normal"]:
                sub_dir = folder / label_name
                if sub_dir.exists():
                    for f in sorted(sub_dir.iterdir()):
                        if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                            self.add_local(str(f), label_name)
                            added_count += 1
        
        elif label_strategy == "filename":
            # 파일명 접두어로 라벨 추정
            for f in sorted(folder.iterdir()):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
                    continue
                
                name_lower = f.stem.lower()
                if name_lower.startswith("fall"):
                    self.add_local(str(f), "fall")
                    added_count += 1
                elif name_lower.startswith("normal") or name_lower.startswith("adl"):
                    self.add_local(str(f), "normal")
                    added_count += 1
                else:
                    logger.warning(f"라벨 추정 불가, 건너뜀: {f.name}")
        
        else:
            # 폴더 내 모든 파일을 지정된 라벨로 (label_strategy를 라벨로 사용)
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                    self.add_local(str(f), label_strategy)
                    added_count += 1
        
        logger.info(f"폴더 등록 완료: {added_count}개 비디오 ({folder_path})")
        return added_count
    
    def add_from_csv(self, csv_path: str) -> int:
        """
        CSV manifest 파일에서 소스 등록
        
        CSV 형식:
            path,label,type (type은 선택)
            https://youtube.com/watch?v=xxx,fall,youtube
            /path/to/video.mp4,normal,local
            
        Args:
            csv_path: CSV 파일 경로
            
        Returns:
            등록된 소스 수
        """
        added_count = 0
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 경로 컬럼 찾기
                path = row.get("path", row.get("url", row.get("file", "")))
                label = row.get("label", row.get("class", ""))
                source_type = row.get("type", "local")
                
                if not path or not label:
                    continue
                
                # 소스 타입 자동 판단
                if "youtube.com" in path or "youtu.be" in path:
                    source_type = "youtube"
                elif path.startswith("http"):
                    source_type = "url"
                else:
                    source_type = "local"
                
                self.add_source(source_type, path, label)
                added_count += 1
        
        logger.info(f"CSV 등록 완료: {added_count}개 소스 ({csv_path})")
        return added_count
    
    # ================================================================
    # 처리 실행
    # ================================================================
    
    def process_all(self) -> dict:
        """
        모든 등록된 소스 처리 (다운로드/복사)
        
        Returns:
            결과 딕셔너리:
            {
                "success": 성공 수,
                "error": 실패 수,
                "total": 전체 수,
                "files": {
                    "fall_count": fall 비디오 수,
                    "normal_count": normal 비디오 수,
                    ...
                }
            }
        """
        total = len(self.sources)
        results = {"success": 0, "error": 0, "total": total}
        
        if total == 0:
            logger.warning("등록된 소스가 없습니다")
            results["files"] = self._get_file_summary()
            return results
        
        logger.info(f"{'='*60}")
        logger.info(f"  데이터 수집 시작: {total}개 소스")
        logger.info(f"{'='*60}")
        
        for idx, source in enumerate(self.sources):
            try:
                self._emit_progress(idx, total, f"처리 중: {source.path[:50]}...")
                source.status = "downloading"
                
                # 소스 타입별 처리
                if source.source_type == "youtube":
                    self._download_youtube(source)
                elif source.source_type == "url":
                    self._download_url(source)
                elif source.source_type == "local":
                    self._copy_local(source)
                else:
                    raise ValueError(f"알 수 없는 소스 타입: {source.source_type}")
                
                source.status = "done"
                results["success"] += 1
                logger.info(f"  ✅ [{idx+1}/{total}] {Path(source.output_path).name}")
                
            except Exception as e:
                source.status = "error"
                source.error_msg = str(e)
                results["error"] += 1
                logger.error(f"  ❌ [{idx+1}/{total}] {source.path[:50]}: {e}")
        
        self._emit_progress(total, total, "데이터 수집 완료")
        results["files"] = self._get_file_summary()
        
        # 결과 요약
        logger.info(f"{'='*60}")
        logger.info(f"  수집 완료: 성공={results['success']}, 실패={results['error']}")
        logger.info(f"  Fall: {results['files']['fall_count']}개, "
                   f"Normal: {results['files']['normal_count']}개")
        logger.info(f"{'='*60}")
        
        return results
    
    # ================================================================
    # 소스 타입별 처리
    # ================================================================
    
    def _download_youtube(self, source: DataSource) -> None:
        """
        YouTube 비디오 다운로드 (yt-dlp 사용)
        
        Args:
            source: DataSource 객체 (path에 YouTube URL)
        """
        # yt-dlp 설치 확인
        if not self._check_command("yt-dlp"):
            raise RuntimeError("yt-dlp가 설치되어 있지 않습니다. pip install yt-dlp")
        
        output_dir = self.raw_dir / source.label
        
        # 출력 템플릿: 제목(50자)_ID.확장자
        output_template = str(output_dir / "%(title).50s_%(id)s.%(ext)s")
        
        cmd = [
            "yt-dlp",
            "--format", self.config.youtube_format,
            "--output", output_template,
            "--merge-output-format", "mp4",
            "--no-playlist",  # 단일 비디오만 (재생목록 X)
            "--no-overwrites",
            "--restrict-filenames",  # 파일명 안전 문자만
        ]
        
        # 파일 크기 제한
        if self.config.max_file_size_mb > 0:
            cmd.extend(["--max-filesize", f"{self.config.max_file_size_mb}M"])
        
        # 길이 제한
        if self.config.youtube_max_duration > 0:
            cmd.extend([
                "--match-filter",
                f"duration<={self.config.youtube_max_duration}"
            ])
        
        cmd.append(source.path)
        
        logger.debug(f"실행: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.download_timeout * 2,
        )
        
        if result.returncode != 0:
            error_msg = result.stderr[:300] if result.stderr else "Unknown error"
            raise RuntimeError(f"yt-dlp 오류: {error_msg}")
        
        # 다운로드된 파일 찾기
        downloaded = self._find_latest_file(output_dir)
        if downloaded:
            source.output_path = str(downloaded)
            source.file_size_mb = downloaded.stat().st_size / (1024 * 1024)
        else:
            raise FileNotFoundError("다운로드된 파일을 찾을 수 없습니다")
    
    def _download_url(self, source: DataSource) -> None:
        """
        HTTP/HTTPS URL에서 비디오 다운로드
        
        Args:
            source: DataSource 객체 (path에 URL)
        """
        try:
            import requests
        except ImportError:
            raise RuntimeError("requests가 설치되어 있지 않습니다. pip install requests")
        
        output_dir = self.raw_dir / source.label
        
        # 파일명 추출
        filename = self._extract_filename_from_url(source.path)
        if not filename:
            # URL에서 파일명 추출 실패 시 해시 기반 이름 생성
            url_hash = abs(hash(source.path)) % 100000
            filename = f"video_{url_hash:05d}.mp4"
        
        output_path = output_dir / filename
        
        # 중복 방지
        output_path = self._get_unique_path(output_path)
        
        # 다운로드
        logger.debug(f"다운로드 시작: {source.path}")
        
        response = requests.get(
            source.path,
            stream=True,
            timeout=self.config.download_timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        
        # Content-Length 확인
        content_length = int(response.headers.get("content-length", 0))
        max_size = self.config.max_file_size_mb * 1024 * 1024
        
        if content_length > max_size > 0:
            raise ValueError(
                f"파일 크기 초과: {content_length / 1024 / 1024:.1f}MB > "
                f"{self.config.max_file_size_mb}MB"
            )
        
        # 파일 저장
        with open(str(output_path), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        source.output_path = str(output_path)
        source.file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    def _copy_local(self, source: DataSource) -> None:
        """
        로컬 파일을 데이터셋 디렉토리로 복사
        
        Args:
            source: DataSource 객체 (path에 로컬 경로)
        """
        src = Path(source.path)
        
        if not src.exists():
            raise FileNotFoundError(f"파일 없음: {source.path}")
        
        if src.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
            raise ValueError(f"지원하지 않는 형식: {src.suffix}")
        
        output_dir = self.raw_dir / source.label
        dest = output_dir / src.name
        
        # 이미 같은 경로면 건너뜀
        if src.resolve() == dest.resolve():
            source.output_path = str(dest)
            source.file_size_mb = dest.stat().st_size / (1024 * 1024)
            logger.debug(f"이미 대상 경로에 있음: {dest}")
            return
        
        # 중복 방지
        dest = self._get_unique_path(dest)
        
        # 복사
        shutil.copy2(str(src), str(dest))
        
        source.output_path = str(dest)
        source.file_size_mb = dest.stat().st_size / (1024 * 1024)
    
    # ================================================================
    # 유틸리티 메서드
    # ================================================================
    
    def _check_command(self, cmd: str) -> bool:
        """명령어 존재 여부 확인"""
        try:
            subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _extract_filename_from_url(self, url: str) -> Optional[str]:
        """URL에서 파일명 추출"""
        from urllib.parse import urlparse, unquote
        
        parsed = urlparse(url)
        path = unquote(parsed.path)
        name = Path(path).name
        
        if Path(name).suffix.lower() in SUPPORTED_VIDEO_FORMATS:
            return name
        return None
    
    def _find_latest_file(self, directory: Path) -> Optional[Path]:
        """디렉토리에서 가장 최근 수정된 비디오 파일 반환"""
        files = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_FORMATS
        ]
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)
    
    def _get_unique_path(self, path: Path) -> Path:
        """중복 방지: 파일이 존재하면 _1, _2 등 붙여서 반환"""
        if not path.exists():
            return path
        
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        
        while path.exists():
            path = parent / f"{stem}_{counter}{suffix}"
            counter += 1
        
        return path
    
    def _get_file_summary(self) -> dict:
        """수집된 파일 요약"""
        fall_dir = self.raw_dir / "fall"
        normal_dir = self.raw_dir / "normal"
        
        fall_files = [
            f for f in fall_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_FORMATS
        ] if fall_dir.exists() else []
        
        normal_files = [
            f for f in normal_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_FORMATS
        ] if normal_dir.exists() else []
        
        return {
            "fall_count": len(fall_files),
            "normal_count": len(normal_files),
            "total_count": len(fall_files) + len(normal_files),
            "fall_size_mb": sum(f.stat().st_size for f in fall_files) / (1024 * 1024),
            "normal_size_mb": sum(f.stat().st_size for f in normal_files) / (1024 * 1024),
            "fall_dir": str(fall_dir),
            "normal_dir": str(normal_dir),
        }
    
    def _emit_progress(self, current: int, total: int, message: str) -> None:
        """진행 콜백 호출"""
        if self.progress_callback:
            self.progress_callback(current, total, message)
    
    # ================================================================
    # 상태 조회
    # ================================================================
    
    def get_summary(self) -> str:
        """현재 소스 요약 문자열"""
        by_label = {"fall": 0, "normal": 0}
        by_type = {"youtube": 0, "url": 0, "local": 0}
        by_status = {"pending": 0, "downloading": 0, "done": 0, "error": 0}
        
        for s in self.sources:
            by_label[s.label] = by_label.get(s.label, 0) + 1
            by_type[s.source_type] = by_type.get(s.source_type, 0) + 1
            by_status[s.status] = by_status.get(s.status, 0) + 1
        
        lines = [
            f"총 소스: {len(self.sources)}개",
            f"  라벨: Fall={by_label.get('fall', 0)}, Normal={by_label.get('normal', 0)}",
            f"  타입: YouTube={by_type['youtube']}, URL={by_type['url']}, Local={by_type['local']}",
            f"  상태: 완료={by_status['done']}, 대기={by_status['pending']}, 오류={by_status['error']}",
        ]
        return "\n".join(lines)
    
    def get_sources(self) -> List[DataSource]:
        """등록된 모든 소스 반환"""
        return self.sources
    
    def clear_sources(self) -> None:
        """등록된 소스 모두 삭제"""
        self.sources.clear()
        logger.info("모든 소스 삭제됨")


# ============================================================
# CLI 실행
# ============================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Home Safe Solution - 데이터 수집 모듈",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 폴더에서 fall/normal 구조로 등록
  python data_ingest.py --folder ./videos --strategy folder
  
  # YouTube 비디오 다운로드
  python data_ingest.py --youtube "https://youtube.com/watch?v=xxx" --label fall
  
  # CSV manifest에서 등록
  python data_ingest.py --csv manifest.csv
  
  # 로컬 파일 등록
  python data_ingest.py --local ./video.mp4 --label normal
        """
    )
    
    parser.add_argument("--folder", help="비디오 폴더 경로")
    parser.add_argument("--strategy", default="folder",
                        choices=["folder", "filename"],
                        help="폴더 라벨 전략 (default: folder)")
    parser.add_argument("--csv", help="CSV manifest 경로")
    parser.add_argument("--youtube", help="YouTube URL")
    parser.add_argument("--url", help="HTTP/HTTPS URL")
    parser.add_argument("--local", help="로컬 파일 경로")
    parser.add_argument("--label", default="normal",
                        choices=["fall", "normal"],
                        help="라벨 (--youtube, --url, --local용)")
    parser.add_argument("--output-dir", default=None,
                        help="출력 디렉토리 (기본: dataset/raw_videos)")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 다운로드 없이 소스 등록만")
    
    args = parser.parse_args()
    
    # 설정 생성
    config = DataIngestConfig()
    if args.output_dir:
        config.raw_video_dir = args.output_dir
    
    # 엔진 생성
    engine = DataIngestEngine(
        config,
        progress_callback=lambda c, t, m: print(f"  [{c}/{t}] {m}"),
    )
    
    # 소스 등록
    if args.folder:
        engine.add_folder(args.folder, args.strategy)
    if args.csv:
        engine.add_from_csv(args.csv)
    if args.youtube:
        engine.add_youtube(args.youtube, args.label)
    if args.url:
        engine.add_url(args.url, args.label)
    if args.local:
        engine.add_local(args.local, args.label)
    
    # 요약 출력
    print("\n" + engine.get_summary())
    
    # 실행
    if not args.dry_run and engine.sources:
        print("\n")
        results = engine.process_all()
        print(f"\n최종 결과: {results['files']}")
    elif args.dry_run:
        print("\n(--dry-run: 다운로드 건너뜀)")


if __name__ == "__main__":
    main()
