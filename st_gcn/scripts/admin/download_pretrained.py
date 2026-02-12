#!/usr/bin/env python3
"""
ST-GCN Pre-trained 모델 다운로드 및 분석 스크립트
PYSKL NTU RGB+D 60 HRNet (17 keypoints) 모델 사용
"""

import os
import sys
import urllib.request
import torch
import torch.nn as nn
from collections import OrderedDict

# ============================================================================
# 설정
# ============================================================================

# Pre-trained 모델 URL (PYSKL 공식)
PRETRAINED_MODELS = {
    'stgcn_ntu60_hrnet': {
        'url': 'https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j.pth',
        'filename': 'stgcn_ntu60_hrnet.pth',
        'description': 'ST-GCN NTU60 XSub HRNet 17kpts (기본)',
        'num_class': 60,
        'expected_acc': '86.6%'
    },
    'stgcnpp_ntu60_hrnet': {
        'url': 'https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_hrnet/j.pth',
        'filename': 'stgcnpp_ntu60_hrnet.pth',
        'description': 'ST-GCN++ NTU60 XSub HRNet 17kpts (향상된 버전)',
        'num_class': 60,
        'expected_acc': '89.1%'
    },
    'stgcn_ntu120_hrnet': {
        'url': 'https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/j.pth',
        'filename': 'stgcn_ntu120_hrnet.pth',
        'description': 'ST-GCN NTU120 XSub HRNet 17kpts (더 많은 클래스)',
        'num_class': 120,
        'expected_acc': '83.2%'
    }
}

# 저장 경로
SAVE_DIR = '/home/gjkong/dev_ws/st_gcn/pretrained'


def download_progress(block_num, block_size, total_size):
    """다운로드 진행률 표시"""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    bar_length = 50
    filled = int(bar_length * percent / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    sys.stdout.write(f'\r  [{bar}] {percent:.1f}%')
    sys.stdout.flush()


def download_model(model_key: str, save_dir: str = SAVE_DIR):
    """
    Pre-trained 모델 다운로드
    
    Args:
        model_key: 모델 키 (stgcn_ntu60_hrnet, stgcnpp_ntu60_hrnet, 등)
        save_dir: 저장 디렉토리
    """
    if model_key not in PRETRAINED_MODELS:
        print(f"[ERROR] Unknown model: {model_key}")
        print(f"Available models: {list(PRETRAINED_MODELS.keys())}")
        return None
    
    model_info = PRETRAINED_MODELS[model_key]
    url = model_info['url']
    filename = model_info['filename']
    filepath = os.path.join(save_dir, filename)
    
    # 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 이미 존재하는지 확인
    if os.path.exists(filepath):
        print(f"[INFO] Model already exists: {filepath}")
        return filepath
    
    print(f"\n{'='*60}")
    print(f"Downloading: {model_info['description']}")
    print(f"URL: {url}")
    print(f"Expected Accuracy: {model_info['expected_acc']}")
    print(f"{'='*60}")
    
    try:
        urllib.request.urlretrieve(url, filepath, download_progress)
        print(f"\n[OK] Downloaded: {filepath}")
        return filepath
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        return None


def analyze_checkpoint(filepath: str):
    """
    체크포인트 구조 분석
    
    Args:
        filepath: 체크포인트 파일 경로
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    # 로드
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    # 최상위 키
    print(f"\n[1] Top-level keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  - {key}: dict with {len(checkpoint[key])} items")
        elif isinstance(checkpoint[key], (int, float, str)):
            print(f"  - {key}: {checkpoint[key]}")
        else:
            print(f"  - {key}: {type(checkpoint[key]).__name__}")
    
    # State dict 분석
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"\n[2] State dict structure ({len(state_dict)} parameters):")
        
        # 레이어별 그룹화
        layers = {}
        for key in state_dict.keys():
            parts = key.split('.')
            layer_name = parts[0] if len(parts) > 0 else 'unknown'
            if layer_name not in layers:
                layers[layer_name] = []
            layers[layer_name].append(key)
        
        for layer_name, keys in layers.items():
            print(f"\n  [{layer_name}] ({len(keys)} params)")
            for key in keys[:5]:  # 처음 5개만 표시
                shape = tuple(state_dict[key].shape)
                print(f"    - {key}: {shape}")
            if len(keys) > 5:
                print(f"    ... and {len(keys)-5} more")
        
        # FC layer 확인
        print(f"\n[3] Classification head (FC layer):")
        fc_keys = [k for k in state_dict.keys() if 'fc' in k.lower() or 'cls' in k.lower()]
        for key in fc_keys:
            shape = tuple(state_dict[key].shape)
            print(f"  - {key}: {shape}")
        
        # 총 파라미터 수
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"\n[4] Total parameters: {total_params:,}")
        
        return checkpoint
    
    return checkpoint


def convert_for_fall_detection(checkpoint_path: str, output_path: str, num_classes: int = 2):
    """
    Pre-trained 체크포인트를 Fall Detection용으로 변환
    (FC layer를 제외한 backbone만 저장)
    
    Args:
        checkpoint_path: 원본 체크포인트 경로
        output_path: 출력 경로
        num_classes: 출력 클래스 수 (기본 2: Normal, Fall)
    """
    print(f"\n{'='*60}")
    print(f"Converting for Fall Detection ({num_classes} classes)")
    print(f"{'='*60}")
    
    # 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    # PYSKL 형식에서 backbone. prefix 제거
    new_state_dict = OrderedDict()
    excluded_keys = []
    
    for key, value in state_dict.items():
        # FC layer는 제외
        if 'fc' in key.lower() or 'cls_head' in key.lower():
            excluded_keys.append(key)
            continue
        
        # backbone. prefix 제거
        new_key = key
        if key.startswith('backbone.'):
            new_key = key.replace('backbone.', '')
        
        new_state_dict[new_key] = value
    
    print(f"\n[INFO] Excluded layers (will be re-initialized):")
    for key in excluded_keys:
        print(f"  - {key}")
    
    print(f"\n[INFO] Converted layers: {len(new_state_dict)}")
    
    # 새 체크포인트 저장
    converted = {
        'state_dict': new_state_dict,
        'meta': {
            'original': os.path.basename(checkpoint_path),
            'num_classes': num_classes,
            'description': 'Backbone only (FC layer excluded for fine-tuning)'
        }
    }
    
    torch.save(converted, output_path)
    print(f"\n[OK] Saved: {output_path}")
    
    return new_state_dict


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ST-GCN Pre-trained Model Downloader')
    parser.add_argument('--model', type=str, default='stgcn_ntu60_hrnet',
                        choices=list(PRETRAINED_MODELS.keys()),
                        help='Model to download')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze checkpoint structure')
    parser.add_argument('--convert', action='store_true',
                        help='Convert for fall detection')
    parser.add_argument('--list', action='store_true',
                        help='List available models')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR,
                        help='Directory to save models')
    
    args = parser.parse_args()
    
    # 모델 목록 출력
    if args.list:
        print("\n" + "="*60)
        print("Available Pre-trained Models")
        print("="*60)
        for key, info in PRETRAINED_MODELS.items():
            print(f"\n[{key}]")
            print(f"  Description: {info['description']}")
            print(f"  Classes: {info['num_class']}")
            print(f"  Expected Acc: {info['expected_acc']}")
        return
    
    # 다운로드
    filepath = download_model(args.model, args.save_dir)
    if filepath is None:
        return
    
    # 분석
    if args.analyze:
        analyze_checkpoint(filepath)
    
    # 변환
    if args.convert:
        output_path = filepath.replace('.pth', '_backbone.pth')
        convert_for_fall_detection(filepath, output_path)


if __name__ == '__main__':
    # 단독 실행 시 기본 동작
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("ST-GCN Pre-trained Model Downloader & Analyzer")
        print("="*60)
        print("\nUsage:")
        print("  python download_pretrained.py --list              # 모델 목록 보기")
        print("  python download_pretrained.py --model stgcn_ntu60_hrnet --analyze")
        print("  python download_pretrained.py --model stgcn_ntu60_hrnet --convert")
        print("\nDefault: Download and analyze stgcn_ntu60_hrnet")
        print("-"*60)
        
        # 기본 실행
        filepath = download_model('stgcn_ntu60_hrnet')
        if filepath:
            analyze_checkpoint(filepath)
    else:
        main()
