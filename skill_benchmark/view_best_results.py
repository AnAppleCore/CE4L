#!/usr/bin/env python
"""
查看训练后的最优结果
用法: python view_best_results.py [checkpoint_dir]
"""

import torch
import os
import sys
import argparse
from pathlib import Path

def view_checkpoint(checkpoint_path):
    """查看 checkpoint 文件的内容"""
    if not os.path.exists(checkpoint_path):
        print(f"错误: 文件不存在 {checkpoint_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"查看文件: {checkpoint_path}")
    print(f"{'='*60}")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # 文件信息
        file_size = os.path.getsize(checkpoint_path) / 1024 / 1024
        print(f"\n文件大小: {file_size:.2f} MB")
        
        # 训练信息
        print(f"\n训练信息:")
        print(f"  - Epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"  - Best Accuracy: {ckpt.get('best_prec', 'N/A'):.4f} ({ckpt.get('best_prec', 0)*100:.2f}%)")
        
        # 模型参数信息
        print(f"\n模型参数:")
        for key in sorted(ckpt.keys()):
            if 'state_dict' in key:
                if isinstance(ckpt[key], tuple):
                    state_dict = ckpt[key][0]
                    param_count = sum(p.numel() for p in state_dict.values())
                    print(f"  - {key}:")
                    print(f"     参数层数: {len(state_dict.keys())}")
                    print(f"     参数量: {param_count:,}")
                else:
                    state_dict = ckpt[key]
                    param_count = sum(p.numel() for p in state_dict.values())
                    print(f"  - {key}:")
                    print(f"     参数层数: {len(state_dict.keys())}")
                    print(f"     参数量: {param_count:,}")
            else:
                print(f"  - {key}: {ckpt[key]}")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"错误: 无法加载文件 - {e}")

def find_best_models(base_dir='exps'):
    """查找所有最佳模型文件"""
    best_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == '_model_best.pth.tar':
                full_path = os.path.join(root, file)
                best_files.append(full_path)
    return best_files

def main():
    parser = argparse.ArgumentParser(description='查看训练后的最优结果')
    parser.add_argument('checkpoint_path', nargs='?', default=None,
                       help='Checkpoint 文件路径（可选，如果不提供则查找所有最佳模型）')
    parser.add_argument('--dir', type=str, default='exps',
                       help='搜索目录（默认: exps）')
    
    args = parser.parse_args()
    
    if args.checkpoint_path:
        # 查看指定的文件
        view_checkpoint(args.checkpoint_path)
    else:
        # 查找所有最佳模型
        print("查找所有最佳模型文件...")
        best_files = find_best_models(args.dir)
        
        if not best_files:
            print(f"\n未找到最佳模型文件（_model_best.pth.tar）")
            print(f"\n提示: 最佳模型文件通常保存在以下位置:")
            print(f"  - exps/checkpoints_aall_g100_tn_rn/_model_best.pth.tar (Baseline)")
            print(f"  - exps/checkpoints_aall_g100_tn_ry/_model_best.pth.tar (RN)")
            print(f"  - exps/checkpoints_aall_g100_ty_rn/_model_best.pth.tar (TL)")
            print(f"\n或者查看最新的 checkpoint 文件:")
            print(f"  - exps/checkpoints_*/_*_checkpoint.pth.tar")
        else:
            print(f"\n找到 {len(best_files)} 个最佳模型文件:\n")
            for i, f in enumerate(best_files, 1):
                print(f"{i}. {f}")
                view_checkpoint(f)

if __name__ == '__main__':
    main()

