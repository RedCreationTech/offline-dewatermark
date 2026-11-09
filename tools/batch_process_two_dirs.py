"""批量处理指定目录的所有图片/PDF文件。"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wm import pipeline
from wm.contract import Thresholds


def process_directory(input_dir: Path, output_dir: Path):
    """处理单个目录下的所有文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有图片/PDF
    files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.pdf'):
        files.extend(input_dir.rglob(ext))
    
    files = sorted(files)
    print(f'发现 {len(files)} 个文件')
    
    # 使用桌面默认参数（笔画补全开启）
    thr = Thresholds(watermark_recall_priority=True)
    
    results = []
    for i, file_path in enumerate(files, 1):
        print(f'[{i}/{len(files)}] {file_path.relative_to(input_dir)}')
        
        try:
            # 处理文件
            processed = pipeline.process_file(file_path, dpi=200, thr=thr)
            
            # 保存结果
            for page_idx, result in enumerate(processed):
                if len(processed) > 1:
                    out_name = f'{file_path.stem}_p{page_idx+1}.png'
                else:
                    out_name = f'{file_path.stem}.png'
                
                out_path = output_dir / out_name
                Image.fromarray(result.output.image).save(out_path)
                
                # 记录结果
                mask = result.mask.binary()
                results.append({
                    'input': str(file_path.relative_to(input_dir)),
                    'output': str(out_path.relative_to(output_dir)),
                    'page': page_idx + 1,
                    'method': result.mask.method,
                    'selected_pixels': int(mask.sum()),
                    'changed_pixels': int(np.any(result.source.image != result.output.image, axis=2).sum()),
                })
        
        except Exception as e:
            print(f'  错误: {e}')
            results.append({
                'input': str(file_path.relative_to(input_dir)),
                'error': str(e),
            })
    
    return results


def main():
    input_root = ROOT / 'data/应用测试输入'
    output_root = ROOT / 'out/批量测试' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for dir_name in ['02_人工增量', '03_基础测试']:
        input_dir = input_root / dir_name
        if not input_dir.exists():
            print(f'目录不存在: {input_dir}')
            continue
        
        print(f'\n=== 处理 {dir_name} ===')
        output_dir = output_root / dir_name
        results = process_directory(input_dir, output_dir)
        all_results[dir_name] = results
    
    # 保存结果清单
    summary_path = output_root / '处理结果.json'
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    
    print(f'\n=== 完成 ===')
    print(f'输出目录: {output_root}')
    print(f'结果清单: {summary_path}')
    
    # 统计
    total = sum(len(v) for v in all_results.values())
    errors = sum(1 for v in all_results.values() for r in v if 'error' in r)
    print(f'总文件: {total}, 错误: {errors}')


if __name__ == '__main__':
    main()
