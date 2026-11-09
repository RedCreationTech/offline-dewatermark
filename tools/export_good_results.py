"""导出处理效果好的水印图到zip文件。"""
from __future__ import annotations
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_good_results(results_dir: Path, threshold: int = 10000):
    """导出修改像素超过阈值的图片。
    
    Args:
        results_dir: 处理结果目录
        threshold: 修改像素阈值，默认10000
    """
    # 读取结果
    results_file = results_dir / '处理结果.json'
    data = json.loads(results_file.read_text())
    
    # 创建导出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_dir = results_dir / f'处理效果好_{timestamp}'
    export_dir.mkdir(exist_ok=True)
    
    # 统计信息
    stats = {'total': 0, 'exported': 0, 'by_style': {}}
    
    # 遍历所有结果
    for dir_name, results in data.items():
        for r in results:
            stats['total'] += 1
            
            # 检查是否达到阈值
            if r['changed_pixels'] >= threshold:
                stats['exported'] += 1
                
                # 从输入路径提取样式
                input_path = r['input']
                parts = input_path.split('/')
                if len(parts) >= 2:
                    style_dir = parts[0]
                    style_name = style_dir.split('_', 1)[-1] if '_' in style_dir else style_dir
                else:
                    style_name = 'unknown'
                
                # 统计按样式
                if style_name not in stats['by_style']:
                    stats['by_style'][style_name] = 0
                stats['by_style'][style_name] += 1
                
                # 创建样式子目录
                style_export_dir = export_dir / style_name
                style_export_dir.mkdir(exist_ok=True)
                
                # 复制输出图片
                output_file = results_dir / dir_name / r['output']
                if output_file.exists():
                    dest = style_export_dir / r['output']
                    shutil.copy2(output_file, dest)
                    
                    # 同时复制原始输入（用于对比）
                    input_file = ROOT / 'data/应用测试输入' / dir_name / r['input']
                    if input_file.exists():
                        input_dest = style_export_dir / f'原图_{r["output"]}'
                        shutil.copy2(input_file, input_dest)
    
    # 创建zip文件
    zip_path = results_dir / f'处理效果好_{timestamp}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in export_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(export_dir)
                zipf.write(file_path, arcname)
    
    # 生成说明文件
    readme_path = export_dir / '说明.txt'
    readme_content = f'''处理效果好的水印图导出
========================

导出时间: {timestamp}
筛选阈值: 修改像素 >= {threshold:,}

统计信息:
- 总文件数: {stats['total']}
- 导出文件数: {stats['exported']}
- 导出比例: {stats['exported']/stats['total']*100:.1f}%

按样式统计:
'''
    for style, count in sorted(stats['by_style'].items(), key=lambda x: x[1], reverse=True):
        readme_content += f'- {style}: {count}个文件\n'
    
    readme_content += f'''

文件结构:
- 每个样式一个子目录
- 每个目录包含:
  * 处理后的图片 (原文件名)
  * 原始输入图片 (前缀"原图_")

用途:
这些图片展示了算法对不同水印样式的处理效果，可用于:
- 向他人展示算法能力
- 对比不同样式的处理强度
- 测试和验证

生成工具: tools/export_good_results.py
'''
    
    readme_path.write_text(readme_content, encoding='utf-8')
    
    # 重新打包zip（包含说明文件）
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in export_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(export_dir)
                zipf.write(file_path, arcname)
    
    return zip_path, stats


def main():
    # 查找最新的结果目录
    base_dir = ROOT / 'out/批量测试'
    result_dirs = sorted(base_dir.glob('202*'), reverse=True)
    
    if not result_dirs:
        print('未找到处理结果目录')
        return
    
    results_dir = result_dirs[0]
    print(f'处理目录: {results_dir}')
    
    # 导出结果
    zip_path, stats = export_good_results(results_dir)
    
    print(f'\n=== 导出完成 ===')
    print(f'总文件数: {stats["total"]}')
    print(f'导出文件数: {stats["exported"]}')
    print(f'导出比例: {stats["exported"]/stats["total"]*100:.1f}%')
    print(f'\n按样式统计:')
    for style, count in sorted(stats['by_style'].items(), key=lambda x: x[1], reverse=True):
        print(f'  {style}: {count}个文件')
    print(f'\nZIP文件: {zip_path}')
    print(f'解压目录: {zip_path.with_suffix("")}')


if __name__ == '__main__':
    main()
