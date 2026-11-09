"""分析批量处理结果并生成可视化报告。"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def analyze_results(results_dir: Path):
    """分析处理结果JSON。"""
    results_file = results_dir / '处理结果.json'
    data = json.loads(results_file.read_text())
    
    # 统计信息
    stats = {
        'total_files': 0,
        'by_directory': defaultdict(lambda: {'count': 0, 'methods': defaultdict(int), 
                                             'total_selected': 0, 'total_changed': 0}),
        'by_style': defaultdict(lambda: {'count': 0, 'methods': defaultdict(int),
                                        'total_selected': 0, 'total_changed': 0}),
        'by_method': defaultdict(int),
        'zero_change_count': 0,
    }
    
    for dir_name, results in data.items():
        for r in results:
            stats['total_files'] += 1
            stats['by_directory'][dir_name]['count'] += 1
            stats['by_directory'][dir_name]['total_selected'] += r['selected_pixels']
            stats['by_directory'][dir_name]['total_changed'] += r['changed_pixels']
            
            method = r['method']
            stats['by_directory'][dir_name]['methods'][method] += 1
            stats['by_method'][method] += 1
            
            # 从文件名提取样式
            input_path = r['input']
            style = input_path.split('/')[0].split('_', 1)[-1] if '_' in input_path.split('/')[0] else input_path.split('/')[0]
            stats['by_style'][style]['count'] += 1
            stats['by_style'][style]['methods'][method] += 1
            stats['by_style'][style]['total_selected'] += r['selected_pixels']
            stats['by_style'][style]['total_changed'] += r['changed_pixels']
            
            if r['changed_pixels'] == 0:
                stats['zero_change_count'] += 1
    
    return stats


def generate_html_report(stats: dict, results_dir: Path):
    """生成HTML可视化报告。"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 按样式排序
    sorted_styles = sorted(stats['by_style'].items(), key=lambda x: x[1]['count'], reverse=True)
    
    # 生成样式表格行
    style_rows = ''
    for style, data in sorted_styles:
        methods_str = ', '.join(f'{m}: {c}' for m, c in sorted(data['methods'].items()))
        avg_selected = data['total_selected'] / data['count'] if data['count'] > 0 else 0
        avg_changed = data['total_changed'] / data['count'] if data['count'] > 0 else 0
        
        # 根据处理效果标记颜色
        if avg_changed == 0:
            status_class = 'status-none'
            status_text = '未处理'
        elif avg_changed < 10000:
            status_class = 'status-low'
            status_text = '轻微'
        elif avg_changed < 50000:
            status_class = 'status-medium'
            status_text = '中等'
        else:
            status_class = 'status-high'
            status_text = '强烈'
        
        style_rows += f'''
        <tr>
            <td>{style}</td>
            <td>{data['count']}</td>
            <td><span class="{status_class}">{status_text}</span></td>
            <td>{avg_selected:,.0f}</td>
            <td>{avg_changed:,.0f}</td>
            <td style="font-size: 0.85em;">{methods_str}</td>
        </tr>'''
    
    # 生成方法统计
    method_rows = ''
    for method, count in sorted(stats['by_method'].items(), key=lambda x: x[1], reverse=True):
        percentage = count / stats['total_files'] * 100
        method_rows += f'<tr><td>{method}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>'
    
    # 生成目录统计
    dir_rows = ''
    for dir_name, data in stats['by_directory'].items():
        methods_str = ', '.join(f'{m}: {c}' for m, c in sorted(data['methods'].items()))
        dir_rows += f'''
        <tr>
            <td>{dir_name}</td>
            <td>{data['count']}</td>
            <td>{data['total_selected']:,}</td>
            <td>{data['total_changed']:,}</td>
            <td style="font-size: 0.85em;">{methods_str}</td>
        </tr>'''
    
    html = f'''<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>批量处理结果分析 {timestamp}</title>
    <style>
        body {{
            font: 15px/1.8 system-ui, -apple-system, sans-serif;
            max-width: 1400px;
            margin: 30px auto;
            padding: 25px;
            color: #243d33;
            background: #f3f6f3;
        }}
        h1 {{
            color: #1a5c3a;
            border-bottom: 3px solid #2d8659;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #1a5c3a;
            margin-top: 30px;
            border-left: 4px solid #2d8659;
            padding-left: 12px;
        }}
        .summary {{
            background: white;
            border: 1px solid #dae4ef;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .stat-box {{
            background: #f8faf8;
            border: 1px solid #d0e0d0;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2d8659;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background: #2d8659;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        tr:hover {{
            background: #f0f5f0;
        }}
        .status-none {{
            background: #e0e0e0;
            color: #666;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .status-low {{
            background: #fff4e0;
            color: #b8860b;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .status-medium {{
            background: #ffe8cc;
            color: #d2691e;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .status-high {{
            background: #ffd6cc;
            color: #cd5c5c;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .note {{
            background: #fff9e6;
            border-left: 4px solid #ffa500;
            padding: 12px;
            margin: 15px 0;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>批量处理结果分析报告</h1>
    <p><strong>生成时间：</strong>{timestamp}</p>
    <p><strong>结果目录：</strong><code>{results_dir}</code></p>
    
    <div class="summary">
        <h2>总体统计</h2>
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-value">{stats['total_files']}</div>
                <div class="stat-label">总文件数</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats['zero_change_count']}</div>
                <div class="stat-label">未修改文件</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(stats['by_style'])}</div>
                <div class="stat-label">水印样式种类</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(stats['by_method'])}</div>
                <div class="stat-label">处理方法种类</div>
            </div>
        </div>
    </div>
    
    <h2>按水印样式统计</h2>
    <table>
        <thead>
            <tr>
                <th>水印样式</th>
                <th>文件数</th>
                <th>处理强度</th>
                <th>平均选中像素</th>
                <th>平均修改像素</th>
                <th>使用方法</th>
            </tr>
        </thead>
        <tbody>{style_rows}</tbody>
    </table>
    
    <div class="note">
        <strong>处理强度说明：</strong><br>
        <span class="status-none">未处理</span> 修改像素为0，未检测到水印或未处理<br>
        <span class="status-low">轻微</span> 平均修改像素 &lt; 10,000<br>
        <span class="status-medium">中等</span> 平均修改像素 10,000 - 50,000<br>
        <span class="status-high">强烈</span> 平均修改像素 &gt; 50,000
    </div>
    
    <h2>按处理方法统计</h2>
    <table>
        <thead>
            <tr>
                <th>处理方法</th>
                <th>使用次数</th>
                <th>占比</th>
            </tr>
        </thead>
        <tbody>{method_rows}</tbody>
    </table>
    
    <h2>按目录统计</h2>
    <table>
        <thead>
            <tr>
                <th>目录</th>
                <th>文件数</th>
                <th>总选中像素</th>
                <th>总修改像素</th>
                <th>使用方法分布</th>
            </tr>
        </thead>
        <tbody>{dir_rows}</tbody>
    </table>
    
    <div class="note">
        <strong>说明：</strong>本报告基于处理结果JSON自动生成，展示各水印样式的处理效果分布。
        "处理强度"反映算法对该样式水印的响应程度，但不直接等同于去水印成功率。
    </div>
</body>
</html>'''
    
    report_path = results_dir / '处理结果分析报告.html'
    report_path.write_text(html, encoding='utf-8')
    return report_path


def main():
    # 查找最新的结果目录
    base_dir = ROOT / 'out/批量测试'
    result_dirs = sorted(base_dir.glob('202*'), reverse=True)
    
    if not result_dirs:
        print('未找到处理结果目录')
        return
    
    results_dir = result_dirs[0]
    print(f'分析目录: {results_dir}')
    
    # 分析结果
    stats = analyze_results(results_dir)
    
    # 生成报告
    report_path = generate_html_report(stats, results_dir)
    print(f'报告已生成: {report_path}')
    
    # 打印摘要
    print(f'\n=== 摘要 ===')
    print(f'总文件数: {stats["total_files"]}')
    print(f'未修改文件: {stats["zero_change_count"]}')
    print(f'水印样式种类: {len(stats["by_style"])}')
    print(f'处理方法种类: {len(stats["by_method"])}')
    
    print(f'\n=== 处理强度最高的5种样式 ===')
    sorted_by_intensity = sorted(
        stats['by_style'].items(),
        key=lambda x: x[1]['total_changed'] / x[1]['count'] if x[1]['count'] > 0 else 0,
        reverse=True
    )
    for style, data in sorted_by_intensity[:5]:
        avg_changed = data['total_changed'] / data['count'] if data['count'] > 0 else 0
        print(f'  {style}: 平均修改 {avg_changed:,.0f} 像素 ({data["count"]}个文件)')
    
    print(f'\n=== 未处理的样式（修改像素为0） ===')
    zero_styles = [s for s, d in stats['by_style'].items() if d['total_changed'] == 0]
    if zero_styles:
        for style in zero_styles:
            print(f'  {style}')
    else:
        print('  无')


if __name__ == '__main__':
    main()
