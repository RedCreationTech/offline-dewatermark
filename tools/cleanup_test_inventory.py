"""清理测试集：去除重复的 base_id+style 组合，保留合理样本。"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / 'data/应用测试输入'

def main():
    # 加载当前清单
    inventory_path = INPUT_DIR / '输入来源与校验.json'
    source = json.loads(inventory_path.read_text())
    
    # 统计各来源样本数
    print('=== 原始清单统计 ===')
    for subset in source['sets']:
        print(f"{subset['set']}: {len(subset['files'])} 个样本")
    
    # 收集所有样本并按 base_id+style 分组
    all_samples = []
    for subset in source['sets']:
        for item in subset['files']:
            all_samples.append({**item, '_source': subset['set']})
    
    groups = defaultdict(list)
    for sample in all_samples:
        # 从 id 提取 base_id（去掉路径前缀和样式后缀）
        sample_id = sample['id']
        style = sample['style']
        
        # 尝试从文件名提取 base_id
        if '/' in sample_id:
            filename = sample_id.split('/')[-1]
            # 移除扩展名
            base_name = filename.rsplit('.', 1)[0]
            # 移除样式后缀
            for suffix in ['__local_text', '__negative', '__shadow', '__tiled_irregular', 
                          '__dot_grid', '__heavy_dark', '__center_text', '__center_white',
                          '__cropped_edge', '__faint', '__gradient', '__line_net',
                          '__logo_opaque', '__logo_transparent', '__mixed_color',
                          '__multiple', '__outlined', '__signature', '__tiled_diagonal',
                          '__timestamp', '__tiny_corner', '__website']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            base_id = base_name
        else:
            base_id = sample_id
        
        key = (base_id, style)
        groups[key].append(sample)
    
    # 找出重复的组合
    duplicates = {k: v for k, v in groups.items() if len(v) > 1}
    
    print(f'\n=== 重复的 base_id+style 组合: {len(duplicates)} ===')
    for (base_id, style), samples in list(duplicates.items())[:10]:
        print(f'{base_id} + {style}: {len(samples)}个样本')
        for s in samples:
            print(f'  - {s["id"]} (来自 {s["_source"]})')
    
    # 去重策略：对每个重复组合，优先保留来自主测试目录的样本
    kept_samples = []
    removed_samples = []
    
    for key, samples in groups.items():
        if len(samples) == 1:
            kept_samples.append(samples[0])
        else:
            # 优先级：01_快速20张 > 02_人工增量110张 > 03_基础测试202张 > 其他
            def priority(s):
                source = s['_source']
                if '01_' in source:
                    return 0
                elif '02_' in source:
                    return 1
                elif '03_' in source:
                    return 2
                else:
                    return 3
            
            sorted_samples = sorted(samples, key=priority)
            kept_samples.append(sorted_samples[0])
            removed_samples.extend(sorted_samples[1:])
    
    print(f'\n=== 去重结果 ===')
    print(f'保留: {len(kept_samples)} 个样本')
    print(f'移除: {len(removed_samples)} 个重复样本')
    
    # 按来源重新组织
    new_sets = defaultdict(list)
    for sample in kept_samples:
        # 找到原始来源
        for subset in source['sets']:
            if any(f['id'] == sample['id'] for f in subset['files']):
                new_sets[subset['set']].append(sample)
                break
    
    # 生成新清单
    new_source = {
        'created_at': source['created_at'],
        'only_inputs': source['only_inputs'],
        'excluded': source.get('excluded', []),
        'sets': [],
        'deduplication_note': f'已去除{len(removed_samples)}个重复样本（同base_id+style），保留{len(kept_samples)}个唯一样本'
    }
    
    for set_name in ['01_快速20张', '02_人工增量110张', '03_基础测试202张']:
        if set_name in new_sets:
            files = new_sets[set_name]
            # 重新计算统计
            styles = defaultdict(int)
            negatives = 0
            for f in files:
                styles[f['style']] += 1
                if f['negative']:
                    negatives += 1
            
            new_source['sets'].append({
                'set': set_name,
                'images': len(files),
                'negative': negatives,
                'groups': dict(styles),
                'files': files
            })
    
    # 保存新清单
    output_path = INPUT_DIR / '输入来源与校验_去重.json'
    output_path.write_text(json.dumps(new_source, ensure_ascii=False, indent=2))
    print(f'\n新清单已保存: {output_path}')
    
    # 保存被移除的样本列表
    removed_path = INPUT_DIR / '已移除的重复样本.json'
    removed_path.write_text(json.dumps(removed_samples, ensure_ascii=False, indent=2))
    print(f'移除记录: {removed_path}')

if __name__ == '__main__':
    main()
