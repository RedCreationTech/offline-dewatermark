"""基线/防退化控制轮的双报告生成器（固定模板 v2）。

每次 freeze / test / test-release 跑完生成两份：
- 摘要报告（可发送单文件）：结论置顶 + 数字看板 + 处理清单 + 分层总表，
  样本图全部 base64 内嵌，命名带标题+日期+时间。
- 明细报告（本地工具）：结论置顶 + 三模式 tab + 搜索/筛选 + 效果三档分卡（可点击过滤）
  + 同图重复批次合并（默认显示较差档）+ 逐页展开，图片保留相对路径
  （AGENTS.md §8 对全量逐页明细的 base64 豁免）。

口径（数据驱动，两份报告一致）：
- 分档只看：运行错误、无水印对照是否误选改动、笔画召回率、非水印区误差。
  问题较多 = 笔画召回率<30% 或 负例有改动 或 运行错误；
  效果较好 = 召回率≥60% 且 非水印区误差≤1.0 且 评分完整；其余为 一般/待复核。
- 多选背景不作为分档指标：它仍逐张显示在卡片上（绝对像素+占全图百分比），供目视参考。
- 术语浮层词表见 GLOSSARY（点击"?"弹白话解释，可收起）。

调用：build_both(out_dir)；regression_control.write_report 在写出旧版报告后调用。
历史封存不追溯重渲染。
"""
from __future__ import annotations
import base64
import html
import io
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote

from PIL import Image

# 术语词表：专业词 -> 白话解释。新词直接加在这里，两份报告自动生效。
GLOSSARY = {
    '笔画召回率': '真实水印笔画里被选中的比例。100% = 一笔没漏；数字越低漏得越多。分母含淡笔画，不能用"淡的看不见"掩饰漏检。',
    '明显笔画召回': '只统计不透明度较高（看得清）的笔画召回率，比"笔画召回率"更宽松一点，两者一起看能判断淡边缘的丢失程度。',
    '选区精确率': '程序选中的像素里，真是水印的比例。低 = 把背景或正文也圈进来修了。',
    '漏选': '真水印笔画里没被程序选中的像素，蓝色标记。漏选越多，修完残留越多。',
    '多选': '程序选中但其实不是水印的像素（背景/正文），橙色标记。按占全图百分比判定严重程度，大小图同一标准。',
    '页数': '该水印类型在本次测试中实际运行的页数。同一张图按页计数；本表只统计桌面默认模式。',
    '均笔画召回率': '该类型所有页“笔画召回率”的平均值。笔画召回率＝选中的水印笔画像素 ÷ 全部水印笔画像素（含淡边缘），越高表示水印选得越全。',
    '均多选占比': '该类型所有页“多选背景占全图面积”的平均百分比。多选占比＝误选的非水印像素 ÷ 全图总像素，越高表示误伤背景/正文越多。',
    '均精确率': '该类型所有页“选区精确率”的平均值。精确率＝选区里真正属于水印的像素 ÷ 全部选中像素，越高表示选得越准、误伤越少。',
    '问题较多占比': '该类型中被自动分进“问题较多”档的页数 ÷ 该类型总页数。分档只看运行错误、负例改动、笔画召回率、非水印区误差；多选背景不参与分档。',
    'tonal': '修复通道名：按颜色/明暗连续调性（tonal）做修复的处理路径，适合渐变色、明暗渐变类水印。通道名是程序内部定位方式的标识。',
    '字洞': '字的内部空洞、被笔画圈死的缝隙。选区把它们也填上 = 糊成实心块，属于误伤的一种。',
    '选区占比': '选中像素占整张图的比例，衡量修复范围大小。',
    '参考误差': '输出图与干净参考图在水印区域内的平均像素差（MAE）。是重建差异，不等于水印残留率。',
    '非水印区误差': '非水印区域的平均像素差。理想应为 0——不为 0 说明好地方被改动了。',
    '页次': '一张输入图片算 1 页次；多页 PDF 每页各算 1 页次。三种模式各跑一遍，所以总页次 = 样本页数 × 3。',
    '处理模式': '同一批输入跑三遍：桌面默认（笔画补全开）、桌面关闭增强、CLI 默认。用于发现模式间的差异退化。',
    '负例': '无水印的对照图。合格线：选区为空且输出与输入逐字节一致，任何改动都算失败。',
    '退化': '和基线比，同一样本的指标明显变差（如召回下降、误伤增加）。',
    '待复核': '输出或选区发生了变化但自动判定拿不准，需要人看图确认，不自动算通过也不算失败。',
    '门槛': '预登记的放行条件（config/regression_control.json）。比较前定好，不允许看到结果后放宽。',
    '反混合': '把半透明水印按叠印公式 I = s·B + t 反解出原图 B 的修复方式，优先于重画。',
    'LaMa': '大掩码修复模型，用于没有解析解的区域做内容重画；只应处理已确认选区的连通域。',
    '笔画补全': '对已确认的重复水印，按方向/行距把被正文遮挡的笔画补回来；只补有证据的残笔。',
    '观察基线': '当前状态的封存记录，包含既有缺陷；是新工作的对照基准，不代表质量合格。',
    '独立验证集': '从未参与调参的锁柜评测集，只在里程碑一次性查看；据其调参即降级。',
}

LOW_RECALL = 0.3
GOOD_RECALL = 0.6
BAD_FP_FRAC = 0.005     # 多选占比 0.5% 列入摘要异常清单（不影响分档）
GOOD_FP_FRAC = 0.002    # 多选占比 0.2%（fp 专页入口阈值）
CMP_RECALL_TOL = 0.03   # 与上次基线对比：召回率差异 ≤3pp 视为持平
CMP_MAE_TOL = 5.0       # 与上次基线对比：水印区误差差异 ≤5 视为持平
BAD_FP_PX = 20000       # 绝对兜底：小图上 0.5% 太小时按 2 万像素判
GOOD_BG_MAE = 1.0

STYLE = """
*{box-sizing:border-box}body{font:15px/1.75 system-ui,"PingFang SC","Microsoft YaHei",sans-serif;background:#f2f5f7;color:#243b46;margin:0;padding:26px}
main{max-width:1500px;margin:auto}
.banner{border-radius:12px;padding:18px 24px;margin:16px 0;font-size:17px}
.banner.good{background:#e7f4ec;border:1.5px solid #2c7a4f}.banner.bad{background:#fbeae7;border:1.5px solid #b03a2c}
.board{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.chip{background:white;border:1px solid #d8e1e8;border-radius:10px;padding:10px 18px;min-width:130px}
.chip .v{font-size:24px;font-weight:700;line-height:1.25}.chip .k{font-size:12px;color:#61727f}
.bad{color:#b03a2c}.good{color:#1c7a4d}.mid{color:#b0761a}.muted{color:#7a8a94;font-size:12px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0;align-items:center}
.tabs button{padding:10px 18px;border:1px solid #d8e1e8;border-radius:8px;background:#e9eef2;cursor:pointer;font:inherit}
.tabs button.on{background:white;font-weight:700;border-bottom:2.5px solid #2c7a4f}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;background:white;border:1px solid #d8e1e8;border-radius:8px;padding:12px 14px;margin:0 0 6px}
.toolbar input[type=search]{flex:1;min-width:220px;padding:9px 12px;border:1px solid #c7d3dc;border-radius:7px;font:inherit}
.toolbar select{padding:9px 10px;border:1px solid #c7d3dc;border-radius:7px;font:inherit}
.toolbar label{font-size:13px;display:flex;gap:5px;align-items:center}
.grades{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:10px 0 14px}
.grade{background:white;border:1px solid #d8e1e8;border-radius:12px;padding:16px 20px;cursor:pointer}
.grade.on{outline:2.5px solid #243b46}
.grade h3{margin:0;font-size:15px}.grade .n{font-size:30px;font-weight:700}
.case{background:white;border:1px solid #d8e1e8;border-radius:12px;padding:16px 20px;margin:14px 0}
.case h3{margin:0 0 4px;font-size:16px}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:10px 0}
figure{margin:0}img{width:100%;height:300px;object-fit:contain;background:#f6f8f9;border-radius:6px}
figcaption{text-align:center;font-size:12px;color:#7a8a94;padding-top:4px}
.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:4px 22px;font-size:13.5px;margin:8px 0}
.mgrid b{font-weight:650}
details{margin:8px 0}summary{cursor:pointer;color:#176a4d}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f6f8f9;border-radius:8px;padding:12px}
table{border-collapse:collapse;font-size:13px;background:white;width:100%}td,th{border:1px solid #d5dfe8;padding:6px 10px;text-align:left}
td.r{text-align:right}
.term{border-bottom:1.5px dotted #b0761a;cursor:help;position:relative}
.term i{display:inline-block;width:15px;height:15px;line-height:15px;text-align:center;font-size:10px;font-style:normal;background:#b0761a;color:white;border-radius:50%;margin-left:3px;vertical-align:2px}
#term-pop{position:fixed;display:none;max-width:360px;background:#243b46;color:#f2f5f7;font-size:13px;line-height:1.7;padding:14px 18px;border-radius:10px;box-shadow:0 8px 30px rgba(0,0,0,.35);z-index:99}
#term-pop b{color:#ffd58a}
.note{background:#fff8e6;border:1px solid #e8d9a0;border-radius:8px;padding:10px 16px;font-size:13px;margin:14px 0}
"""

TERMS_JS = """
const GLOSSARY = __GLOSSARY__;
const pop = document.getElementById('term-pop');
document.addEventListener('click', e => {
  const t = e.target.closest('.term');
  if (t) {
    e.stopPropagation();
    const name = t.dataset.term || t.textContent.replace('?','').trim();
    const def = GLOSSARY[name] || '（暂无解释）';
    pop.innerHTML = '<b>' + name + '</b><br>' + def + '<br><span style="opacity:.6;font-size:11px">点击任意处收起</span>';
    pop.style.display = 'block';
    const r = t.getBoundingClientRect();
    pop.style.left = Math.min(window.innerWidth - 380, Math.max(8, r.left)) + 'px';
    pop.style.top = Math.min(window.innerHeight - 160, r.bottom + 8) + 'px';
    return;
  }
  if (pop.style.display === 'block') pop.style.display = 'none';
});
""".replace('__GLOSSARY__', json.dumps(GLOSSARY, ensure_ascii=False))

GRADE_LABEL = {'good': '效果较好', 'warn': '一般/待复核', 'bad': '问题较多', 'err': '问题较多'}
GRADE_CLS = {'good': 'good', 'warn': 'mid', 'bad': 'bad', 'err': 'bad'}
GRADE_RANK = {'err': 3, 'bad': 2, 'warn': 1, 'good': 0}


def esc(x):
    return html.escape('—' if x is None else str(x))


def mark_terms(text: str) -> str:
    for term in sorted(GLOSSARY, key=len, reverse=True):
        text = text.replace(term, f'<span class="term" data-term="{term}">{term}<i>?</i></span>')
    return text


def fmt_pct(v):
    return '—' if v is None else f'{v * 100:.1f}%'


def fmt_px(v):
    return '—' if v is None else f'{v:,}'


def fmt_fp(px, frac):
    """多选背景统一双标：绝对像素 + 占全图百分比。"""
    if px is None:
        return '—'
    base = f'{px:,}px'
    return base + (f'（{frac * 100:.2f}%）' if frac is not None else '')


def fmt_mae(v):
    return '—' if v is None else f'{v:.2f}'


def load_run(out: Path) -> dict:
    return json.loads((out / '运行记录.json').read_text())


def run_moment(run: dict) -> str:
    stamp = run.get('completed_at') or run.get('created_at', '')
    for fmt in ('%Y%m%d_%H%M%S_%f', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(stamp[:26], fmt).strftime('%Y-%m-%d %H:%M')
        except ValueError:
            continue
    return str(stamp)[:16]


def _total_pixels(r: dict) -> int | None:
    """该页处理结果的全图像素数：优先由 selected_fraction 反推，兜底读结果图尺寸。"""
    sel = r.get('selected_pixels')
    frac = r.get('selected_fraction')
    if sel and frac:
        return int(round(sel / frac))
    p = (r.get('files') or {}).get('处理结果.png')
    if p and Path(p).is_file():
        try:
            with Image.open(p) as im:
                return im.width * im.height
        except Exception:
            return None
    return None


def _enrich(r: dict) -> dict:
    total = _total_pixels(r)
    fp = r.get('false_positive_pixels')
    r['_total_px'] = total
    r['_fp_frac'] = (fp / total) if (fp is not None and total) else None
    return r


def _grade(r: dict) -> str:
    """由记录指标分档（纯数据驱动）。多选背景不作为分档指标——
    它仍显示在每张卡片上，但不参与问题档/较好档判定，避免误伤效果好的样本。"""
    if r['status'] == 'error':
        return 'err'
    if r['negative']:
        return 'good' if not r.get('selected_pixels') and not r.get('changed_pixels') else 'bad'
    if r.get('recall') is None:
        return 'warn'
    if r['recall'] < LOW_RECALL:
        return 'bad'
    if r['recall'] >= GOOD_RECALL and (r.get('background_mae') or 99) <= GOOD_BG_MAE:
        return 'good'
    return 'warn'


def load_reference(out: Path) -> tuple[str, dict] | tuple[None, None]:
    """载入“上次基线”参考（供退化/优化对比）。两级来源：
    1. 本封存目录里的 参考_*/**/逐图效果初筛_*.json（手动导入的历史包，如上周旧机基线）；
    2. 回退到基线指针 .regression-control/baseline.json 指向的本机封存基线的运行记录
       （以后每轮控制运行的受控对比对象）。
    返回 (参考标签, {key: 该页指标})；没有参考返回 (None, None)。"""
    for ref_json in sorted(out.glob('参考_*/**/逐图效果初筛_*.json')):
        got = _load_ref_table(ref_json, out)
        if got:
            return got
    return _load_ref_from_pointer(out)


def _load_ref_table(ref_json: Path, out: Path) -> tuple[str, dict] | None:
    try:
        rows = json.loads(ref_json.read_text())
        m = re.search(r'(20\d{6})', ref_json.parent.name) or re.search(r'(20\d{6})', str(ref_json))
        label = f'{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]} 基线' if m else ref_json.parent.name
        import os
        table = {}
        for r in rows:
            if not r.get('key'):
                continue
            op = r.get('output_preview')
            if op:
                p = (ref_json.parent / unquote(str(op))).resolve()
                if p.is_file():
                    r['_old_output'] = os.path.relpath(p, out)
            table[r['key']] = r
        return label, table
    except Exception:
        return None


def _load_ref_from_pointer(out: Path) -> tuple[str, dict] | tuple[None, None]:
    """从基线指针回退加载：.regression-control/baseline.json → 封存基线目录的 运行记录.json。"""
    try:
        root = out.parents[2]  # out = <root>/out/项目防退化控制/<dir>
        ptr = json.loads((root / '.regression-control' / 'baseline.json').read_text())
        base = (root / ptr['path']).resolve()
        if base == out.resolve():
            return None, None  # 基线轮自身不和自己比
        rec_path = base / '运行记录.json'
        if not rec_path.is_file():
            return None, None
        run = json.loads(rec_path.read_text())
        m = re.search(r'(20\d{6})_(\d{2})(\d{2})', base.name)
        label = f'{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]} {m.group(2)}:{m.group(3)} 封存基线' if m else base.name
        import os
        table = {}
        for r in run.get('records', []):
            key = r.get('key')
            if not key:
                continue
            f = r.get('files') or {}
            if f.get('处理结果.png'):
                p = Path(f['处理结果.png']).resolve()
                if p.is_file():
                    r['_old_output'] = os.path.relpath(p, out)
            table[key] = r
        return label, table
    except Exception:
        return None, None


def _compare_rec(rec: dict, ref: dict | None) -> tuple[str | None, float | None, float | None]:
    """单页对比：返回 (better/worse/same/None, 召回率差, 误差改善)。容差见 CMP_*。"""
    if ref is None:
        return None, None, None
    if rec['negative'] or ref.get('negative'):
        new_bad, old_bad = bool(rec.get('changed_pixels')), bool(ref.get('changed_pixels'))
        if new_bad and not old_bad:
            return 'worse', None, None
        if old_bad and not new_bad:
            return 'better', None, None
        return 'same', 0.0, 0.0
    a, b = rec.get('recall'), ref.get('recall')
    if a is None or b is None:
        return None, None, None
    d = a - b
    ma, mb = rec.get('watermark_mae'), ref.get('watermark_mae')
    dmae = (mb - ma) if (ma is not None and mb is not None) else None  # 正数=误差变小=变好
    if abs(d) <= CMP_RECALL_TOL:
        if dmae is not None and dmae > CMP_MAE_TOL:
            return 'better', d, dmae
        if dmae is not None and dmae < -CMP_MAE_TOL:
            return 'worse', d, dmae
        return 'same', d, dmae
    return ('better' if d > 0 else 'worse'), d, dmae


def analyse(run: dict) -> dict:
    recs = [_enrich(r) for r in run['records']]
    for r in recs:
        r['_grade'] = _grade(r)
    paired = [r for r in recs if not r['negative']]
    neg = [r for r in recs if r['negative']]
    errors = [r for r in recs if r['status'] == 'error']
    low = [r for r in paired if r.get('recall') is not None and r['recall'] < LOW_RECALL]
    highfp = [r for r in paired if (r['_fp_frac'] or 0) > BAD_FP_FRAC or (r.get('false_positive_pixels') or 0) > BAD_FP_PX]
    neg_bad = [r for r in neg if r['_grade'] != 'good']
    neg_safe = [r for r in neg if r['_grade'] == 'good']
    elapsed = [r['elapsed_ms'] for r in recs if r.get('elapsed_ms')]
    # 按类型的聚合（结论区用，统一取“桌面默认”模式）
    by_style = {}
    for r in paired:
        if r['mode'] != 'desktop':
            continue
        by_style.setdefault(r.get('style') or '未标注', []).append(r)
    styles = []
    for name, rows in by_style.items():
        rc = [x['recall'] for x in rows if x.get('recall') is not None]
        fp = [x['_fp_frac'] for x in rows if x['_fp_frac'] is not None]
        prec = [x['precision'] for x in rows if x.get('precision') is not None]
        wmae = [x['watermark_mae'] for x in rows if x.get('watermark_mae') is not None]
        bmae = [x['background_mae'] for x in rows if x.get('background_mae') is not None]
        methods = {}
        for x in rows:
            methods[x.get('method') or 'unknown'] = methods.get(x.get('method') or 'unknown', 0) + 1
        n_bad = sum(1 for x in rows if x['_grade'] in ('bad', 'err'))
        n_good = sum(1 for x in rows if x['_grade'] == 'good')
        styles.append(dict(name=name, n=len(rows), recall=(sum(rc) / len(rc)) if rc else None,
                           fp_frac=(sum(fp) / len(fp)) if fp else None,
                           precision=(sum(prec) / len(prec)) if prec else None,
                           watermark_mae=(sum(wmae) / len(wmae)) if wmae else None,
                           background_mae=(sum(bmae) / len(bmae)) if bmae else None,
                           method=max(methods, key=methods.get), method_n=methods[max(methods, key=methods.get)],
                           errors=sum(1 for x in rows if x['status'] == 'error'),
                           bad_rate=n_bad / len(rows), good=n_good, bad=n_bad))
    styles.sort(key=lambda s: (-s['bad_rate'], s['recall'] if s['recall'] is not None else 0))
    return dict(recs=recs, paired=paired, neg=neg, errors=errors, low_recall=low,
                high_fp=highfp, neg_safe=neg_safe, neg_bad=neg_bad,
                mean_elapsed=(sum(elapsed) / len(elapsed) / 1000) if elapsed else None,
                inputs=len({r['id'] for r in recs}), styles=styles, style_label={})


def _style_cause(s: dict) -> str:
    """由聚合数字生成一句初步原因分析（规则启发式，标注为初步）。"""
    if s['errors']:
        return f"初步分析：{s['errors']} 页次直接运行错误（工程缺陷，优先修复），其余页受影响连带评分缺失。"
    causes = []
    if s['fp_frac'] is not None and (s['fp_frac'] > BAD_FP_FRAC or s['precision'] is not None and s['precision'] < 0.5):
        causes.append(f"选区误伤背景/正文：多选占全图 {s['fp_frac'] * 100:.2f}%、精确率仅 {fmt_pct(s['precision'])}"
                      f"——疑似『{s['method']}』通道在证据不足时仍给出大掩码")
    if s['recall'] is not None and s['recall'] < LOW_RECALL and (s['fp_frac'] or 0) <= BAD_FP_FRAC:
        causes.append(f"水印基本未被检出：笔画召回率只有 {fmt_pct(s['recall'])}——现有专用通道对该样式不适用，"
                      f"走到了『{s['method']}』")
    if s['watermark_mae'] is not None and s['watermark_mae'] > 20 and not causes:
        causes.append(f"能选到但修不好：水印区参考误差 {s['watermark_mae']:.1f}，修复重建质量差")
    if s['background_mae'] is not None and s['background_mae'] > 1 and not causes:
        causes.append(f"非水印区有改动（误差 {s['background_mae']:.2f}），保护逻辑可能失效")
    if not causes:
        causes.append(f"指标处于中间地带（召回 {fmt_pct(s['recall'])}、多选 {fmt_pct(s['fp_frac'])}），需逐页目视定位")
    return '初步分析：' + '；'.join(causes) + '。'


def conclusion_html(run: dict, a: dict) -> str:
    """结论区：测试规模 → 每种类型的测试结果（桌面默认口径）→ 问题最大样式与初步原因。
    所有句子只由运行记录的数字生成。"""
    def zh(name: str) -> str:
        return a['style_label'].get(name, name)
    # 与上次基线对比（有参考才显示）放在结论最前
    cmp_block = ''
    if a.get('ref_label'):
        c = a['cmp_counts']
        n = c['better'] + c['worse'] + c['same']
        if c['better'] > c['worse'] * 1.2 and c['better'] > 0:
            direction = '整体偏优化'
        elif c['worse'] > c['better'] * 1.2 and c['worse'] > 0:
            direction = '整体偏退化'
        else:
            direction = '整体大体持平'
        cmp_block = (f'<div class="banner mid" style="background:#e8eef5;border:1.5px solid #3a6b8a"><b>与上次基线对比</b>　'
                     f'对比对象：{esc(a["ref_label"])}（旧机旧代码运行，仅供参考口径）。'
                     f'配对 {n} 页次：优化 {c["better"]}、退化 {c["worse"]}、持平 {c["same"]}'
                     f'{f"、未配对 {c["unpaired"]}" if c["unpaired"] else ""}。{direction}。'
                     f'<br><span style="font-size:13px;opacity:.85">⚠ 非受控对照：差异混有代码与运行环境变化，不能归因于某次改动；'
                     f'从下一轮起与本机封存基线对比才是验收口径。退化/优化明细见下方“较上次”两个 tab。</span></div>')
    pages_per_mode = max(sum(1 for r in a['recs'] if r['mode'] == m) for m in ('desktop', 'desktop_protected', 'cli'))
    complete = run.get('complete') and run.get('integrity_verified')
    multi = pages_per_mode - a['inputs']
    lines = [f'本次测试 {a["inputs"]} 个输入文件、共 {pages_per_mode} 页：其中 {multi} 页来自多页 PDF（一个 PDF 文件含多页，每页单独处理、单独评分），'
             f'其余是一张图一页。三种处理模式各把全部 {pages_per_mode} 页跑一遍（合计 {pages_per_mode * 3} 页次）；'
             f'以下类型结果按“桌面默认”模式口径。']
    if not complete:
        lines.append(f'⚠ 本轮未完整：完成 {len(a["recs"]):,} / {len(run["expected_keys"]):,} 页次，结果不能用于验收。')
    if a['errors']:
        ids = sorted({r['id'] for r in a['errors']})
        lines.append(f'运行错误 {len(a["errors"])} 页次，涉及 {len(ids)} 个输入（{"、".join(i.rsplit("/", 1)[-1] for i in ids[:3])}{" 等" if len(ids) > 3 else ""}）。')
    lines.append(f'无水印负例 {len(a["neg_safe"])} / {len(a["neg"])} 安全；平均耗时 {a["mean_elapsed"]:.1f} 秒/页。'
                 if a['mean_elapsed'] else f'无水印负例 {len(a["neg_safe"])} / {len(a["neg"])} 安全。')
    head = '观察基线已建立：记录当前状态（含既有缺陷），作为以后新旧对照的基准。' if run.get('baseline_kind') else '本轮控制运行结果如下。'
    banner = (f'<div class="banner {"good" if complete and not a["errors"] else "bad"}"><b>结论</b>　{esc(head)}<br>'
              + '<br>'.join('· ' + mark_terms(esc(x)) for x in lines) + '</div>')

    # 每种类型测试结果（桌面默认）
    rows = []
    for s in a['styles']:
        rc_cls = 'bad' if (s['recall'] or 0) < LOW_RECALL else ('mid' if (s['recall'] or 0) < GOOD_RECALL else 'good')
        rows.append(f'<tr><td>{esc(zh(s["name"]))}</td><td class="r">{s["n"]}</td>'
                    f'<td class="r {rc_cls}">{fmt_pct(s["recall"])}</td>'
                    f'<td class="r">{fmt_pct(s["fp_frac"])}</td>'
                    f'<td class="r">{fmt_pct(s["precision"])}</td>'
                    f'<td class="r {"bad" if s["bad_rate"] > .5 else ("mid" if s["bad_rate"] > 0 else "good")}">{s["bad_rate"] * 100:.0f}%（{s["bad"]}/{s["n"]}）</td></tr>')
    style_table = ('<table><tr><th>' + mark_terms('水印类型（样式）') + '</th><th>' + mark_terms('页数') + '</th><th>'
                   + mark_terms('均笔画召回率') + '</th><th>' + mark_terms('均多选占比') + '</th><th>'
                   + mark_terms('均精确率') + '</th><th>' + mark_terms('问题较多占比') + '</th></tr>' + ''.join(rows) + '</table>')

    # 问题最大的样式 Top5 + 初步原因
    top = [s for s in a['styles'] if s['bad'] > 0 or s['errors'] > 0][:5]
    if top:
        items = ''.join(f'<li><b>{esc(zh(s["name"]))}</b>（{s["bad"] + s["errors"]}/{s["n"]} 页进问题档）：{mark_terms(esc(_style_cause(s)))}</li>' for s in top)
        worst = (f'<h3 style="margin:14px 0 6px">问题最大的水印样式与初步分析（按问题档占比排序）</h3><ol>{items}</ol>')
    else:
        worst = '<p class="good">没有样式进入问题档。</p>'
    return cmp_block + banner + worst + f'<h3 style="margin:16px 0 6px">每种类型测试结果（桌面默认模式，按问题占比排序）</h3>{style_table}'


def chips_html(run: dict, a: dict) -> str:
    modes = {}
    for r in a['recs']:
        modes[r['mode']] = modes.get(r['mode'], 0) + 1
    mode_label = {'desktop': '桌面默认', 'desktop_protected': '桌面关闭补全', 'cli': 'CLI 默认'}
    items = [
        (f'{len(a["recs"]):,}/{len(run["expected_keys"]):,}', '完成页次', 'good' if not a['errors'] else ''),
        (str(len(a['errors'])), '运行错误', 'bad' if a['errors'] else 'good'),
        (f'{len(a["neg_safe"])}/{len(a["neg"])}', '负例安全', 'good' if not a['neg_bad'] else 'bad'),
        (str(len(a['low_recall'])), f'召回<{int(LOW_RECALL * 100)}% 页', 'bad' if a['low_recall'] else 'good'),
        (str(len(a['high_fp'])), '多选超线页', 'bad' if a['high_fp'] else 'good'),
        (f'{a["mean_elapsed"]:.1f}s' if a['mean_elapsed'] else '—', '平均耗时/页', ''),
    ]
    for m in ('desktop', 'desktop_protected', 'cli'):
        items.append((str(modes.get(m, 0)), mode_label[m], ''))
    return '<div class="board">' + ''.join(
        f'<div class="chip"><div class="v {c}">{esc(v)}</div><div class="k">{mark_terms(esc(k))}</div></div>' for v, k, c in items) + '</div>'


def _thumb(path: Path, max_px: int = 800, quality: int = 78) -> str:
    im = Image.open(path)
    if max(im.size) > max_px:
        r = max_px / max(im.size)
        im = im.resize((int(im.width * r), int(im.height * r)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.convert('RGB').save(buf, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def _case_title(r: dict) -> str:
    return f'{r["group"]} · {r.get("style") or r["id"].rsplit("/", 1)[-1]} · 第{r["page"]}页'


def _case_metric_line(r: dict) -> str:
    if r['negative']:
        return f'负例：{"安全（零改动）" if r["_grade"] == "good" else "⚠ 有改动"}'
    if r['status'] == 'error':
        return '运行出错，无指标。'
    return (f'笔画召回 {fmt_pct(r.get("recall"))} · 精确率 {fmt_pct(r.get("precision"))} · '
            f'漏选/多选 {fmt_px(r.get("missed_pixels"))} / {fmt_fp(r.get("false_positive_pixels"), r["_fp_frac"])} · '
            f'耗时 {r.get("elapsed_ms", 0) / 1000:.1f}s · {esc(str(r.get("method")))}')


def build_summary(out: Path, run: dict, a: dict, moment: str) -> Path:
    e = esc
    outliers = list(a['low_recall']) + [r for r in a['high_fp'] if r not in a['low_recall']]
    ranked, seen = [], set()
    for r in sorted(outliers, key=lambda x: ((x.get('recall') if x.get('recall') is not None else 0),
                                             -(x.get('false_positive_pixels') or 0))):
        if r['key'] in seen:
            continue
        seen.add(r['key'])
        ranked.append(r)
    show = a['errors'][:10] + ranked[:30]

    def figs_embed(r):
        imgs = []
        for fname, cap in [('原图预览.jpg', '原图'), ('选区预览.jpg', '选区'), ('处理结果.png', '结果'),
                           ('漏选多选核对.jpg', '蓝漏选/橙多选')]:
            p = (r.get('files') or {}).get(fname)
            if p and Path(p).is_file():
                imgs.append(f'<figure><img loading="lazy" src="{_thumb(Path(p))}"><figcaption>{cap}</figcaption></figure>')
        return f'<div class="figs">{"".join(imgs)}</div>' if imgs else '<p class="muted">无图像产物。</p>'

    blocks = []
    for r in show:
        why = '运行错误' if r['status'] == 'error' else (
            f'召回 {fmt_pct(r.get("recall"))}' + (f'、多选 {fmt_fp(r.get("false_positive_pixels"), r["_fp_frac"])}'
                                                 if (r['_fp_frac'] or 0) > BAD_FP_FRAC or (r.get('false_positive_pixels') or 0) > BAD_FP_PX else ''))
        blocks.append(f'<div class="case"><h3>{e(_case_title(r))}　<span class="muted">{e(r["mode"])}</span></h3>'
                      f'<p class="bad">▲ {e(why)}</p><p class="muted">{e(_case_metric_line(r))}</p>{figs_embed(r)}</div>')
    list_html = ''.join(blocks) if blocks else '<p class="good">没有运行错误，也没有触发异常参考线的样本。</p>'

    grid = {}
    for r in a['paired']:
        if r.get('recall') is not None:
            grid.setdefault((r['mode'], r['group']), []).append(r)
    mode_label = {'desktop': '桌面默认', 'desktop_protected': '桌面关闭补全', 'cli': 'CLI 默认'}
    groups = sorted({g for _, g in grid})
    rows = []
    for g in groups:
        tds = [f'<td>{e(g)}</td>']
        for m in ('desktop', 'desktop_protected', 'cli'):
            cell = grid.get((m, g), [])
            if not cell:
                tds.append('<td class="muted">—</td>')
                continue
            rc = sum(x['recall'] for x in cell) / len(cell)
            fpf = [x['_fp_frac'] for x in cell if x['_fp_frac'] is not None]
            fp_mean = sum(fpf) / len(fpf) if fpf else None
            cls = 'bad' if rc < LOW_RECALL else ('mid' if rc < GOOD_RECALL else 'good')
            tds.append(f'<td class="r"><span class="{cls}">{fmt_pct(rc)}</span><br>'
                       f'<span class="muted">多选均 {fmt_pct(fp_mean)} · {len(cell)}页</span></td>')
        rows.append('<tr>' + ''.join(tds) + '</tr>')
    table_html = ('<table><tr><th>批次</th><th>桌面默认（均召回）</th><th>桌面关闭补全</th><th>CLI 默认</th></tr>'
                  + ''.join(rows) + '</table>')

    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>项目防退化控制摘要报告 {e(moment)}</title><style>{STYLE}</style></head><body><main>
<h1>项目防退化控制摘要报告 <span class="muted">{e(moment)}</span></h1>
{conclusion_html(run, a)}
{chips_html(run, a)}
<p class="note">本报告为摘要单文件（图片 base64 内嵌，可直接发送）；逐页明细见同目录《项目防退化控制明细报告 {e(moment)}》。分档指标：运行错误、负例改动、笔画召回率、非水印区误差；多选背景只逐张展示、不参与分档。像素误差不是水印残留率；所有样本不作独立泛化证明。</p>
<h2>处理清单（运行错误 + 触发异常参考线的样本）</h2>
{list_html}
<h2>分层总表（带水印样本，按模式 × 批次平均）</h2>
{table_html}
<p class="muted">参考线口径：平均笔画召回率 &lt;30% 标红、30–60% 标黄；多选背景超线样本列入上方清单但不影响分档。统计含全部完成页，错误页另列。</p>
</main><div id="term-pop"></div><script>{TERMS_JS}</script></body></html>'''
    dest = out / f'项目防退化控制摘要报告_{moment.replace("-", "").replace(" ", "_").replace(":", "")}.html'
    dest.write_text(page)
    return dest


def build_detail(out: Path, run: dict, a: dict, moment: str) -> Path:
    """明细报告：完全复用《效果分档总览》版式（tools/assets 下的 CSS/JS），
    页首加结论区（测试规模 / 每种类型结果-桌面默认 / 问题最大样式与初步分析）。
    图片为相对路径（AGENTS.md §8 对全量逐页明细的 base64 豁免）。"""
    e = esc
    assets = Path(__file__).parent / 'assets'
    css = (assets / 'effect_overview_style.css').read_text()
    app_js = (assets / 'effect_overview_app.js').read_text()

    def rel(path: Path) -> str:
        return quote(str(Path(path).relative_to(out)), safe='/')

    manifest = json.loads((out / '原始清单.json').read_text())
    input_by_id = {c['id']: c['input'] for c in manifest['cases']}
    style_label = {}
    for c in manifest['cases']:
        style_label.setdefault(c.get('style'), c.get('label') or c.get('style'))

    def rel_input(r: dict) -> str:
        src = input_by_id.get(r['id'])
        if not src or not Path(src).is_file():
            return ''
        import os
        return quote(os.path.relpath(src, out), safe='/')

    rows = []
    for r in a['recs']:
        g = r['_grade']
        bucket = 'good' if g == 'good' else ('review' if g == 'warn' else 'poor')
        f = r.get('files') or {}
        reasons = []
        if r['status'] == 'error':
            reasons.append('运行错误：' + str(r.get('error', '')).splitlines()[0][:120])
        elif r['negative']:
            reasons.append('无水印且没有选区、没有改动，安全保留' if g == 'good' else '⚠ 无水印对照发生了改动/误选')
        else:
            if r.get('recall') is not None:
                reasons.append(f'笔画召回率 {fmt_pct(r["recall"])}，明显笔画召回 {fmt_pct(r.get("strong_recall"))}')
            else:
                reasons.append('没有有效评分（未检出或未通过评分条件）')
            fp = r.get('false_positive_pixels')
            if fp is not None:
                reasons.append(f'多选背景 {fp:,}px' + (f'（占全图 {r["_fp_frac"] * 100:.2f}%）' if r['_fp_frac'] is not None else ''))
            if r.get('hole_selected_pixels'):
                reasons.append(f'字洞误选 {r["hole_selected_pixels"]:,}px')
            if r.get('outside_selection_changed_pixels'):
                reasons.append(f'选区外有 {r["outside_selection_changed_pixels"]:,}px 被改动')
            if r.get('_cmp') in ('better', 'worse') and r.get('_d_recall') is not None:
                arrow = '↑' if r['_cmp'] == 'better' else '↓'
                old_r = fmt_pct(r.get('_old_recall'))
                reasons.append(f'较上次基线：召回率 {old_r} → 本次 {fmt_pct(r.get("recall"))}（{r["_d_recall"] * 100:+.1f}pp）{arrow}')
            elif r.get('_cmp') == 'worse':
                reasons.append('较上次基线：发生退化（无召回数据，见误差指标）')
        rows.append(dict(key=r['key'], id=r['id'], page=r['page'], mode=r['mode'], status=r['status'],
                         input_sha256=r.get('input_sha256'), style=r.get('style') or '未标注',
                         label=style_label.get(r.get('style')), group=r['group'], negative=r['negative'],
                         bucket=bucket, reasons=reasons,
                         recall=r.get('recall'), precision=r.get('precision'), strong_recall=r.get('strong_recall'),
                         missed_pixels=r.get('missed_pixels'), false_positive_pixels=r.get('false_positive_pixels'),
                         fp_frac=r.get('_fp_frac'),
                         cmp=r.get('_cmp'), d_recall=r.get('_d_recall'), d_mae=r.get('_d_mae'),
                         old_output=r.get('_old_output'),
                         hole_selected_pixels=r.get('hole_selected_pixels'),
                         selected_pixels=r.get('selected_pixels'), selected_fraction=r.get('selected_fraction'),
                         changed_pixels=r.get('changed_pixels'),
                         outside_selection_changed_pixels=r.get('outside_selection_changed_pixels'),
                         source_watermark_mae=None, watermark_mae=r.get('watermark_mae'),
                         watermark_error_reduction=None,
                         background_damage_mae=r.get('background_mae'), background_damage_p99=None,
                         background_mae=r.get('background_mae'),
                         exact_reference_match=None, manual_quality_reviewed=False,
                         recognition_quality=None, residual_quality=None, whitespace_quality=None,
                         elapsed_ms=r.get('elapsed_ms'), error=r.get('error'),
                         method=r.get('method'), repair_mode=r.get('repair_mode'),
                         source_preview=rel(Path(f['原图预览.jpg'])) if f.get('原图预览.jpg') else None,
                         output_preview=rel(Path(f['处理结果.png'])) if f.get('处理结果.png') else None,
                         output=rel(Path(f['处理结果.png'])) if f.get('处理结果.png') else None,
                         selection_preview=rel(Path(f['选区预览.jpg'])) if f.get('选区预览.jpg') else None,
                         diagnostic=rel(Path(f['漏选多选核对.jpg'])) if f.get('漏选多选核对.jpg') else None,
                         original=rel_input(r)))
    counts = {}
    negative_counts = {}
    for r in rows:
        counts.setdefault(r['mode'], {}).setdefault(r['bucket'], 0)
        counts[r['mode']][r['bucket']] += 1
        if r['negative']:
            negative_counts[r['mode']] = negative_counts.get(r['mode'], 0) + 1
    for m in counts:
        counts[m].setdefault('good', 0)
        counts[m].setdefault('review', 0)
        counts[m].setdefault('poor', 0)
        negative_counts.setdefault(m, 0)
    import hashlib as _hl
    payload = dict(rows=rows, counts=counts, negative_counts=negative_counts,
                   sources=dict(original_report='项目防退化控制报告.html',
                                original_records='运行记录.json', inventory='原始清单.json', rules='验收规则.json'),
                   manifest=dict(created_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                                 source_completed_at=run.get('completed_at'),
                                 source_record_sha256=_hl.sha256((out / '运行记录.json').read_bytes()).hexdigest(),
                                 record_count=len(rows), sample_scope_revision=None,
                                 baseline=str(out)))
    cmp_tabs = ''
    if a.get('ref_label'):
        cmp_tabs = (f'<button id="tab-worse" class="tab" type="button" role="tab" aria-controls="effect-panel" aria-selected="false" tabindex="-1" data-bucket="worse">'
                    f'<span class="label">较上次退化</span><span class="num" id="count-worse">—</span>'
                    f'<span class="small">对比 {e(a["ref_label"])}，按召回降幅排序</span></button>'
                    f'<button id="tab-better" class="tab" type="button" role="tab" aria-controls="effect-panel" aria-selected="false" tabindex="-1" data-bucket="better">'
                    f'<span class="label">较上次优化</span><span class="num" id="count-better">—</span>'
                    f'<span class="small">对比 {e(a["ref_label"])}，按召回升幅排序</span></button>')
    cmp_note = ''
    if a.get('ref_label'):
        cmp_note = (f'<h3>“较上次退化／优化”怎么判定</h3><p>与封存目录里的 {e(a["ref_label"])} 逐页配对（模式＋输入＋页码），'
                    f'以笔画召回率差（容差 ±{CMP_RECALL_TOL * 100:.0f} 个百分点）为主、水印区参考误差差（容差 ±{CMP_MAE_TOL:.0f}）为辅判定；'
                    f'无水印对照按是否发生改动判定。该参考运行在旧机器＋旧代码上，对比是观察口径，不是验收门槛。'
                    f'从下一轮起，对比对象为本机封存基线，才是受控口径。</p>')
    data_json = json.dumps(payload, ensure_ascii=False).replace('</script', '<\\/script')

    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>项目防退化控制明细报告 {e(moment)}</title>
<style>{css}
.conclusion {{margin:0 0 18px}}
.conclusion table {{width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;background:#fff}}
.conclusion th,.conclusion td {{padding:6px 10px;border:1px solid #d5dfe8;text-align:left}}
.conclusion td.r {{text-align:right}}
.conclusion .good {{color:#1c7a46;font-weight:600}}
.conclusion .mid {{color:#8a6d1a;font-weight:600}}
.conclusion .bad {{color:#b03325;font-weight:600}}
.tabs {{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}}
.conclusion table {{font-size:13px}}
.banner {{border-radius:12px;padding:16px 22px;margin:14px 0;font-size:15.5px}}
.banner.good {{background:#e7f4ec;border:1.5px solid #2c7a4f}}.banner.bad {{background:#fbeae7;border:1.5px solid #b03a2c}}
.note {{background:#fff8e6;border:1px solid #e8d9a0;border-radius:8px;padding:10px 16px;font-size:13px}}
.term {{border-bottom:1.5px dotted #b0761a;cursor:help}}
.term i {{display:inline-block;width:15px;height:15px;line-height:15px;text-align:center;font-size:10px;font-style:normal;background:#b0761a;color:#fff;border-radius:50%;margin-left:3px;vertical-align:2px}}
#term-pop {{position:fixed;display:none;max-width:360px;background:#243b46;color:#f2f5f7;font-size:13px;line-height:1.7;padding:14px 18px;border-radius:10px;box-shadow:0 8px 30px rgba(0,0,0,.35);z-index:99}}
#term-pop b {{color:#ffd58a}}
</style></head><body><main class="shell">
<a class="back" href="../水印验证报告总目录.html">← 返回报告总目录</a>
<div class="conclusion">
<h1>项目防退化控制明细报告 <span style="font-size:15px;color:#7a8a94">{e(moment)}</span></h1>
<p class="byline">来源：{e(run.get('completed_at') or run.get('created_at'))} 完成的控制运行 · 版式同《效果分档总览》</p>
{conclusion_html(run, a)}
<p class="note">三档是自动指标初筛，不是验收结论；专业术语带 <i style="display:inline-block;width:14px;height:14px;line-height:14px;text-align:center;font-size:10px;font-style:normal;background:#b0761a;color:#fff;border-radius:50%">?</i> 的可点击看解释。可发送的单文件摘要在同目录。</p>
</div>
<p class="notice"><strong>三档是自动指标初筛，尚未逐张人工验收。</strong>“效果较好”不等于全部内容恢复正确；像素误差减少不等于水印去除率。无水印对照默认隐藏，勾选后可查看；它们仍用于检查误删，不计作去水印成功。</p>
<div class="modebar"><label>处理模式<select id="mode"><option value="desktop">桌面默认（笔画补全开启）</option><option value="desktop_protected">桌面关闭笔画补全</option><option value="cli">CLI 默认（逐文件）</option></select></label><span class="scope" id="scope"></span></div>
<nav class="tabs" role="tablist" aria-label="按去水印效果分类">
<button id="tab-good" class="tab" type="button" role="tab" aria-controls="effect-panel" aria-selected="true" data-bucket="good"><span class="label">效果较好</span><span class="num" id="count-good">—</span><span class="small">指标初筛较好／无水印安全保留</span></button>
<button id="tab-review" class="tab" type="button" role="tab" aria-controls="effect-panel" aria-selected="false" tabindex="-1" data-bucket="review"><span class="label">一般／待复核</span><span class="num" id="count-review">—</span><span class="small">有差异、证据不足或需要目视核对</span></button>
<button id="tab-poor" class="tab" type="button" role="tab" aria-controls="effect-panel" aria-selected="false" tabindex="-1" data-bucket="poor"><span class="label">问题较多</span><span class="num" id="count-poor">—</span><span class="small">未有效处理、误差改善少或损伤风险</span></button>
<button id="tab-fp" class="tab" type="button" role="tab" aria-controls="effect-panel" aria-selected="false" tabindex="-1" data-bucket="fp"><span class="label">多选背景</span><span class="num" id="count-fp">—</span><span class="small">误伤检查：占比&gt;0.2%，按占比从大到小</span></button>
{cmp_tabs}
</nav>
<section id="effect-panel" role="tabpanel" aria-labelledby="tab-good">
<div class="filters"><input id="search" type="search" aria-label="搜索样本" placeholder="搜索样本、样式或归类原因…"><select id="style" aria-label="筛选水印样式"><option value="">全部样式</option></select><select id="group" aria-label="筛选样本来源"><option value="">全部来源</option></select><label><input id="duplicates" type="checkbox">显示重复批次记录</label><label><input id="negatives" type="checkbox">显示无水印对照（检查误删）</label></div>
<p class="view-note" id="view-note" role="status" aria-live="polite"></p><div class="cards" id="cards"></div><div class="empty" id="empty" hidden>当前条件下没有图片，可切换档位、样式或清空搜索。</div>
<div class="pager"><button id="previous" type="button">上一页</button><span id="pagination"></span><button id="next" type="button">下一页</button></div>
</section>
<details class="methods"><summary>查看分档规则、三模式统计和数据来源</summary>
<h3>这是效果初筛，不是验收门槛</h3>
<p>效果较好：笔画召回率≥60%、非水印区新增平均误差≤1.0，且评分完整；无水印且没有选区、没有改动的对照也放在本档并单独标注。</p>
<p>问题较多：运行错误；或无水印对照发生误选／改动；或笔画召回率&lt;30%。其余与无法评估的记录放入一般／待复核。多选背景不作为分档指标，只逐张展示（绝对像素＋占全图百分比），供目视参考。</p>
<p>笔画召回的真值为 alpha&gt;0 的全部水印像素，不隐藏淡边缘。字间空白、实际水印残留和正文识别正确率尚未可靠评估；像素误差不等于残留率。</p>
<h3>重复样本如何处理</h3>
<p id="deduplication-note">默认按“模式＋输入文件哈希＋页码”合并，同图不同批次展示较差档。勾选“显示重复批次记录”可查看全部批次。PDF 与 JPG 属于不同输入。</p>
{cmp_note}
<div id="mode-summary"></div><div class="sources" id="sources"></div>
<p>这些数据来自开发、回归和历史样本，没有新的独立泛化验证；基线轮记录当时表现（含既有缺陷），不代表质量合格。原始封存文件未修改，本页分档规则只用于阅读，不改变防退化验收规则。</p>
</details>
<footer>所有图片均来自实际输入、实际输出或已保存记录的评分后核对图。按图片逐张判断，不用平均分抵消坏图。</footer>
</main>
<dialog id="viewer"><div class="dialogbar"><span id="viewer-title"></span><a id="viewer-link" target="_blank" rel="noopener">打开原文件</a><button id="viewer-close" type="button">关闭</button></div><img id="viewer-image" alt="放大的实际测试图片"></dialog>
<script id="reviewed-data" type="application/json">{data_json}</script>
<div id="term-pop"></div>
<script>{TERMS_JS}
{app_js}</script>
</body></html>'''
    dest = out / f'项目防退化控制明细报告_{moment.replace("-", "").replace(" ", "_").replace(":", "")}.html'
    dest.write_text(page)
    return dest


def build_both(out: Path) -> tuple[Path, Path]:
    out = Path(out).resolve()
    run = load_run(out)
    a = analyse(run)
    # 样式中文名：取原始清单里每个样式的中文标签
    try:
        manifest = json.loads((out / '原始清单.json').read_text())
        for c in manifest['cases']:
            a['style_label'].setdefault(c.get('style'), c.get('label') or c.get('style'))
    except Exception:
        pass
    a['style_label'].setdefault(None, '未标注')
    # 与上次基线对比（参考包在封存目录里才启用）
    ref_label, ref_map = load_reference(out)
    a['ref_label'] = ref_label
    cmp_counts = {'better': 0, 'worse': 0, 'same': 0, 'unpaired': 0}
    for r in a['recs']:
        ref_row = ref_map.get(r['key']) if ref_map else None
        cmp_, d, dmae = _compare_rec(r, ref_row)
        r['_cmp'], r['_d_recall'], r['_d_mae'] = cmp_, d, dmae
        r['_old_output'] = (ref_row or {}).get('_old_output')
        r['_old_recall'] = (ref_row or {}).get('recall')
        cmp_counts[cmp_ if cmp_ in ('better', 'worse', 'same') else 'unpaired'] += 1
    a['cmp_counts'] = cmp_counts
    moment = run_moment(run)
    return build_summary(out, run, a, moment), build_detail(out, run, a, moment)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True, help='防退化控制轮目录（含运行记录.json）')
    args = ap.parse_args()
    summary, detail = build_both(args.run_dir)
    print(summary)
    print(detail)
