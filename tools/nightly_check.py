"""每夜固定回归巡检：跑最小回归集，和上一夜对比，生成一页晨间摘要。

用法（crontab 示例，每天 02:00）：
  0 2 * * * cd /Users/a123/Desktop/offline-dewatermark-ai && HF_HUB_OFFLINE=1 QT_QPA_PLATFORM=offscreen .venv/bin/python tools/nightly_check.py >> out/nightly_cron.log 2>&1

设计口径：
- 巡检集 = data/watermark_regression_min_v1（固定哨兵，43 输入／59 页）。
  它是开发回归哨兵，不是独立泛化验证；独立集按里程碑一次性查看。
- 第一夜没有历史，只记录结果，不比较；从第二夜起与上一夜逐页对比。
- 晨间摘要 HTML 图片全部 base64 内嵌（只内嵌有变化/失败的样本，控制体积），
  单文件可直接发送；全量图在当夜目录里。
- 回归运行返回非零（有失败页）不代表本脚本失败；只有基础设施错误才非零退出。
"""
from __future__ import annotations
import argparse
import base64
import fcntl
import html
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NIGHTLY = ROOT / 'out' / 'nightly'
LOCK = NIGHTLY / '.lock'


def embed(path: Path, max_px: int = 640) -> str:
    im = Image.open(path)
    if max(im.size) > max_px:
        r = max_px / max(im.size)
        im = im.resize((int(im.width * r), int(im.height * r)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.convert('RGB').save(buf, format='JPEG', quality=75)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def load_run(d: Path) -> dict:
    return json.loads((d / 'summary.json').read_text())


def page_map(summary: dict) -> dict:
    out = {}
    for r in summary.get('results', []):
        for p in r.get('pages', []):
            out[(r['case_id'], p['page'])] = dict(case=r, page=p)
    return out


def classify(t_now, t_prev) -> str:
    now = (t_now.get('passed') is True, t_now.get('status'))
    prev = (t_prev.get('passed') is True, t_prev.get('status')) if t_prev else None
    if now[1] == 'error':
        return 'error'
    if prev and prev[1] != 'error' and now[0] and not prev[0]:
        return 'improved'
    if prev and not now[0] and prev[0]:
        return 'regressed'
    if not now[0]:
        return 'still_failing'
    return 'ok'


def render_morning(run_dir: Path, summary: dict, prev_dir: Path | None, prev_map: dict) -> Path:
    e = html.escape
    pages = page_map(summary)
    rows = []
    for key in sorted(pages):
        cur = pages[key]['page']
        prev = prev_map.get(key, {}).get('page') if prev_map else None
        rows.append(dict(key=key, case=pages[key]['case'], page=cur, prev=prev,
                         cls=classify(cur, prev)))
    n_paired = [r for r in rows if not r['case']['negative']]
    n_neg = [r for r in rows if r['case']['negative']]
    paired_pass = sum(1 for r in n_paired if r['page'].get('passed') is True)
    neg_safe = sum(1 for r in n_neg if r['page'].get('passed') is True)
    errors = [r for r in rows if r['cls'] == 'error']
    regressed = [r for r in rows if r['cls'] == 'regressed']
    improved = [r for r in rows if r['cls'] == 'improved']
    still = [r for r in rows if r['cls'] == 'still_failing']
    manual = sum(1 for r in rows if r['page'].get('status') == 'manual_review')

    def figs(r):
        p = r['page']
        imgs = []
        for rel, cap in [(p.get('input_preview'), '输入'), (p.get('output'), '输出'),
                         (p.get('predicted_mask'), '掩码')]:
            if rel:
                imgs.append(f'<figure><img loading="lazy" src="{embed(run_dir / rel)}"><figcaption>{cap}</figcaption></figure>')
        return f'<div class="figs">{"".join(imgs)}</div>' if imgs else ''

    def block(r, note):
        c = r['case']
        prev_note = ''
        if r['prev']:
            prev_note = f'\u003cspan class="muted"\u003e昨晚：{e(str(r["prev"].get("passed")))}\u003c/span\u003e'
        return (f'\u003carticle class="case {r["cls"]}"\u003e\u003cb\u003e{e(c["case_id"])}\u003c/b\u003e 第{r["page"]["page"]}页 '
                f'\u003cspan class="muted\"\u003e{e(c["label"])} · {e(c.get("score_mode", ""))}\u003c/span\u003e {note} {prev_note}'
                f'{figs(r)}\u003c/article\u003e')

    need = ''.join(block(r, '\u003cspan class="bad"\u003e✗ 退化\u003c/span\u003e') for r in regressed) \
        + ''.join(block(r, '\u003cspan class="bad"\u003e✗ 出错\u003c/span\u003e') for r in errors)
    if not prev_dir:
        # 第一夜没有历史："需要处理" = 当前所有未过页，建立今日修复清单。
        need = ''.join(block(r, '\u003cspan class="bad"\u003e✗ 未过\u003c/span\u003e') for r in still) \
            + ''.join(block(r, '\u003cspan class="bad"\u003e✗ 出错\u003c/span\u003e') for r in errors)
    if not need:
        need = '\u003cp class="good"\u003e没有退化和新错误。\u003c/p\u003e'
    imp = ''.join(block(r, '\u003cspan class="good"\u003e✓ 改善\u003c/span\u003e') for r in improved) or '\u003cp class="muted"\u003e无。\u003c/p\u003e'
    all_rows = ''.join(
        f'\u003ctr\u003e\u003ctd\u003e{e(r["case"]["case_id"])}\u003c/td\u003e\u003ctd\u003e{r["page"]["page"]}\u003c/td\u003e\u003ctd\u003e{e(r["case"]["label"])}\u003c/td\u003e'
        f'\u003ctd\u003e{e(r["page"].get("status", ""))}\u003c/td\u003e\u003ctd\u003e{e(str(r["page"].get("passed")))}\u003c/td\u003e'
        f'\u003ctd\u003e{e({"regressed": "退化", "improved": "改善", "error": "出错", "still_failing": "仍未过", "ok": "—"}.get(r["cls"], ""))}\u003c/td\u003e\u003c/tr\u003e'
        for r in rows)
    when = summary.get('created_at', '')[:16].replace('T', ' ')
    prev_note = f'对比上一夜（{e(prev_dir.name)}）' if prev_dir else '第一夜：只记录结果，无对比'
    page = f'''\u003c!doctype html\u003e\u003chtml lang="zh-CN"\u003e\u003chead\u003e\u003cmeta charset="utf-8"\u003e\u003cmeta name="viewport" content="width=device-width,initial-scale=1"\u003e
\u003ctitle\u003e去水印每夜巡检晨间摘要 {e(run_dir.name[:8])}\u003c/title\u003e\u003cstyle\u003e
*{{box-sizing:border-box}}body{{font:15px/1.8 system-ui,"PingFang SC",sans-serif;background:#f1f4f8;color:#25394e;padding:25px;max-width:1200px;margin:auto}}
.board{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}}.chip{{background:white;border:1px solid #dae4ef;border-radius:10px;padding:12px 20px;min-width:140px}}
.chip .v{{font-size:26px;font-weight:700}}.chip .k{{font-size:12px;color:#61727f}}
.case{{background:white;border:1px solid #dae4ef;border-radius:10px;padding:14px 18px;margin:12px 0}}
.bad{{color:#c0392b;font-weight:700}}.good{{color:#1c7a4d;font-weight:700}}.muted{{color:#7a8a94;font-size:12px}}
.figs{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px}}figure{{margin:0}}img{{width:100%;height:220px;object-fit:contain;background:#f3f5f7}}
table{{border-collapse:collapse;font-size:13px;background:white;width:100%}}td,th{{border:1px solid #d5dfe8;padding:6px 10px}}
details{{background:white;border:1px solid #dae4ef;border-radius:10px;padding:12px 18px;margin:14px 0}}
.note{{background:#eef3f8;border:1px solid #cfdae6;border-radius:8px;padding:10px 16px;font-size:13px}}
\u003c/style\u003e\u003c/head\u003e\u003cbody\u003e
\u003ch1\u003e去水印每夜巡检 · 晨间摘要\u003c/h1\u003e
\u003cp class="muted"\u003e运行时间 {e(when)} · {prev_note} · 巡检集：固定最小回归（开发哨兵，不证明通用成功率）\u003c/p\u003e
\u003cdiv class="board"\u003e
\u003cdiv class="chip"\u003e\u003cdiv class="v {'good' if paired_pass == len(n_paired) else 'bad'}"\u003e{paired_pass}/{len(n_paired)}\u003c/div\u003e\u003cdiv class="k"\u003e配对正例通过页\u003c/div\u003e\u003c/div\u003e
\u003cdiv class="chip"\u003e\u003cdiv class="v {'good' if neg_safe == len(n_neg) else 'bad'}"\u003e{neg_safe}/{len(n_neg)}\u003c/div\u003e\u003cdiv class="k"\u003e无水印负例安全\u003c/div\u003e\u003c/div\u003e
\u003cdiv class="chip"\u003e\u003cdiv class="v {'bad' if regressed or errors else 'good'}"\u003e{len(regressed) + len(errors)}\u003c/div\u003e\u003cdiv class="k"\u003e退化/新错误\u003c/div\u003e\u003c/div\u003e
\u003cdiv class="chip"\u003e\u003cdiv class="v"\u003e{len(improved)}\u003c/div\u003e\u003cdiv class="k"\u003e较昨晚改善\u003c/div\u003e\u003c/div\u003e
\u003cdiv class="chip\"\u003e\u003cdiv class="v mid-n\"\u003e{manual}\u003c/div\u003e\u003cdiv class="k\"\u003e待人工复核页\u003c/div\u003e\u003c/div\u003e
\u003cdiv class="chip\"\u003e\u003cdiv class="v"\u003e{len(still)}\u003c/div\u003e\u003cdiv class="k"\u003e一直未过（非新增）\u003c/div\u003e\u003c/div\u003e
\u003c/div\u003e
\u003cp class="note"\u003e口径：配对正例按应用四条件判定通过；负例要求零改动。第一页看完\u0026#12300;退化/新错误\u0026#12301;和\u0026#12300;一直未过\u0026#12301;两节，就知道今天该修什么。\u003c/p\u003e
\u003ch2\u003e今早需要处理（退化 / 新错误）\u003c/h2\u003e{need}
\u003ch2\u003e较昨晚有改善\u003c/h2\u003e{imp}
\u003cdetails\u003e\u003csummary\u003e\u003cb\u003e全部 {len(rows)} 页结果（点击展开）\u003c/b\u003e\u003c/summary\u003e
\u003ctable\u003e\u003ctr\u003e\u003cth\u003e用例\u003c/th\u003e\u003cth\u003e页\u003c/th\u003e\u003cth\u003e标签\u003c/th\u003e\u003cth\u003e状态\u003c/th\u003e\u003cth\u003e通过\u003c/th\u003e\u003cth\u003e对比昨晚\u003c/th\u003e\u003c/tr\u003e{all_rows}\u003c/table\u003e\u003c/details\u003e
\u003c/body\u003e\u003c/html\u003e'''
    dest = run_dir / '晨间摘要.html'
    dest.write_text(page)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--render-only', type=Path, help='只按已有结果重建某一夜的晨间摘要，不重跑推理')
    ap.add_argument('--keep-going-window', type=int, default=0, help='忽略锁的宽限秒数（默认 0=严格）')
    args = ap.parse_args()
    if args.render_only:
        run_dir = args.render_only
        summary = load_run(run_dir)
        history = sorted(d for d in NIGHTLY.iterdir() if d.is_dir() and d.name != run_dir.name and (d / 'summary.json').exists())
        prev_dir = history[-1] if history else None
        prev_map = page_map(load_run(prev_dir)) if prev_dir else {}
        print('MORNING', render_morning(run_dir, summary, prev_dir, prev_map), flush=True)
        return 0
    NIGHTLY.mkdir(parents=True, exist_ok=True)
    with open(LOCK, 'w') as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('另一夜巡检仍在运行，跳过。', flush=True)
            return 0
        stamp = time.strftime('%Y%m%d_%H%M%S')
        run_dir = NIGHTLY / stamp
        cmd = [sys.executable, str(ROOT / 'tools/run_min_regression.py'), '--suite', 'all', '--out', str(run_dir)]
        print('RUN', ' '.join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, 'HF_HUB_OFFLINE': '1', 'QT_QPA_PLATFORM': 'offscreen'})
        print('regression exit code', proc.returncode, flush=True)
        summary = load_run(run_dir)
        history = sorted(d for d in NIGHTLY.iterdir() if d.is_dir() and d.name != stamp and (d / 'summary.json').exists())
        prev_dir = history[-1] if history else None
        prev_map = page_map(load_run(prev_dir)) if prev_dir else {}
        dest = render_morning(run_dir, summary, prev_dir, prev_map)
        print('MORNING', dest, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
