"""渲染训练优化进度页（唯一入口：先改 state JSON，再跑本工具）。

进度页：out/训练优化进度_<日期>.html
状态文件：out/训练优化进度_<日期>.state.json

不要手工改 HTML；手工改的内容会在下次渲染时丢失。
"""
from __future__ import annotations
import argparse, html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STATUS_LABEL = {'todo': '未开始', 'doing': '进行中', 'done': '完成', 'blocked': '受阻'}
STATUS_CLASS = {'todo': 'todo', 'doing': 'doing', 'done': 'done', 'blocked': 'blocked'}


def render(state: dict) -> str:
    e = html.escape
    phase_cards = []
    for ph in state['phases']:
        items = ''.join(
            f'<li class="{STATUS_CLASS[i["status"]]}"><b>[{STATUS_LABEL[i["status"]]}]</b> {e(i["title"])}'
            + (f' — <a href="{e(i["link"])}">{e(i.get("link_text") or "产物")}</a>' if i.get('link') else '')
            + (f'<span class="muted">（{e(i["note"])}）</span>' if i.get('note') else '')
            + '</li>'
            for i in ph['items'])
        phase_cards.append(
            f'<section><h2>阶段 {ph["no"]}｜{e(ph["name"])}</h2>'
            f'<p class="muted">{e(ph.get("goal", ""))}</p><ul class="tasks">{items}</ul>'
            + (f'<p>{e(ph["result"])}</p>' if ph.get('result') else '') + '</section>')
    events = ''.join(f'<tr><td>{e(ev["date"])}</td><td>{e(ev["what"])}</td><td>{e(ev.get("detail", ""))}</td></tr>'
                     for ev in state['events'])
    decisions = ''.join(f'<li><b>{e(d["date"])}</b>　{e(d["text"])}</li>' for d in state['decisions'])
    metrics_rows = ''.join(f'<tr><td>{e(k)}</td><td>{e(str(v))}</td></tr>' for k, v in state.get('metrics', {}).items())
    metrics = f'<section><h2>关键数字</h2><table><tr><th>指标</th><th>数值</th></tr>{metrics_rows}</table></section>' if state.get('metrics') else ''
    next_steps = ''.join(f'<li>{e(s)}</li>' for s in state.get('next', []))
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(state["title"])}</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f4;color:#243c32;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1100px;margin:auto;padding:36px 24px 70px}}h1{{font-size:28px;line-height:1.4}}h2{{font-size:20px;margin-top:0}}
section{{background:white;border:1px solid #dde5df;border-radius:12px;padding:22px 24px;margin:18px 0}}
table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #dfe5df}}th{{background:#f0f4f0}}
.tasks{{list-style:none;padding:0}}.tasks li{{padding:8px 0;border-bottom:1px dashed #e3e9e4}}
.done b{{color:#1c7a4d}}.doing b{{color:#b0761a}}.blocked b{{color:#b03030}}.todo b{{color:#7a8a80}}
.note{{background:#eaf4ed;border-left:4px solid #287653;padding:14px 16px;border-radius:0 8px 8px 0}}
.muted{{color:#617267;font-size:13px}}a{{color:#176a4d}}
header p.muted{{font-size:14px}}</style></head><body><main>
<header><h1>{e(state["title"])}</h1>
<p class="muted">更新于 {e(state["updated"])}　·　目标：{e(state["goal_short"])}　·　状态：{e(state["status_word"])}</p>
<p class="note">{e(state["banner"])}</p></header>
{''.join(phase_cards)}
{metrics}
<section><h2>决策记录（为什么这样做）</h2><ul>{decisions}</ul></section>
<section><h2>事件时间线</h2><table><tr><th>日期</th><th>事项</th><th>说明</th></tr>{events}</table></section>
<section><h2>下一步</h2><ol>{next_steps}</ol></section>
</main></body></html>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default='20261102')
    args = ap.parse_args()
    state_path = ROOT / 'out' / f'训练优化进度_{args.date}.state.json'
    state = json.loads(state_path.read_text())
    out = ROOT / 'out' / f'训练优化进度_{args.date}.html'
    out.write_text(render(state))
    print(out)


if __name__ == '__main__':
    main()
