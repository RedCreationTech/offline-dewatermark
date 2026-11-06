"""合并版周报：白话导读在前、正式报告在后，顶部导航互跳。

复用 build_weekly_plain_20261105.build_html 与 build_weekly_summary_20261105.build_html，
只取两者 <body> 内部内容，套统一模板。合并版入总目录；两个分册保持不动。
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out/周报'
STAMP = datetime.now().strftime('%Y-%m-%d %H:%M')
REPORT = OUT / f'本周工作总结_合并版_20261102-1105_{datetime.now().strftime("%H%M")}.html'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _body(html: str) -> str:
    return html.split('<body>', 1)[1].rsplit('</body>', 1)[0]


def main():
    plain = _body(_load('build_weekly_plain_20261105').build_html())
    formal = _body(_load('build_weekly_summary_20261105').build_html())
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>本周工作总结（合并版：白话导读+正式报告）（11-02~11-05） {STAMP}</title>
<style>
body{{font:15px/1.9 'PingFang SC','Hiragino Sans GB',system-ui,sans-serif;background:#eef2f6;color:#25394e;margin:0;padding:40px 20px}}
.wrap{{max-width:960px;margin:0 auto;background:#fff;border-radius:14px;box-shadow:0 2px 18px rgba(30,60,90,.08);padding:44px 52px;box-sizing:border-box}}
h1{{font-size:24px;text-align:center;margin:6px 0 4px;color:#1b3a54;letter-spacing:.5px}}
.subtitle{{text-align:center;color:#7a8a94;font-size:13.5px;margin-bottom:8px}}
.nav{{display:flex;gap:14px;justify-content:center;margin:20px 0 10px;flex-wrap:wrap}}
.nav a{{background:#2c6e9b;color:#fff;text-decoration:none;padding:9px 22px;border-radius:22px;font-size:14.5px}}
.nav a.alt{{background:#7a94a8}}
h2{{font-size:17.5px;color:#1b3a54;border-left:5px solid #2c6e9b;padding-left:12px;margin:38px 0 14px}}
h3{{font-size:15.5px;color:#2c6e9b;margin:22px 0 6px}}
table{{border-collapse:collapse;background:#fff;width:100%;font-size:13.8px;margin:14px 0;border-radius:8px;overflow:hidden}}
td,th{{border:1px solid #e2e8ef;padding:9px 13px;text-align:left;vertical-align:top}}
th{{background:#f2f6fa;color:#1b3a54;font-weight:600}}
tr:nth-child(even) td{{background:#fafcfe}}
tr:hover td{{background:#f0f6fb}}
.note,.warn,.good,.plain,.ana{{border-radius:10px;padding:14px 18px;font-size:14px;margin:14px 0;line-height:1.85}}
.note{{background:#fff8e6;border:1px solid #e8d9a0}}
.warn{{background:#fdecea;border:1px solid #e6b0aa}}
.good{{background:#eafaf1;border:1px solid #b7e4c7}}
.plain{{background:#f7fafc;border:1px solid #e2e8ef}}
.ana{{background:#fffdf5;border:1px solid #e8d9a0}}
.ok{{color:#1c7a4d;font-weight:600}} .bad{{color:#c0392b;font-weight:600}}
code{{background:#eef1f4;padding:1px 6px;border-radius:5px;font-size:12.8px}}
.meta{{color:#7a8a94;font-size:13px;text-align:center;margin-top:36px}}
.term{{border-bottom:1px dashed #2c6e9b;cursor:help;position:relative}}
.tip{{display:none;position:absolute;left:0;top:1.5em;z-index:9;width:340px;background:#fff;border:1px solid #b9cbda;border-radius:8px;
box-shadow:0 4px 14px rgba(30,60,90,.18);padding:10px 13px;font-size:13px;line-height:1.7;color:#33475b;cursor:default;font-weight:400}}
.term.open .tip{{display:block}}
.pri{{display:inline-block;min-width:64px;text-align:center;border-radius:6px;padding:2px 8px;font-size:13px;font-weight:600;white-space:nowrap}}
.p1{{background:#c0392b;color:#fff}} .p2{{background:#e67e22;color:#fff}} .p3{{background:#2c6e9b;color:#fff}} .px{{background:#95a5a6;color:#fff}}
em{{color:#c0392b;font-style:normal;font-weight:600}}
.divider{{border:none;border-top:2px dashed #cdd9e4;margin:46px 0}}
</style></head><body>
<p class="subtitle" style="text-align:center;margin-top:6px">{STAMP}</p>
<div class="nav"><a href="#plain">🍵 白话导读（先看这个）</a><a class="alt" href="#formal">📋 正式报告（细节与数字）</a></div>
<hr class="divider"/>
<section id="plain">{plain}</section>
<hr class="divider"/>
<section id="formal">{formal}</section>
<script>
document.addEventListener('click',function(e){{
  var t=e.target.closest('.term');
  document.querySelectorAll('.term.open').forEach(function(x){{if(x!==t)x.classList.remove('open');}});
  if(t)t.classList.toggle('open');
}});
document.addEventListener('keydown',function(e){{if(e.key==='Escape')document.querySelectorAll('.term.open').forEach(function(x){{x.classList.remove('open');}});}});
</script>
</body></html>"""
    REPORT.write_text(html, encoding='utf-8')
    print(f'written: {REPORT} ({len(html) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
