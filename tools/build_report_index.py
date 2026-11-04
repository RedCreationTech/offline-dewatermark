"""Organize local reports into offline searchable tabs; never touch sealed runs."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from html import escape, unescape
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import shutil

mode_label_legacy = {'desktop': '桌面默认', 'desktop_protected': '桌面关闭补全', 'cli': 'CLI 默认'}
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out'
INDEX = OUT / '水印验证报告总目录.html'
CATEGORIES = {
    'baseline': ('观察基线', '保存当时版本的表现，用于后续逐张比较；记录完成不代表当前代码全量通过或去水印质量合格。'),
    'control': ('项目管控', '版本冻结、防退化规则、工具自检和工程记录。工具自检只证明控制链路可运行。'),
    'diagnosis': ('失败与诊断', '已记录的失败案例、定位分析与瓶颈判断；保留未修复和待验证事项。'),
    'experiment': ('算法实验', '定位、笔画补全及修复方法的实验对照。每轮的适用条件和验收结论以原报告为准。'),
    'history': ('历史回归', '此前样本回归、旧版本结果和错误示例。历史结果不能当作当前成绩或独立验证。'),
    'material': ('方案材料', '技术方案、可编辑 Word / PPT 和交付说明；材料更新不代表新增算法验证。'),
    'all': ('全部报告', '查看所有分类中的报告入口，或搜索标题、说明和目录路径。'),
}


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.links = []
        self.title = []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == 'title':
            self.in_title = True
        if tag == 'a':
            self.links.extend(value for key, value in attrs if key == 'href')

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False

    def handle_data(self, data):
        self.parts.append(data)
        if self.in_title:
            self.title.append(data)


def plain(value):
    parser = Text()
    parser.feed(value)
    return ' '.join(parser.parts)


def href(path):
    import os
    return quote(Path(os.path.relpath(path, OUT)).as_posix(), safe='/')


def clean_document_links(body):
    """Local document entry points end at the real extension; keep page-only anchors."""
    extensions = {'.html', '.htm', '.md', '.pdf', '.docx', '.pptx', '.xlsx', '.txt', '.json', '.csv'}
    def replace(match):
        value = unescape(match.group(2))
        parts = urlsplit(value)
        if parts.scheme not in ('', 'file') or not parts.path:
            return match.group(0)
        if Path(unquote(parts.path)).suffix.lower() not in extensions:
            return match.group(0)
        clean = urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))
        return 'href=' + match.group(1) + escape(clean, quote=True) + match.group(1)
    return re.sub(r'''href=(["'])(.*?)\1''', replace, body)


def available_local_links(body, base=None):
    """Keep missing files visible without publishing a broken clickable link.

    Retain the original target so a later rebuild can restore the same link
    when that exact file returns. Never edit or substitute historical assets.
    """
    base = OUT if base is None else Path(base)
    note = '<small data-missing-note="true">（本地文件缺失）</small>'

    def replace(match):
        attrs, inner = match.group(1), match.group(2)
        active = re.search(r'''(?<![\w-])href\s*=\s*(["'])(.*?)\1''', attrs)
        dormant = re.search(r'''data-missing-href\s*=\s*(["'])(.*?)\1''', attrs)
        original = active or dormant
        if original is None:
            return match.group(0)
        value = unescape(original.group(2))
        parts = urlsplit(value)
        if parts.scheme or not parts.path:
            return match.group(0)
        exists = (base / unquote(parts.path)).exists()
        if active is not None and exists:
            return match.group(0)
        if dormant is not None and not exists:
            return match.group(0) if note in inner else f'<a{attrs}>{inner}{note}</a>'
        attrs = attrs[:original.start()] + attrs[original.end():]
        attrs = re.sub(r'''\s+aria-disabled\s*=\s*(["'])true\1''', '', attrs)
        if exists:
            return f'<a{attrs} href="{escape(value, quote=True)}">{inner.replace(note, "")}</a>'
        return (f'<a{attrs} data-missing-href="{escape(value, quote=True)}" '
                f'aria-disabled="true">{inner}{note}</a>')

    return re.sub(r'<a\b([^>]*)>(.*?)</a>', replace, body, flags=re.S)


def category(value):
    if '完整基线效果总览' in value or '效果分档总览' in value:
        return 'baseline'
    if any(k in value for k in ('tender-', '方案材料｜', '答辩材料｜', 'proposal/output')):
        return 'material'
    if any(k in value for k in ('项目防退化控制', '分层防退化检查', '项目整理', '样本移除记录', 'project-control', '空间清理审计', '验收报告')):
        return 'control'
    if any(k in value for k in ('失败案例', 'bottleneck', 'failure-cases', '瓶颈判断', '漏检原因分析',
                                'batch_diagnosis', 'report_before.html', 'export_selection_diagnosis', '分步体检')):
        return 'diagnosis'
    if any(k in value for k in ('历史错误', '旧版错误', 'recall_priority', '历史示意版', '早期分析',
                                'center_refinement', 'new_samples', 'app_quick20', 'small_dataset_rerun',
                                'user_optimization', '当前默认基线', '指标数据｜', 'stroke_selection_20261028/final')):
        return 'history'
    return 'experiment'


def card(identifier, group, body):
    return dict(id=identifier, category=group, body=body)


def original_cards(source):
    # Subsequent rebuilds retain curated explanations and any newly added entries.
    matches = re.findall(r'<article\b([^>]*data-report-entry[^>]*)>(.*?)</article>', source, re.S)
    if matches:
        result = []
        for attrs, body in matches:
            ident = re.search(r'\bid="([^"]+)"', attrs).group(1)
            group = re.search(r'\bdata-category="([^"]+)"', attrs).group(1)
            body = re.sub(r'^\s*<span class="tag">.*?</span>', '', body, count=1, flags=re.S)
            result.append(card(ident, group, body))
        # Other report writers may still append ordinary flat sections.
        result.extend(original_cards_legacy(re.sub(r'<section\b[^>]*data-catalog-panel.*?</section>', '', source, flags=re.S)))
        return result
    return original_cards_legacy(source)


def original_cards_legacy(source):
    result = []
    for attrs, body in re.findall(r'<section\b([^>]*)>(.*?)</section>', source, re.S):
        if '<h2' not in body or 'data-report-entry' in body:
            continue
        match = re.search(r'\bid="([^"]+)"', attrs)
        identifier = match.group(1) if match else 'entry-' + hashlib.sha256(body.encode()).hexdigest()[:12]
        result.append(card(identifier, category(unquote(body) + identifier), body))
    return result


def baseline_card():
    pointer_path = ROOT / '.regression-control/baseline.json'
    if not pointer_path.is_file():
        return None, None  # 本机尚未建立观察基线，跳过基线卡片
    pointer = json.loads(pointer_path.read_text())
    directory = Path(pointer['path'])
    run = json.loads((directory / '运行记录.json').read_text())
    complete = run.get('complete') and run.get('integrity_verified') and (directory / '封存校验.json').is_file()
    rows = run['records']
    failures = sum(r['status'] != 'completed' for r in rows)
    finished = run.get('completed_at')
    date = datetime.strptime(finished, '%Y%m%d_%H%M%S_%f').strftime('%Y-%m-%d %H:%M') if finished else '尚未完成'
    modes = Counter(r['mode'] for r in rows)
    state = '已完成记录并封存' if complete else '记录尚未完整，不可验收'
    version = directory.name.rsplit('_', 1)[-1]
    overview = OUT / '完整基线效果总览_20261029_V1/去水印效果分档总览_20261029.html'
    summary_reports = sorted(directory.glob('项目防退化控制摘要报告_*.html'))
    detail_reports = sorted(directory.glob('项目防退化控制明细报告_*.html'))
    if summary_reports:
        entry, entry_label = summary_reports[-1], '摘要报告（结论与处理清单）'
    elif overview.exists():
        entry, entry_label = overview, '按效果分档查看'
    elif detail_reports:
        entry, entry_label = detail_reports[-1], '明细报告（逐张核查）'
    else:
        entry, entry_label = directory / '项目防退化控制报告.html', '逐张结果（旧版）'
    manifest = json.loads((directory / '原始清单.json').read_text())
    body = f'<h2><a href="{href(entry)}">完整观察基线 {date}｜{entry_label}</a></h2><p class="state">{state} · {date} · 北京时间</p>'
    if overview.exists():
        body += '<p>新增三档效果总览：效果较好、一般／待复核、问题较多。可查看实际原图与输出；分档仅为自动初筛。</p>'
        removal = ROOT / 'config/regression_retirements.json'
        if removal.exists():
            body += '<p>按用户要求移除3张手写样本后，当前效果页与后续回归为395个输入／423页／1,269页次；下列1,278页次是原始封存范围。样本变更不代表算法效果提升。</p>'
    if len(modes) > 1:
        mode_bits = '、'.join(f'{mode_label_legacy.get(m, m)} {n}' for m, n in sorted(modes.items())) + ' 页次'
    else:
        mode_bits = f'共 {len(rows):,} 页次'
    body += (f'<p>{manifest["total_inputs"]}个输入场景、{manifest["total_pages"]}张／页，'
             f'共 <strong>{len(rows):,}／{len(run["expected_keys"]):,} 页次</strong>。'
             f'运行或评分异常 {failures}；{mode_bits}。</p>'
             '<p><strong>观察基线记录当时表现，包括既有缺陷；不是当前工作版本全量通过、全部图片效果合格或独立泛化验证。</strong></p>')
    links = []
    summary_reports = sorted(directory.glob('项目防退化控制摘要报告_*.html'))
    detail_reports = sorted(directory.glob('项目防退化控制明细报告_*.html'))
    if summary_reports:
        links.append(('摘要报告（单文件可发送）', summary_reports[-1].name))
    if detail_reports:
        links.append(('明细报告（tab／搜索）', detail_reports[-1].name))
    legacy = directory / '项目防退化控制报告.html'
    if legacy.exists():
        links.append(('逐图报告（旧版）', '项目防退化控制报告.html'))
    links += [('运行记录与指标', '运行记录.json'),
              ('完整样本清单', '原始清单.json'), ('冻结验收规则', '验收规则.json'), ('封存校验', '封存校验.json')]
    body += '<p class="links">' + ' · '.join(f'<a href="{href(directory / filename)}">{label}</a>' for label, filename in links) + '</p>'
    return card('complete-observation-baseline-20261028-v3', 'baseline', body), directory


def discover(cards, baseline):
    linked = set()
    for item in cards:
        parser = Text()
        parser.feed(item['body'])
        for link in parser.links:
            if not urlsplit(link).scheme:
                linked.add((OUT / unquote(urlsplit(link).path)).resolve())
    additions = []
    for path in sorted(OUT.rglob('*.html')):
        if path.resolve() in linked or path == INDEX:
            continue
        # 防退化控制封存：未完整跑完的轮次不进总目录（避免半截报告冒充成绩）。
        if '项目防退化控制' in str(path):
            run_record = path.parent / '运行记录.json'
            if not run_record.exists() or not json.loads(run_record.read_text()).get('complete'):
                continue
        name = path.name
        if any(k in str(path) for k in ('/代码快照/', '/基线代码/', '/暂存代码快照/', '/可维护源文件/', '/原生编辑工作区/', '/样本移除留档/', '/整理前备份/', '/更新前备份/')):
            continue
        if any(k in name for k in ('入口片段', '备份', '留档', '修改前', '更新前', '添加方案入口前', '分类前')):
            continue
        if name in ('水印选区与处理结果复核.html', '官网列表.html'):
            continue
        if name == 'review.html' and any(k in str(path) for k in ('recall_priority', 'stroke_selection')):
            continue  # Batch drilldowns are reachable from their parent report.
        parser = Text()
        raw = path.read_text(errors='replace')
        parser.feed(raw)
        title = ''.join(parser.title).strip()
        # 防退化控制报告的标题补日期时间（封存件不改，只在目录显示名上补）。
        if name == '项目防退化控制报告.html':
            stamp = re.search(r'(\d{8}_\d{6})', path.parent.name)
            if stamp:
                moment = datetime.strptime(stamp.group(1), '%Y%m%d_%H%M%S')
                title = f'项目防退化控制报告 {moment:%Y-%m-%d %H:%M}'
        if not title or title == '水印验证报告总目录':
            continue
        relative = path.relative_to(OUT).as_posix()
        group = category(relative + title)
        if group == 'control' and '自检' in relative:
            title = '管控自检｜' + ('同版本对照' if '同版本' in relative else '冻结与三模式执行')
        if group == 'history' and title == '最小集回归结果':
            title = '历史固定回归｜' + path.parent.name
        identifier = 'report-' + hashlib.sha256(relative.encode()).hexdigest()[:12]
        external_images = re.findall(r'''src\s*=\s*(["'])(?!data:|https?:)([^"']+)\1''', raw)
        description = ('仅验证工具链路，不作为去水印质量成绩。' if '自检' in relative
                       else '补充收录的已有报告。样本、参数、代码版本和结论以原报告为准，不能直接并入当前成绩。')
        if external_images:
            description += ' 本报告引用外部图片文件，单独发送该 HTML 可能不完整。'
        body = (f'<h2><a href="{href(path)}">{escape(title)}</a></h2><p>{description}</p>'
                f'<p class="path">{escape(relative)}</p>')
        additions.append(card(identifier, group, body))
    return additions


CSS = '''
:root{--bg:#f3f5f3;--ink:#203a32;--muted:#5c7068;--line:#dce5df;--green:#176a4d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
.shell{max-width:1240px;margin:0 auto;padding:44px 32px 70px}.eyebrow{font-size:12px;letter-spacing:2px;font-weight:650;color:var(--green)}
h1{font-size:34px;line-height:1.3;margin:12px 0}header>p{color:var(--muted);max-width:870px;margin:10px 0}.timestamp{font-size:13px;color:var(--muted)}
.summary{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0}.summary span{padding:6px 12px;background:white;border:1px solid var(--line);border-radius:7px;font-size:13px}
.tabs{display:flex;gap:5px;overflow-x:auto;border-bottom:1px solid var(--line);padding-bottom:1px}.tabs button{white-space:nowrap;min-height:48px;padding:11px 15px;border:0;border-bottom:3px solid transparent;background:transparent;color:var(--muted);font:inherit;cursor:pointer}
.tabs button[aria-selected=true]{border-bottom-color:var(--green);color:var(--green);font-weight:650}.count{font-size:12px;margin-left:7px;opacity:.8}
button:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #5ba887;outline-offset:3px;border-radius:3px}
.toolbar{display:flex;justify-content:space-between;gap:22px;align-items:center;margin:25px 0 20px}.toolbar h2{font-size:20px;margin:0}.toolbar p{font-size:13px;color:var(--muted);margin:5px 0 0}
.search{display:flex;align-items:center;gap:8px;flex-shrink:0}.search input{width:280px;max-width:100%;border:1px solid #c7d5cc;border-radius:8px;padding:11px 12px;background:white;color:var(--ink);font:inherit}.search button{background:none;border:0;color:var(--green);font:inherit;cursor:pointer;padding:8px}
.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card{min-width:0;background:white;padding:24px;border:1px solid var(--line);border-radius:12px}
.card[data-category=baseline]{grid-column:1/-1;border-top:3px solid var(--green);padding:28px}.card h2{font-size:18px;line-height:1.55;margin:0 0 9px}.card p{margin:9px 0;font-size:14px}.card a{color:var(--green);text-underline-offset:3px;text-decoration-thickness:1px}.card h2 a{text-decoration:none}.card h2 a:hover{text-decoration:underline}
.card .path{font-size:11px;color:var(--muted);overflow-wrap:anywhere}.state{font-weight:650;color:var(--green)}.tag{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:1px 7px;font-size:11px;color:var(--muted);margin-bottom:12px}
.card a[aria-disabled=true]{color:var(--muted);text-decoration:none;cursor:default}.card small[data-missing-note]{color:#885c24}
[hidden]{display:none!important}.empty{padding:40px;background:white;border:1px dashed var(--line);border-radius:10px;text-align:center;color:var(--muted)}footer{border-top:1px solid var(--line);margin-top:32px;padding-top:16px;font-size:12px;color:var(--muted)}
@media(max-width:760px){.shell{padding:24px 16px 40px}h1{font-size:27px}.cards{grid-template-columns:1fr}.toolbar{flex-direction:column;align-items:stretch}.search{width:100%}.search input{width:100%}.search label{flex:1}.tabs button{padding:10px}.card{padding:19px}}
@media print{.tabs,.search{display:none}.card[hidden]{display:block!important}.cards{display:block}.card{break-inside:avoid;margin:12px 0}.shell{padding:0}}
'''

JS = '''
(() => {
  const buttons = [...document.querySelectorAll('[role="tab"]')];
  const cards = [...document.querySelectorAll('[data-report-entry]')];
  const search = document.querySelector('#report-search');
  const title = document.querySelector('#category-heading');
  const description = document.querySelector('#category-description');
  const panel = document.querySelector('#report-panel');
  const result = document.querySelector('#result-count');
  const empty = document.querySelector('#empty-state');
  let category = 'baseline';
  function filter(writeHash = true) {
    const button = buttons.find(item => item.dataset.category === category) || buttons[0];
    category = button.dataset.category;
    buttons.forEach(item => {
      const active = item === button;
      item.setAttribute('aria-selected', String(active));
      item.tabIndex = active ? 0 : -1;
    });
    panel.setAttribute('aria-labelledby', button.id);
    title.textContent = button.dataset.title;
    description.textContent = button.dataset.description;
    const query = search.value.trim().toLocaleLowerCase();
    let visible = 0;
    cards.forEach(card => {
      const match = (category === 'all' || card.dataset.category === category) &&
        (!query || card.textContent.toLocaleLowerCase().includes(query));
      card.hidden = !match;
      visible += match ? 1 : 0;
    });
    result.textContent = `显示 ${visible} 个入口`;
    empty.hidden = visible !== 0;
    if (writeHash) {
      const state = new URLSearchParams({tab: category});
      if (query) state.set('q', search.value);
      history.replaceState(null, '', '#' + state.toString());
    }
  }
  function readHash() {
    const hash = location.hash.slice(1);
    if (hash.includes('=')) {
      const state = new URLSearchParams(hash);
      category = state.get('tab') || 'baseline';
      search.value = state.get('q') || '';
    } else if (hash) {
      const target = document.getElementById(decodeURIComponent(hash));
      const card = target?.closest('[data-report-entry]');
      if (card) category = card.dataset.category;
    }
    filter(false);
    if (hash && !hash.includes('=')) document.getElementById(decodeURIComponent(hash))?.scrollIntoView();
  }
  buttons.forEach((button, index) => {
    button.addEventListener('click', () => {category = button.dataset.category; filter();});
    button.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % buttons.length;
      else if (event.key === 'ArrowLeft') next = (index + buttons.length - 1) % buttons.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = buttons.length - 1;
      else return;
      event.preventDefault();
      buttons[next].focus();
      buttons[next].click();
    });
  });
  search.addEventListener('input', () => filter());
  document.querySelector('#clear-search').addEventListener('click', () => {search.value = ''; filter(); search.focus();});
  window.addEventListener('hashchange', readHash);
  readHash();
})();
'''


def main():
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    folder = OUT / '报告目录整理' / now.strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    if INDEX.is_file():
        source = INDEX.read_text()
        shutil.copyfile(INDEX, folder / '总目录_分类前备份.html')
    else:
        source = ''  # 首次在本机生成总目录，从空分类起步
    curated = original_cards(source)
    # 丢弃指向未完整跑完的防退化控制封存的旧条目（半截轮次不是成绩）。
    def links_incomplete_freeze(body: str) -> bool:
        for link in re.findall(r'href="([^"]+)"', body):
            decoded = unquote(link)
            if '项目防退化控制' not in decoded:
                continue
            target = (OUT / urlsplit(decoded).path).resolve()
            record = target.parent / '运行记录.json'
            if record.exists() and not json.loads(record.read_text()).get('complete'):
                return True
        return False
    curated = [item for item in curated if not links_incomplete_freeze(item['body'])]
    # 丢弃主链接目标已不存在的旧条目（如按修订删除的旧版报告），避免总目录残留失效入口。
    def primary_link_missing(body: str) -> bool:
        links = re.findall(r'href="([^"]+)"', body)
        if not links:
            return False
        target = (OUT / urlsplit(unquote(links[0])).path).resolve()
        return not target.is_file()
    curated = [item for item in curated if not primary_link_missing(item['body'])]
    baseline, directory = baseline_card()
    sealed_report_hash = None
    report = None
    if directory is not None:
        report = directory / '项目防退化控制报告.html'
        if report.exists():
            sealed_report_hash = hashlib.sha256(report.read_bytes()).hexdigest()
    # Current full observation supersedes the old wording, not the old files.
    for item in curated:
        if '当前默认基线｜两批14张' in item['body']:
            item['body'] = item['body'].replace('当前默认基线｜两批14张', '历史开发基线｜两批14张')
    cards = ([baseline] if baseline else []) + [item for item in curated if not baseline or item['id'] != baseline['id']]
    current_id = 'current-project-organization-20261029'
    current_body = ('<h2>' + '<a href="' + href(ROOT / 'docs/项目文档总览.html') + '">当前说明｜项目状态、数据与工具入口（2026-10-29）</a></h2>'
                    '<p>先看当前状态，再选择观察基线、实验或历史记录。数据数量从实际清单生成；本轮只整理入口和管理，不形成新的算法效果成绩。</p>'
                    '<p>' + '<a href="' + href(ROOT / 'docs/项目现状与后续开发说明.html') + '">当前状态与待处理事项</a> · '
                    '<a href="' + href(ROOT / 'docs/代码与工具用途说明.html') + '">代码、工具与备份</a></p>')
    cards = [item for item in cards if item['id'] != current_id] + [card(current_id, 'control', current_body)]
    failure_record = ROOT / 'docs/去水印失败案例记录.html'
    if failure_record.is_file() and 'id="baseline-review-20261029"' in failure_record.read_text():
        identifier = 'failure-baseline-review-20261029'
        body = ('<h2><a href="' + href(failure_record) + '">'
                '失败案例更新｜完整观察基线问题归纳与6组代表证据（2026-10-29）</a></h2>'
                '<p>保留原有11个历史案例，补充空选区、重建差异、内容损伤、模式双向影响和修复范围问题；'
                '区分已确认现象、实现风险与待验证原因。仅分析保存结果，未修改算法或重跑推理。</p>')
        cards = [item for item in cards if item['id'] != identifier] + [card(identifier, 'diagnosis', body)]
    for item in cards:
        item['body'] = available_local_links(clean_document_links(item['body']))
    cards += discover(cards, directory)
    unique = {item['id']: item for item in cards}
    cards = list(unique.values())
    for item in cards:
        item['body'] = available_local_links(clean_document_links(item['body']))

    # 全部入口按时间倒序：最新在前。日期取条目正文中的第一个日期（标题/状态行），
    # 无日期时退回链接目标的文件修改时间，都没有的排最后。
    def card_moment(item):
        import datetime as dt
        from urllib.parse import unquote
        # 只从可见文字取日期，避免 href 里目录名（运行开始时间）干扰标题时间。
        text = re.sub(r'<[^>]+>', ' ', item['body'])
        match = re.search(r'(20\d{2})[-_年.]?(\d{1,2})[-_月.]?(\d{1,2})(?:[日_ \u3000]+(\d{1,2})[:：](\d{2}))?', text)
        if match:
            try:
                return dt.datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)),
                                   int(match.group(4) or 0), int(match.group(5) or 0))
            except ValueError:
                pass
        link = re.search(r'href="([^"]+)"', text)
        if link:
            target = (OUT / unquote(link.group(1))).resolve()
            if target.is_file():
                return dt.datetime.fromtimestamp(target.stat().st_mtime)
        return dt.datetime.min
    cards.sort(key=card_moment, reverse=True)
    counts = Counter(item['category'] for item in cards)
    baseline_summary = '<span>历史完整观察基线：1,278 页次已记录</span>' if baseline else '<span>本机尚未建立观察基线</span>'
    initial = 'baseline' if baseline else 'all'
    js = JS.replace("let category = 'baseline';", f"let category = '{initial}';")
    tabs = ''.join(f'<button id="tab-{key}" type="button" role="tab" aria-controls="report-panel" aria-selected="{str(key == initial).lower()}" tabindex="{0 if key == initial else -1}" data-category="{key}" data-title="{label}" data-description="{escape(description, quote=True)}">{label}<span class="count">{len(cards) if key == "all" else counts[key]}</span></button>' for key, (label, description) in CATEGORIES.items())
    body = []
    for item in cards:
        label = CATEGORIES[item['category']][0]
        body.append(f'<article id="{item["id"]}" class="card" data-report-entry="true" data-category="{item["category"]}"><span class="tag">{label}</span>{item["body"]}</article>')
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>水印验证报告总目录</title><style>{CSS}</style></head>
<body><main class="shell"><header><div class="eyebrow">离线智能去水印 · 报告中心</div><h1>水印验证报告总目录</h1><p>按用途查看观察基线、项目管控、失败诊断、算法实验、历史回归与方案材料。保留每轮原始证据，历史结果和未通过实验分别标明。</p>
<!-- report-timestamp:start -->
<p class="timestamp" data-report-timestamp="updated">目录更新时间：<time datetime="{now.isoformat(timespec='seconds')}">{now.strftime('%Y-%m-%d %H:%M:%S')}</time> · 北京时间</p>
<!-- report-timestamp:end -->
<div class="summary"><span>{len(cards)} 个报告与材料入口</span>{baseline_summary}<span>记录完成 ≠ 当前代码验收通过</span></div><p><a href="{href(ROOT / 'docs/项目现状与后续开发说明.html')}">查看当前状态与清单问题</a> · <a href="{href(ROOT / 'docs/项目文档总览.html')}">查看文档用途</a></p></header>
<nav class="tabs" role="tablist" aria-label="报告分类">{tabs}</nav>
<section id="report-panel" data-catalog-panel="true" role="tabpanel" aria-labelledby="tab-{initial}"><div class="toolbar"><div><h2 id="category-heading">全部报告</h2><p id="category-description">按上方标签切换分类。</p><p id="result-count" role="status" aria-live="polite">共 {len(cards)} 个入口</p></div><div class="search"><label><span class="sr-label" hidden>搜索当前分类的报告</span><input id="report-search" type="search" aria-label="搜索当前分类的报告" placeholder="在当前分类搜索报告…"></label><button id="clear-search" type="button">清空</button></div></div>
<div class="cards">{''.join(body)}</div><p id="empty-state" class="empty" hidden>当前分类没有匹配的报告。可清空搜索或切换“全部报告”。</p></section>
<noscript><p>JavaScript 未启用，以下已展示全部报告；浏览器查找仍可使用。</p></noscript><footer>本页与报告均从本地文件打开，无需联网。点击标题查看完整报告；子批次、逐图输出和原始数据从对应主报告继续进入。目录分类不改变原始报告、实验结论或封存记录。</footer></main><script>{js}</script></body></html>'''
    INDEX.write_text(page)
    if sealed_report_hash is not None and hashlib.sha256(report.read_bytes()).hexdigest() != sealed_report_hash:
        raise RuntimeError('封存报告不应被目录整理修改')
    save = dict(updated_at=now.isoformat(), entries=len(cards), by_category=dict(counts),
                baseline_report_unchanged=True, backup=str(folder / '总目录_分类前备份.html') if INDEX.is_file() else None,
                preserved_original_entries=len(curated), added_entries=len(cards) - len(curated))
    (folder / '目录整理记录.json').write_text(json.dumps(save, ensure_ascii=False, indent=2))
    print(json.dumps(save, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
