"""生成每日工作日报 HTML：读取 pi 会话记录统计 + 人工撰写的当日总结。

用法：
    .venv/bin/python tools/build_daily_summary.py            # 今天
    .venv/bin/python tools/build_daily_summary.py --date 2026-11-02

约定（AGENTS.md 第 8 节）：中文命名带日期；正文全平铺不用折叠列表；
术语解释只通过词旁 ❓ 浮层（点击弹出，再点一次、点空白处或按 Esc 收起）；
页面自包含（图片若引用一律 base64 内嵌）；生成后须用 tools/build_report_index.py 重建总目录。
历史日期的人工总结存在 DAILY_SUMMARIES 里，只增不改；统计部分始终从会话文件实时读取。
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from html import escape
import json
from pathlib import Path

SESSIONS_ROOT = Path.home() / '.pi' / 'agent' / 'sessions'
PROJECT_CWD = '/Users/a123/Desktop/offline-dewatermark-ai'
OUT = Path(__file__).resolve().parents[1] / 'out'

# ---------------------------------------------------------------- 术语浮层
GLOSSARY = [
    ('独立验证集', '一组从未参与过调参的测试图（本项目为 66 张），专门用来阶段性证明"改动是真的进步"而不是只对调参过的样本有效。规则要求看过并据其调过算法后，它就要降级为开发用途，不能继续宣称盲测。'),
    ('开发集', '反复用来发现问题、调参数的少量样本（本项目 14 张真实图）。它"小且有偏"是性质不是缺陷——调参只需要方向，不需要统计显著。'),
    ('固定回归集', '一组固定不变的测试输入，作用是"哨兵"：每次改动后重跑，发现"修好了 A 却修坏了 B"的退化。它的价值在固定二字，不随版本改动。'),
    ('召回率', '真实水印笔画中，有多少比例被程序选中了。越高 = 漏掉的水印越少。'),
    ('精确率', '程序选中的像素里，有多少真的是水印。越低 = 误伤的正常画面越多。'),
    ('负例', '明确不含水印的图片。处理这类图的正确行为是零改动；负例受损说明程序在凭空制造水印。'),
    ('基线封存', '用 freeze-baseline 把当前版本在全部样本上的表现完整跑一遍并封存，之后每次测试都与它对比，回答"这次改动是变好还是变坏"。本机此前一直没有这份封存。'),
    ('笔画补全', '"补全水印笔画"功能：根据颜色关系和重复出现的字形，把被复杂背景"吃掉"的水印笔画找回来。桌面上默认开启。'),
    ('基础选区', '关闭笔画补全时，检测程序自己找到的水印区域。增强路径的结果不应比它差，它是召回的"地板"。'),
    ('逐图验证回退', '给笔画补全加的一道"对账"：补全结果能解释基础选区的比例够高才采用；不够就退回基础结果，并如实记录"这张图补全未帮上忙"。止血性质，不解决补全在纹理照片上找不到笔画的深层病根。'),
    ('色带选区', '把整条文字行涂成实心带状的选区形态。覆盖全但把字间空白也涂上了，按项目选区规则属于违规形态，只是部分照片碰巧修得干净。'),
    ('alpha 真值', '合成样本生成时记录的"每个像素被水印盖住多少"的参考答案，0–255。只在推理结束后评分时才能读，检测算法绝不允许偷看，否则等于抄答案。'),
    ('感知哈希', '一种对图片内容算"指纹"的方法：两张图肉眼看差不多，指纹就差不多。本次用它确认新抓的底图与旧底图不是换皮重复。'),
    ('处理模式', '回归只跑桌面默认一种配置：文字识别与补全水印笔画始终开启，用户无需选择。命令行入口参数与桌面一致。'),
]

GLOSS_JS = json.dumps([{'name': n, 'text': e} for n, e in GLOSSARY], ensure_ascii=False)
TERMS = {name: f'<span class="term" data-gid="{i}">{name}<span class="q">?</span></span>'
         for i, (name, _e) in enumerate(GLOSSARY)}


def t(name: str) -> str:
    return TERMS[name]


# ---------------------------------------------------------------- 当日人工总结
# key 为日期字符串 'YYYY-MM-DD'。只增不改；当日事实以会话记录和磁盘产物为准。
DAILY_SUMMARIES: dict[str, dict] = {
    '2026-11-02': {
        'title': '去水印项目工作日报',
        'one_liner': '今天是"测量地基 + 止血修复 + 管控纪律"三线并进的一天：独立验证集建成并跑出首轮基线；'
                     '笔画补全开着的丢水印缺陷完成根因定位并开始实施止血修复；检测提速入库推送；'
                     '产品交互方向（删参数面板、全自动）完成方案裁决但尚未实施。',
        'threads': [
            {
                'name': '① 独立验证集建成并基线化（测量地基）',
                'points': [
                    f'新建 <b>data/watermark_independent_v1/</b>：66 条（60 正例 + 6 {t("负例")}）、12 类场景、'
                    '26 个全新底图（新抓 12 张 CC0 照片，按标题/sha256/' + t('感知哈希') + '与旧数据三重去重）、'
                    '每条带精确 ' + t('alpha 真值') + '。',
                    'QA 全通过：与训练集及 14 张开发图零重叠；过程中拦下一个真实泄漏风险'
                    '（底图复用旧版式导致近重复，MAD 0.74），改版式重建后三项 MAD 距离全部超阈值。',
                    f'一次性基线跑完（66/66、0 错误、负例 6/6 零改动）：{t("召回率")}(alpha&gt;0) <b>33.9%</b>、'
                    '明显笔画召回 <b>38.8%</b>、' + t('精确率') + ' <b>67.3%</b>。最差三类：白色半透明大字、'
                    '透明几何 Logo、局部文字。报告已重做为"先结论、再榜单、后明细"的快速阅读版，'
                    '并补充指标口径说明（A/B/C/D 四格像素计数法）。',
                    '补齐了缺失的 data/应用测试输入（409 个文件、395 个输入场景/423 页，与登记口径核对一致），'
                    '并建成每夜巡检工具 tools/nightly_check.py（第一夜已跑完：23/50 页通过、负例安全）。',
                    f'<b>{t("基线封存")} freeze-baseline 已启动但未完成</b>（52/300 时按你要求停下，改约晚间再跑）。',
                ],
            },
            {
                'name': '② inc11 笔画补全丢选区：诊断完成，止血修复开工',
                'points': [
                    '用你提供的两张截图精确复现：同一平铺水印照片，补全开（默认）选区从 34.92% 缩到 8.79%，'
                    f'{t("召回率")} 82.24% → <b>53.16%</b>，丢的全是真实笔画，实锤缺陷。',
                    '根因（代码证据）：wm/recall.py:128-133 对三种重复文字水印，把' + t('基础选区') +
                    '<b>整体替换</b>为补全结果，而非并集或验证——基础结果成了"说扔就扔"。',
                    '修复探索被数据严格约束：五条路线（阈值分层、颜色准入、降匹配阈值、几何外推、锚定弱证据）'
                    '全部被实验证伪（纹理区定位信号三源全灭，找回上限约 64%）；简单并集等于把' + t('色带选区') +
                    '制度化，违反选区规则。',
                    '结论：唯一有数据支撑的落地措施是' + t('逐图验证回退') + '（对账不及格就退回基础结果并如实标记），'
                    '属止血；深层课题已立案为"为纹理区水印笔画寻找新的定位证据"。',
                    '产出诊断报告 v1.2（自包含单文件、术语 ❓ 浮层、正文平铺），沉淀了两条新约束进 AGENTS.md：'
                    'HTML 图片必须 base64 内嵌且入总目录；术语解释只走 ❓ 浮层不设词表。',
                    '实现已开始：给补全入口加"覆盖率对账 + 回退标记"，基础结果恢复为召回地板。',
                ],
            },
            {
                'name': '③ 检测提速与工程纪律（已入库推送）',
                'points': [
                    '提速改动：四角 OCR 探测合并为一次 + 两处重复计算消除；探测集 10 页结果逐字节不变、'
                    '角标样本页耗时约 4.2s→1.9s、单测全绿。剩余每页 3~7 次识别是各通道的检出证据，'
                    '再砍要拿召回率换，建议不砍，留你决策。',
                    '剩余提速空间、作废检查重跑等均已如实登记；已知证据足够支撑改动保留。',
                    '三个提交已推送远端（main 与 origin/main 一致）：修复前证据闸门+提速（流程图 V1.25）、'
                    'HTML 报告约定与总目录工具、分步体检工具。个人性清单文件和 7 个来源待确认的工具脚本留本地未提交。',
                    '新增工作纪律约束（AGENTS.md 第 1 节）：提问时只回答不动代码；'
                    '曾因误敲 git stash 导致几项检查跑的是原始代码，已全部恢复并声明作废范围。',
                ],
            },
            {
                'name': '④ 产品交互方向：方案裁决完成，实施未开始',
                'points': [
                    '你的方向定案：应用应"给一张图、出一张净图"，参数面板不该由用户调。'
                    '经多轮 grill（反问澄清）后逐项裁决：OCR/补全两个技术开关删除并永远启用；样式选择器保留'
                    '（默认自动）；检测下限/笔画连接/修复外扩三个滑条保留；内部喂模参数沉为常量。',
                    '明确未做：面板删改涉及口径变更，必须与基线重建捆绑执行；且应先完成 inc11 止血修复，'
                    '不能把已知缺陷冻成新基线起点。',
                    '三模式回归保留的理据说清：测的是"开关/入口没坏"，不是担保用户调参效果。',
                ],
            },
            {
                'name': '⑤ 知识问答（不计改动）',
                'points': [
                    '解释了 8% 阈值与 40% 选区守卫的来源：均出自 2026-10-19 首个原型提交，是合成集上的工程初值，'
                    '代码注释自标"须在真实样本上重拟合"；UI 滑块只是把已有算法常量暴露出来，因果方向相反。',
                    '解释了 alpha 真值、色带为何只管行不管笔画、补全为何在文档场景发光在照片场景失明等产品逻辑。',
                ],
            },
        ],
        'metrics': [
            ('独立集首轮基线召回率', '33.9%', 'alpha>0 全笔画口径；明显笔画口径 38.8%'),
            ('独立集首轮基线精确率', '67.3%', '误伤背景仍是主要短板'),
            ('独立集负例', '6/6 零改动', '无水印图片全部安全'),
            ('inc11 复现（补全开/关）', '53.2% / 82.2%', '缺陷样本的召回率对比'),
            ('角标页检测耗时', '4.2s → 1.9s', '提速改动，结果逐字节不变'),
            ('git 提交并推送', '3 个', '闸门+提速 / 报告约定 / 体检工具'),
        ],
        'todo': [
            '<b>今晚</b>：freeze-baseline 重新开跑（约 8 小时；可先跑 tools/run_freeze_once.sh 或叫我启动），'
            '完成后装每夜巡检定时任务，进入"早看摘要、白天修复"的节奏。',
            '完成 inc11' + t('逐图验证回退') + '实现与验证（覆盖率对账阈值需跨样式样本标定，防止为一张图调参）。',
            '面板删改 + 新基线重建捆绑执行（等止血修复落地后）。',
            '独立集基线暴露的最差三类（白色半透明大字、透明几何 Logo、局部文字）进入失败驱动迭代清单。',
            '深层课题"纹理区笔画定位新证据"另行立项；7 个来源待确认的工具脚本等你确认后决定是否入库。',
        ],
    },
}


# ---------------------------------------------------------------- 会话统计（自动读取）
def read_sessions(day: str) -> list[dict]:
    """读取当天全部会话的统计与对话摘要。"""
    cwd_dir = SESSIONS_ROOT / ('--' + PROJECT_CWD.strip('/').replace('/', '-') + '--')
    stats = []
    for path in sorted(cwd_dir.glob(f'{day}T*.jsonl')):
        users, assistants, first_ts, last_ts = [], [], None, None
        with path.open(encoding='utf-8') as fp:
            for line in fp:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = obj.get('timestamp')
                if ts:
                    last_ts = ts
                    first_ts = first_ts or ts
                if obj.get('type') != 'message':
                    continue
                msg = obj.get('message', {})
                content = msg.get('content')
                if isinstance(content, list):
                    texts = [c.get('text', '') for c in content
                             if isinstance(c, dict) and c.get('type') == 'text']
                    content = '\n'.join(texts)
                if not content:
                    continue
                content = str(content).strip()
                if msg.get('role') == 'user':
                    if content.startswith('<') or 'tool_result' in content[:30]:
                        continue
                    users.append(content)
                elif msg.get('role') == 'assistant':
                    assistants.append(content)
        stats.append({
            'file': path.name,
            'start': first_ts or '',
            'end': last_ts or '',
            'user_turns': len(users),
            'assistant_turns': len(assistants),
            'user_chars': sum(len(u) for u in users),
            'assistant_chars': sum(len(a) for a in assistants),
            'users': users,
        })
    return stats


def hhmm(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso.replace('Z', '+00:00')).strftime('%H:%M')
    except ValueError:
        return iso[11:16] if len(iso) > 16 else iso


def esc(s: str) -> str:
    return escape(s, quote=False).replace('\n', '<br>')


def render(day: str, summary: dict, sessions: list[dict]) -> Path:
    total_turns = sum(s['user_turns'] for s in sessions)
    starts = [s['start'] for s in sessions if s['start']]
    ends = [s['end'] for s in sessions if s['end']]
    span = f"{hhmm(min(starts))} – {hhmm(max(ends))}" if starts else '—'

    stat_cards = [
        ('会话数', f'{len(sessions)}', '今日 pi 会话文件数'),
        ('对话轮次', f'{total_turns}', '用户提问/指令总数'),
        ('工作时段', span, '首个会话开始至最后活动（UTC+0 显示）'),
        ('关键产出', '5 项', '独立集 / 诊断报告 / 3 提交 / 巡检工具 / 进度页'),
        ('遗留阻塞', '2 项', '基线封存未完成、止血修复待验证'),
    ]
    cards = ''.join(
        f'<div class="card"><div class="num">{escape(v)}</div>'
        f'<div class="lbl">{escape(k)}</div><div class="sub">{escape(s)}</div></div>'
        for k, v, s in stat_cards)

    threads = ''.join(
        f'<h3>{th["name"]}</h3><ul>' + ''.join(f'<li>{p}</li>' for p in th['points']) + '</ul>'
        for th in summary['threads'])

    metric_rows = ''.join(
        f'<tr><td>{escape(k)}</td><td class="numcell">{escape(v)}</td><td>{escape(n)}</td></tr>'
        for k, v, n in summary['metrics'])
    metrics = (f'<table><thead><tr><th>指标</th><th>数值</th><th>说明</th></tr></thead>'
               f'<tbody>{metric_rows}</tbody></table>')

    todo = ''.join(f'<li>{item}</li>' for item in summary['todo'])

    # 各会话纪要：时间 + 用户提问清单（截断展示）
    session_blocks = []
    for s in sessions:
        qs = ''.join(f'<li>{esc(u[:160])}{"…" if len(u) > 160 else ""}</li>' for u in s['users'][:14])
        more = f'<li class="more">（其余 {len(s["users"]) - 14} 轮从略，详见会话文件）</li>' \
            if len(s['users']) > 14 else ''
        session_blocks.append(
            f'<div class="sess"><div class="sess-head">会话 {hhmm(s["start"])} 起 ｜ '
            f'用户 {s["user_turns"]} 轮 ｜ 助手 {s["assistant_turns"]} 轮 ｜ '
            f'文件 <code>{escape(s["file"][:19])}…</code></div>'
            f'<ul class="qs">{qs}{more}</ul></div>')
    sessions_html = ''.join(session_blocks)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{escape(summary['title'])} {day}</title>
<style>
body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; margin: 0 auto;
  max-width: 980px; padding: 24px; color: #222; line-height: 1.75; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }}
h2 {{ font-size: 18px; margin-top: 34px; padding-top: 12px; border-top: 2px solid #e3efe8; }}
h3 {{ font-size: 15.5px; margin: 18px 0 6px; color: #175c3a; }}
.meta {{ color: #666; font-size: 13px; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0; }}
.card {{ flex: 1 1 150px; background: #f4f9f6; border: 1px solid #d8e8de; border-radius: 10px;
  padding: 12px 14px; }}
.card .num {{ font-size: 22px; font-weight: 700; color: #175c3a; }}
.card .lbl {{ font-size: 13px; margin-top: 2px; }}
.card .sub {{ font-size: 12px; color: #888; margin-top: 2px; }}
.plain {{ background: #f0f7f3; border-left: 4px solid #2a7; padding: 10px 14px; margin: 10px 0; }}
.warn {{ background: #fdf5ec; border-left: 4px solid #e80; padding: 8px 12px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f4f7f5; }}
.numcell {{ font-weight: 700; color: #175c3a; white-space: nowrap; }}
ul {{ padding-left: 22px; margin: 6px 0; }}
li {{ margin: 5px 0; }}
.sess {{ border: 1px solid #e3e3e3; border-radius: 8px; padding: 10px 14px; margin: 10px 0; }}
.sess-head {{ font-size: 13px; color: #555; margin-bottom: 6px; }}
.qs {{ font-size: 13.5px; color: #333; }}
.qs .more {{ color: #999; }}
code {{ background: #f4f4f4; border-radius: 4px; padding: 1px 5px; font-size: 12.5px; }}
.term {{ border-bottom: 1px dotted #2a7; cursor: pointer; }}
.term .q {{ display: inline-block; width: 14px; height: 14px; line-height: 14px; text-align: center;
  font-size: 10px; background: #2a7; color: #fff; border-radius: 50%; margin-left: 3px; vertical-align: super; }}
#gloss-pop {{ display: none; position: fixed; background: #fff; border: 1px solid #b8d8c6; border-radius: 8px;
  box-shadow: 0 6px 20px rgba(0,0,0,.18); padding: 10px 14px; z-index: 99; font-size: 13px; line-height: 1.6;
  max-width: 420px; }}
#gloss-pop.show {{ display: block; }}
#gloss-pop b {{ color: #175c3a; display: block; margin-bottom: 4px; }}
#gloss-pop p {{ margin: 0; color: #333; }}
</style>
</head>
<body>
<h1>{escape(summary['title'])}（{day}）</h1>
<p class="meta">版本 v1.0 ｜ 生成时间 {datetime.now():%Y-%m-%d %H:%M} ｜ 数据来源：本机 pi 会话记录（{len(sessions)} 个会话文件）及项目磁盘产物 ｜ 性质：工作纪要（非效果放行报告）<br>
阅读提示：术语旁的小问号 <span class="q" style="vertical-align:super;">?</span> 点击弹出解释，再点一次、点页面空白处或按 Esc 收起。</p>

<h2>一、今日概览</h2>
<div class="cards">{cards}</div>
<div class="plain"><b>一句话总结：</b>{summary['one_liner']}</div>

<h2>二、今日工作主线</h2>
{threads}

<h2>三、关键数字</h2>
{metrics}
<p class="meta">注：独立集基线数字为"锁柜"前的一次性测量（2026-11-02 轮），仅作后续迭代的对照起点；inc11 复现数字来自缺陷诊断的本地重跑。以上均非当前代码的全量放行成绩。</p>

<h2>四、待办与阻塞</h2>
<ul>{todo}</ul>
<div class="warn"><b>口径提醒：</b>基线封存（freeze-baseline）今日未完成，"新旧对比"基准尚未建立；
止血修复（{t('逐图验证回退')}）尚未跑完跨样式验证，不能宣称缺陷已修复。以上两项完成前，本周期的效果结论以诊断报告为准。</div>

<h2>五、今日会话纪要（按时间）</h2>
<p class="meta">以下为各会话中你的原始提问/指令（按序截断展示），用于回溯当天讨论脉络；完整内容见会话文件。</p>
{sessions_html}

<p class="meta" style="margin-top:36px">修订记录：v1.0（{day}）初版，由 tools/build_daily_summary.py 生成。<br>
生成约定：正文全平铺；术语解释只通过词旁 ❓ 浮层提供；页面自包含，可单文件发送。</p>

<div id="gloss-pop"><b id="gloss-name"></b><p id="gloss-text"></p></div>
<script>
const GLOSS = {GLOSS_JS};
const pop = document.getElementById('gloss-pop');
const popName = document.getElementById('gloss-name');
const popText = document.getElementById('gloss-text');
let popFor = null;
function hidePop() {{ pop.classList.remove('show'); popFor = null; }}
function showPop(term) {{
  const g = GLOSS[Number(term.dataset.gid)];
  popName.textContent = g.name;
  popText.textContent = g.text;
  pop.classList.add('show');
  popFor = term;
  const r = term.getBoundingClientRect();
  const pw = pop.offsetWidth, ph = pop.offsetHeight;
  let x = r.left + r.width / 2 - pw / 2;
  x = Math.max(10, Math.min(x, window.innerWidth - pw - 10));
  let y = r.bottom + 8;
  if (y + ph > window.innerHeight - 10) y = Math.max(10, r.top - ph - 8);
  pop.style.left = x + 'px';
  pop.style.top = y + 'px';
}}
document.addEventListener('click', e => {{
  const term = e.target.closest('.term');
  if (term) {{
    e.stopPropagation();
    if (popFor === term) {{ hidePop(); return; }}
    showPop(term);
    return;
  }}
  if (!e.target.closest('#gloss-pop')) hidePop();
}});
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') hidePop(); }});
window.addEventListener('scroll', hidePop, true);
</script>
</body>
</html>"""
    out_dir = OUT / '日报'
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f'工作日报_{day}.html'
    target.write_text(html, encoding='utf-8')
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description='生成每日工作日报 HTML')
    parser.add_argument('--date', default=date.today().isoformat(), help='日报日期 YYYY-MM-DD（默认今天）')
    args = parser.parse_args()
    summary = DAILY_SUMMARIES.get(args.date)
    if summary is None:
        known = ', '.join(sorted(DAILY_SUMMARIES))
        raise SystemExit(f'暂无 {args.date} 的人工总结（已有：{known}）。请先在 tools/build_daily_summary.py '
                         f'的 DAILY_SUMMARIES 中补充当日总结，再重新生成；会话统计部分会自动读取。')
    sessions = read_sessions(args.date)
    if not sessions:
        raise SystemExit(f'{SESSIONS_ROOT} 下没有找到 {args.date} 的会话文件。')
    target = render(args.date, summary, sessions)
    print(f'已生成: {target} ({target.stat().st_size // 1024} KB)，会话 {len(sessions)} 个')


if __name__ == '__main__':
    main()
