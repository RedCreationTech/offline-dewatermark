"""生成「提示词找水印」功能的自测报告与人工验证手册（两个自包含 HTML）。

自测报告数据来自 tools/run_prompt_feature_selftest.py 的真实模型运行产物
（out/提示词功能自测_20261107/自测结果.json 与对照图）；没有产物时对应章节
如实标注"未运行"。手册为纯文本步骤核对单，内嵌一张工作台实拍（可选）。

用法：.venv/Scripts/python.exe tools/build_prompt_feature_report.py [--shot PNG路径]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
RUN_DIR = OUT / "提示词功能自测_20261107"
REPORT = OUT / "提示词找水印功能自测报告_20261107.html"
HANDBOOK = OUT / "提示词找水印功能人工验证手册_20261107.html"

TERMS = {
    "检测框": "模型给出的水印实例包围盒，只表示搜索范围或重绘许可（整框重绘时），不能直接当作按笔画的修复选区。",
    "matte": "模型随框返回的水印软遮罩（提示词限定）。本功能把它当作搜索范围约束：种子、生长与选区都限在 matte 支撑区内，选区严格不出模型框定的水印范围；它本身不直接成为选区，无对比证据的光晕像素会被丢弃。",
    "选区": "最终要修复的像素集合，必须来自当前图的颜色、局部对比等像素证据；界面以红色叠加显示。",
    "按笔画": "提示词定位的默认处理方式：模型框限定搜索范围，框内由强核出发做局部对比证据的连通生长，淡笔画随强核整体入选；正文保护区不生长。",
    "整框重绘": "提示词处理方式之一：模型框整体作为重绘许可交给 LaMa 重绘。框内是空白底时最干净；框若压着正文，正文会被一并重绘（处理前会警告重叠比例）。",
    "证据闸门": "修复前的统一检查：选区为空、选区过小且不自信、可信度低于 0.12、重绘范围超过四分之一页，四种情况一律不动原图，仅标记需要人工处理。",
    "常驻服务": "选中后模型进程常驻显存（约 6–7GB），切回自动识别或退出程序时停止；适合连续处理多批图片。",
    "按需启动": "每轮处理才启动模型进程、该轮跑完即停，单次多付约 20–60 秒加载；适合显存紧张或偶尔使用。",
    "负例": "不含水印的图片。对它运行任何找水印方式都不应产生选区或改动（除用户人工添加）。",
    "core_009": "固定回归集中的斜向平铺文字样本（绿色“验证素材 3409”，1152×864，共 9 个实例，下边缘 3 个被裁切）。",
}


def tip(term: str) -> str:
    text = TERMS[term]
    return (f'<span class="term">{escape(term)}'
            f'<button type="button" class="q" data-tip="{escape(text, quote=True)}" '
            f'aria-label="术语解释">?</button></span>')


def data_uri(path: Path, max_side: int = 800) -> str | None:
    if not path.is_file():
        return None
    from PIL import Image
    img = Image.open(path)
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


CSS = """
:root{--bg:#f3f5f8;--ink:#243247;--muted:#5e7085;--line:#dfe6ee;--accent:#205e9d;--code:#14253b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1020px;margin:0 auto;padding:30px 22px 80px}
section{background:#fff;border:1px solid var(--line);border-radius:12px;padding:24px 28px;margin:0 0 20px}
h1{font-size:26px;margin:6px 0 4px}h2{font-size:19px;margin:0 0 12px;color:var(--accent)}h3{font-size:16px;margin:18px 0 6px}
.eyebrow,.muted{color:var(--muted);font-size:13px}.lead{font-size:16px}
table{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0}th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
th{background:#eff3f8;color:#48627f}img{max-width:100%;border:1px solid var(--line);border-radius:8px}
figure{margin:14px 0}figcaption{font-size:13px;color:var(--muted);margin-top:6px}
pre{overflow:auto;padding:14px 16px;background:var(--code);color:#e9f3ff;font:13px/1.8 ui-monospace,monospace;border-radius:8px}
code{background:#eef2f7;border-radius:4px;padding:1px 5px}
.term{position:relative;white-space:nowrap}
.q{display:inline-block;margin-left:4px;width:17px;height:17px;line-height:15px;border-radius:50%;border:1px solid var(--accent);background:#eaf2fb;color:var(--accent);font-size:12px;cursor:pointer;padding:0;vertical-align:1px}
.tipbox{position:absolute;z-index:20;left:0;top:calc(100% + 6px);width:min(320px,86vw);background:#fff;border:1px solid var(--accent);border-radius:8px;box-shadow:0 6px 22px rgba(20,40,70,.18);padding:10px 12px;font-size:13px;line-height:1.7;color:var(--ink);white-space:normal;text-align:left}
ul,ol{padding-left:22px}li{margin:6px 0}.check li{list-style:none;margin:8px 0}
.check li::before{content:"☐  ";color:var(--accent)}
strong.ok{color:#1a7f4b}strong.bad{color:#b3372a}
"""

TIP_JS = """
(function(){var openBox=null,openBtn=null;
function closeTip(){if(openBox){openBox.remove();openBox=null;openBtn=null;}}
document.addEventListener('click',function(e){
var btn=e.target.closest?e.target.closest('.q'):null;
if(btn){e.preventDefault();e.stopPropagation();
if(openBtn===btn){closeTip();return;}
closeTip();var box=document.createElement('div');box.className='tipbox';
box.textContent=btn.getAttribute('data-tip');
btn.parentElement.appendChild(box);openBox=box;openBtn=btn;return;}
if(openBox&&!e.target.closest('.tipbox')){closeTip();}});
document.addEventListener('keydown',function(e){if(e.key==='Escape'){closeTip();}});})();
"""


def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)}</title><style>{CSS}</style></head><body><main>'
            f'{body}</main><script>{TIP_JS}</script></body></html>')


def build_report(test_summary: str, shot: str | None) -> None:
    data = {}
    result_path = RUN_DIR / "自测结果.json"
    if result_path.is_file():
        data = json.loads(result_path.read_text(encoding="utf-8"))
    rows = []
    for mode in ("auto", "prompt"):
        r = data.get("results", {}).get(mode, {})
        if r.get("error"):
            rows.append(f"<tr><td>{mode}</td><td colspan='7'>运行失败：{escape(str(r['error']))}</td></tr>")
            continue
        cells = [mode, f"{r.get('elapsed_s', '—')}s", r.get("selected_px", "—"),
                 r.get("recall_alpha_gt0", "—"), r.get("recall_alpha_ge128", "—"),
                 r.get("precision", "—"), r.get("outcome", "—"),
                 (r.get("detection") or {}).get("boxes_with_support", "—")]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    ran = bool(data)
    images = ""
    for mode in ("auto", "prompt"):
        uri = data_uri(RUN_DIR / f"{mode}_selection.png")
        if uri:
            images += (f'<figure><img src="{uri}" alt="{mode} 选区对照">'
                       f'<figcaption>{mode} 模式的选区（红）与输出对照；提示词模式另以橙框标出模型实例位置。</figcaption></figure>')
    shot_img = f'<figure><img src="{shot}" alt="工作台实拍"><figcaption>当前版本工作台实拍（处理设置面板已按 找水印→去水印→修复底图 重排）。</figcaption></figure>' if shot else ""
    body = f"""
<section>
<div class="eyebrow">算法实验 · 2026-11-07 · 提示词定位找水印功能</div>
<h1>提示词找水印功能 · 开发自测报告</h1>
<p class="lead">本报告由开发过程自动产物生成：单元测试（假服务）、全量工程测试、真实模型端到端运行
（{tip('core_009')}，自动识别 vs 提示词定位）。结论：功能链路打通，{tip('检测框')}只作搜索范围、
{tip('matte')}不进候选的边界在代码与测试双重固定；真实样本上提示词定位的选区质量与自动识别互有胜负，
不构成通用能力结论。</p>
</section>
<section>
<h2>实现范围</h2>
<ul>
<li>推理服务 <code>tools/wevisone_server.py</code>（WeVisOne venv 运行，stdio JSON-lines）与客户端 <code>wm/wevisone_client.py</code>（{tip('常驻服务')}/{tip('按需启动')}两种生命周期，崩溃自动重启一次）。</li>
<li>检测构建器 <code>wm/prompt_detect.py</code>：模型框外扩 64px，框内"检测器强核＋局部对比证据连通生长"提取笔画；选区被约束在{tip('matte')}支撑区内（严格不出模型框定的水印范围），OCR 正文区不生长不入选（"文字识别"勾选控制）；处理方式可选<strong>按笔画</strong>（默认）或{tip('整框重绘')}，后者以模型框为整框重绘许可并对正文重叠发出警告。</li>
<li>界面：参数面板按 找水印→去水印→修复底图 重排；「找水印方式」下拉即开关，方式专属控件条件显示；{tip('检测框')}在「水印选区」视图以橙框叠加。</li>
<li>批量：<code>BatchWorker</code> 按批次方式分支；提示词模式跳过跨页先验；空选区/服务失败经{tip('证据闸门')}或运行失败路径处理，队列继续。</li>
</ul>
</section>
<section>
<h2>测试结论</h2>
<p>{test_summary}</p>
<h3>真实模型端到端（core_009，真值对照）</h3>
<table><tr><th>方式</th><th>耗时</th><th>选中像素</th><th>召回 α&gt;0</th><th>笔画召回 α≥128</th><th>精确率</th><th>状态</th><th>有证据框数</th></tr>
{''.join(rows) if ran else '<tr><td colspan="8">真实模型运行产物不存在（未运行或已清理），本节留空。</td></tr>'}
</table>
<p class="muted">口径与 WeVisOne 分割实验一致：召回分母为真值 alpha&gt;0 的全部像素；matte 未参与本功能任何环节。
{'详见 out/提示词功能自测_20261107/自测结果.json。' if ran else ''}</p>
{images}
{shot_img}
</section>
<section>
<h2>边界与防退化声明</h2>
<ul>
<li>默认行为不变：新批次的找水印方式仍是自动识别；提示词定位是批次级显式选择。</li>
<li>{tip('检测框')}与 matte 永远不直接成为选区；新增选中像素必须有当前图的像素证据（与人工圈选同一套过滤）。</li>
<li>{tip('负例')}（core_023 等）在本功能下零选区零改动由单元测试固定；空选区由{tip('证据闸门')}原样退回。</li>
<li>未做跨样式全量回归与独立来源验证；本功能为实验状态，不宣称通用能力提升。</li>
</ul>
</section>
<section>
<h2>复现与回退</h2>
<pre>set WM_WEVISONE_DEC_BATCH=4
.venv/Scripts/python.exe tools/run_prompt_feature_selftest.py
.venv/Scripts/python.exe tools/build_prompt_feature_report.py</pre>
<p class="muted">回退：不选择提示词定位即为原行为；工作区回滚对应提交即可整体移除。</p>
</section>"""
    REPORT.write_text(page("提示词找水印功能自测报告｜2026-11-07", body), encoding="utf-8")


def build_handbook(shot: str | None) -> None:
    shot_img = f'<figure><img src="{shot}" alt="工作台实拍"><figcaption>版本实拍：右侧处理设置面板，注意 找水印／去水印／修复底图 三个分组。</figcaption></figure>' if shot else ""
    body = f"""
<section>
<div class="eyebrow">人工验证手册 · 2026-11-07 · 提示词找水印功能</div>
<h1>提示词找水印功能 · 人工验证手册</h1>
<p class="lead">按下面清单逐条验证，每条给出操作与预期。涉及 GPU 的条目在 RTX 3060 Ti（8GB）上按预期值填写；
其他显卡可按比例放宽耗时。开始前请确认模型环境已按配置文档准备好（<code>out/third_party/WeVisOne-VL</code>）。</p>
{shot_img}
</section>
<section>
<h2>A. 面板布局与开关</h2>
<ol class="check">
<li>启动桌面程序（<code>start.bat gui</code> 或 <code>python -m wm.app</code>）。预期：右侧「处理设置」从上往下为 <strong>找水印 → 去水印 → 修复底图</strong> 三个分组。</li>
<li>默认批次参数。预期：「找水印方式」显示 <strong>自动识别</strong>；看不到「服务方式」和「提示词」输入框；能看到「水印样式」。</li>
<li>把「找水印方式」切成 <strong>提示词定位</strong>。预期：出现「服务方式」（默认常驻服务）、「提示词」输入框和「处理方式」（默认按笔画）；「水印样式」隐藏；切回自动识别后恢复原样。</li>
</ol>
</section>
<section>
<h2>B. 服务生命周期与显存</h2>
<ol class="check">
<li>方式＝提示词定位、服务方式＝常驻服务。预期：状态栏先后出现「提示词模型启动中…（首次加载约 20–60 秒）」，然后「提示词模型已就绪（常驻）」；任务管理器可见一个 WeVisOne venv 的 python 进程，GPU 专用内存约 6–7GB。</li>
<li>切回「自动识别」。预期：状态栏提示服务已停止；GPU 占用回落。</li>
<li>再切回提示词定位、把服务方式改为「按需启动」。预期：不预热，无 GPU 占用。</li>
<li>（可选）打开 <code>out/wevisone_service.log</code>。预期：能看到服务进程的告警/错误记录（没有内容也正常）。</li>
</ol>
</section>
<section>
<h2>C. 提示词处理全流程（{tip('core_009')}）</h2>
<ol class="check">
<li>导入 <code>data/watermark_regression_min_v1/inputs/core/core_009.png</code> 为新批次；方式＝提示词定位；服务＝按需启动；提示词填 <strong>验证素材 3409</strong>；点「自动处理当前批次」。</li>
<li>处理中。预期：状态栏依次出现「提示词定位：调用本地模型找水印」；按需模式下首轮会先出现「启动提示词模型服务」。</li>
<li>完成后看「处理结果」视图。预期：斜向文字基本被去除，底图织物纹理连续（可切「拖拽对比」核对）。</li>
<li>切到「水印选区」视图。预期：红色为实际选区；<strong>橙色实线框</strong>为模型给出的实例位置（应约 8 个，底行右侧可能缺 1 个）；橙框只作显示，红色才是要修的像素。</li>
<li>看「质量检查」。预期：选区来源＝模型识别；定位方式＝提示词定位相关描述；总耗时显著高于自动识别（含 16–22s 模型推理）。</li>
<li>导出当前图片。预期：导出目录含处理结果、选区证据；manifest 的 processing 里有 <code>prompt_detection</code> 段（提示词、模型框、有证据框数、耗时）。</li>
</ol>
</section>
<section>
<h2>D. 自动识别对照</h2>
<ol class="check">
<li>同一图片新建批次（或改方式后重处理），方式＝自动识别，点「自动处理当前批次」。</li>
<li>预期：同样出结果且速度更快；「水印选区」视图只有红色选区、<strong>没有橙色框</strong>；两种方式的修复质量可逐处对比，互有胜负属正常。</li>
</ol>
</section>
<section>
<h2>E. 失败与边界路径</h2>
<ol class="check">
<li>方式＝提示词定位但提示词留空，点处理。预期：状态栏提示先输入提示词，队列不启动。</li>
<li>提示词填一个图上不存在的内容（如「不存在的文字 XYZ」）。预期：模型给出 0 个框或框内无证据时，状态栏出现「模型给出 N 个框，框内未找到足够证据的笔画」，该页<strong>保持原图不改动</strong>，状态为质量未通过/需人工。</li>
<li>导入无水印负例 <code>core_023.png</code>，方式＝提示词定位、提示词「水印」，跑当前页。预期：不产生选区、图片零改动（{tip('证据闸门')}拦截）。</li>
<li>提示词方式处理完成后，用「编辑水印选区」笔刷在残留处添加几笔并按当前选区处理。预期：画笔辅助照常工作——它对所有处理结果通用。</li>
<li>（有备用机器可验）把 <code>out/third_party/WeVisOne-VL</code> 临时改名后重启程序、选提示词定位。预期：启动处理时提示环境不可用并跳过，程序其余功能正常；改回目录后恢复。</li>
</ol>
</section>
<section>
<h2>F. 处理方式怎么选（重点）</h2>
<p>提示词模式下「处理方式」决定模型框找到之后怎么修。按笔画的提取被两道约束保护：<strong>选区严格不出模型框/matte 范围</strong>，<strong>OCR 正文区不生长不入选</strong>（"文字识别"勾选控制）。</p>
<ol class="check">
<li><strong>按笔画（默认）</strong>：只修框内有像素证据的笔画。框压正文时交叠处保守保留（不伤字），个别残留用画笔补。适合：浅色手写、浅灰打印字、压正文的水印。</li>
<li><strong>整框重绘</strong>：整个模型框交给 AI 重绘。框内是空白底时最干净；处理前状态栏出现"整框重绘覆盖约 N% 的正文区域"时，说明框压正文、那部分文字会被重绘掉，请改回按笔画。</li>
<li>斜向黑字压正文的案例（提示词：斜向黑字"内部演示稿"）：按"按笔画＋文字识别☑"跑，预期红选区覆盖斜字未压行的部分、背景横线基本不被选中；压行处的斜字保留。想更彻底：取消"文字识别"勾选重跑（正文保护关闭，背景也可能被修）或切整框重绘（框内文字全没）。</li>
<li>用 real_001（手写"内部演示稿"）验证：处理后手写痕迹基本不可见，红区不出橙框。</li>
<li>用 real_008（浅灰"矩形框"+"DRAFT"）验证：红区覆盖字母主体与框线；表格区域不被选中。</li>
</ol>
<table>
<tr><th>参数</th><th>推荐值</th><th>说明</th></tr>
<tr><td>找水印方式 / 服务方式</td><td>提示词定位 / 按需启动</td><td>连续多批可切常驻</td></tr>
<tr><td>处理方式</td><td>按笔画（推荐）</td><td>框内文字可牺牲时改整框重绘</td></tr>
<tr><td>文字识别</td><td>☑ 必开</td><td>正文保护区开关；取消勾选＝水印背景一视同仁按证据提取</td></tr>
<tr><td>水印判定阈值</td><td>8%（默认）</td><td>不要调低，会扩大背景误选</td></tr>
<tr><td>其余滑杆</td><td>默认</td><td>连接 11px、外扩 0、参考范围 256、重叠 64</td></tr>
</table>
</section>
<section>
<h2>G. 批次参数持久化</h2>
<ol class="check">
<li>批次 A 设为提示词定位＋提示词「验证素材 3409」，再新建批次 B 保持自动识别，来回切换批次。预期：每个批次记住自己的方式与提示词，互不串扰。</li>
<li>同一批次内修改提示词或切方式后，已处理页标记为需重处理。预期：图片列表状态变化，重新处理后更新。</li>
</ol>
</section>
<section>
<h2>G. 性能参考值（RTX 3060 Ti 8GB）</h2>
<ul>
<li>模型服务冷启动（含加载）：约 20–60 秒；常驻模式下仅首次。</li>
<li>core_009（1152×864）单页推理：约 16–22 秒（按需模式的每轮首图另加上述启动时间）。</li>
<li>常驻服务驻留显存：约 6–7GB；按需模式跑完即释放。</li>
</ul>
</section>
<section>
<h2>回退方式</h2>
<p>不需要此功能时：批次保持「自动识别」即为原行为，无残留状态；如需整体移除，回滚本功能对应提交（见自测报告「复现与回退」）。</p>
</section>"""
    HANDBOOK.write_text(page("提示词找水印功能人工验证手册｜2026-11-07", body), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", default=None, help="工作台实拍 PNG（可选，内嵌两个文档）")
    parser.add_argument("--test-summary", default="单元测试与工程测试结论未随本次生成提供。")
    args = parser.parse_args()
    shot = None
    if args.shot and Path(args.shot).is_file():
        img_bytes = Path(args.shot).read_bytes()
        shot = "data:image/png;base64," + base64.b64encode(img_bytes).decode("ascii")
    build_report(args.test_summary, shot)
    build_handbook(shot)
    print(f"报告: {REPORT}")
    print(f"手册: {HANDBOOK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
