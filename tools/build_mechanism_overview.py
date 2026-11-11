"""生成《去水印机制与流程总览》HTML：汇总当前全部处理机制的内部说明页。

面向项目内部阅读理解：只讲机制分工、触发条件与能力边界，不写效果分数。
页面自包含（内联样式与脚本，无外部依赖），术语解释通过词旁 ❓ 浮层提供。

用法：
    .venv/bin/python tools/build_mechanism_overview.py            # 写入 docs/去水印机制与流程总览.html
    .venv/bin/python tools/build_mechanism_overview.py --out 路径  # 写到指定文件
内容以 docs/去水印处理流程图.md 的当前版本为准；处理流程变更后须同步修订本页并重新生成。
"""
from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'docs' / '去水印机制与流程总览.html'

CSS = '''
*{box-sizing:border-box}
body{margin:0;background:#f4f6f4;color:#243c32;font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
aside{position:fixed;top:0;left:0;bottom:0;width:230px;background:#1f3d31;color:#d8e6dd;padding:22px 14px;overflow:auto}
aside h2{font-size:15px;margin:0 0 10px;color:#fff}
aside a{display:block;color:#bcd6c6;text-decoration:none;padding:5px 8px;border-radius:6px;font-size:14px}
aside a:hover{background:#2a5040;color:#fff}
main{margin-left:230px;padding:34px 30px 80px;max-width:1080px}
h1{font-size:28px;line-height:1.4;margin:0 0 6px}
h2{font-size:21px;margin:0 0 14px;padding-bottom:8px;border-bottom:2px solid #cfe0d5}
section{background:#fff;border:1px solid #dde5df;border-radius:12px;padding:26px;margin:22px 0}
p{margin:10px 0}
ul,ol{margin:8px 0;padding-left:24px}
li{margin:6px 0}
a{color:#176a4d;text-underline-offset:3px}
table{width:100%;border-collapse:collapse;font-size:14px}
td,th{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #e2e8e3;overflow-wrap:anywhere}
th{background:#f0f4f0}
code{font-size:13px;background:#eef3ee;padding:1px 5px;border-radius:4px}
.meta{color:#617267;font-size:13px;margin:2px 0 18px}
.note{background:#eaf4ed;border-left:4px solid #287653;padding:14px 16px;border-radius:0 8px 8px 0}
.warn{background:#fdf3e7;border-left:4px solid #c07a1f;padding:14px 16px;border-radius:0 8px 8px 0}
h3{font-size:17px;margin:20px 0 6px}
h3 .q{font-weight:400;font-size:13px;color:#617267;margin-left:8px}
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:6px;margin:14px 0}
.step{flex:1 1 200px;min-width:180px;background:#f2f7f3;border:1px solid #cfe0d5;border-radius:10px;padding:12px 14px;position:relative}
.step b{display:block;font-size:15px;margin-bottom:4px}
.step .no{display:inline-block;background:#287653;color:#fff;border-radius:50%;width:22px;height:22px;line-height:22px;text-align:center;font-size:13px;margin-right:6px}
.step span{font-size:13px;color:#46584d}
.arrow{align-self:center;color:#287653;font-size:20px}
.badge{display:inline-block;font-size:12px;padding:1px 8px;border-radius:10px;white-space:nowrap}
.b-default{background:#dff0e5;color:#1d6a44;border:1px solid #b3d8c2}
.b-optional{background:#e3ecf7;color:#2c5587;border:1px solid #bfd2ea}
.b-manual{background:#f3e8f8;color:#6d3a8a;border:1px solid #dcc4ea}
.b-lab{background:#fbeaea;color:#a03a3a;border:1px solid #ecc4c4}
.term{position:relative;border-bottom:1px dashed #7fa38f;cursor:help;padding:0 1px}
.term .tip{display:none;position:absolute;left:0;bottom:calc(100% + 8px);width:300px;max-width:78vw;background:#20392e;color:#e8f2ec;font-size:13px;line-height:1.7;padding:12px 14px;border-radius:10px;z-index:30;box-shadow:0 6px 20px rgba(0,0,0,.25)}
.term .tip::after{content:"";position:absolute;top:100%;left:14px;border:7px solid transparent;border-top-color:#20392e}
.term.open .tip{display:block}
.qmark{color:#287653;font-size:12px;margin-left:2px}
footer{margin:26px 0 0;color:#617267;font-size:13px}
@media(max-width:800px){aside{display:none}main{margin-left:0;padding:20px 14px 60px}}
@media print{aside{display:none}main{margin:0;max-width:none}section{break-inside:avoid-page;border:none;margin:0;padding:14px 0}}
'''

JS = '''
document.addEventListener('click', function (event) {
  var term = event.target.closest ? event.target.closest('.term') : null;
  document.querySelectorAll('.term.open').forEach(function (node) {
    if (node !== term) node.classList.remove('open');
  });
  if (term) term.classList.toggle('open');
});
document.addEventListener('keydown', function (event) {
  if (event.key === 'Escape') document.querySelectorAll('.term.open').forEach(function (node) { node.classList.remove('open'); });
});
'''


def term(word, tip):
    """词旁 ❓ 浮层：点击弹出，再点一次、点空白处或按 Esc 收起。"""
    return (f'<span class="term" title="{escape(tip, quote=True)}">{escape(word)}'
            f'<span class="qmark">❓</span><span class="tip">{escape(tip)}</span></span>')


BADGE = {
    'default': '<span class="badge b-default">默认启用</span>',
    'optional': '<span class="badge b-optional">批次可选</span>',
    'manual': '<span class="badge b-manual">显式人工</span>',
    'lab': '<span class="badge b-lab">实验·默认关闭</span>',
}

SIDEBAR = [
    ('start', '从这里开始'),
    ('map', '一页看懂'),
    ('import', '导入与批次'),
    ('pdf', 'PDF 先试无损剥离'),
    ('detect', '找水印的两种方式'),
    ('crosspage', '多图辅助'),
    ('recall', '笔画补全'),
    ('gate', '修复前证据闸门'),
    ('repair', '修复分流'),
    ('check', '自动检查与待复核'),
    ('brush', '手动画笔修正'),
    ('export', '导出与追溯'),
    ('lab', '实验与试用机制'),
    ('control', '质量管控机制'),
    ('scores', '成绩在哪里看'),
    ('limits', '已知限制汇总'),
    ('about', '维护与版本'),
]

# 机制总表：机制 | 管什么 | 何时触发 | 状态
MECHANISMS = [
    ('批次与队列', '整批导入、按尺寸分组、暂停/停止、单张失败继续、整批进度与计时', '每次处理', 'default'),
    ('PDF 结构剥离', '水印作为独立叠加对象时，只省略显示指令、不重修页面', '仅未改动的 PDF 页，最先尝试', 'default'),
    ('规则点阵/重复纹理', '验证小点间距、形状跨区一致后按叠加关系还原', '自动识别中发现规律小点', 'default'),
    ('角标与深色大字', '按位置、形状和画面证据确定水印区域', '自动识别中发现角落编号或深色大字', 'default'),
    ('文字识别线索', '读出文字的位置、大小、倾斜，帮区分正文与水印', '文字识别可用时每页', 'default'),
    ('图内重复字形', '同一张图多处比对，确认实际可见的笔画', '检测到重复水印时', 'default'),
    ('透明叠加估计', '估计水印颜色和遮盖程度，为底色反推提供依据', '能从纸面、正文和水印颜色判断叠加关系时', 'default'),
    ('阴影彩色字分层还原', '彩色字形与错位阴影配准拟合，分层反解还原', '彩色大字带可核对错位阴影时', 'default'),
    ('日期/时间小行种子', '识别到日期时间样式小行后，仅选实测染色笔画做逐通道逆变换', '纸面页上的日期/时间样式行，且框内实测证据达标', 'default'),
    ('通用明暗颜色兜底', '在明亮纸面上找反复出现的灰度深浅和色差候选', '上面各通道都不适用时', 'default'),
    ('提示词定位', '本地模型按用户描述逐页找水印实例，模型框只作搜索范围', '批次显式选择该方式', 'optional'),
    ('提示词·按笔画', '框内由强笔画核连通生长选出淡笔画，受模型软遮罩和正文保护区约束', '提示词定位的默认处理方式', 'optional'),
    ('提示词·整框重绘', '模型框整体作为重绘许可，直接补图（正文可能被一并重绘）', '用户显式选择，自担正文风险', 'optional'),
    ('多图辅助', '同批同尺寸图互相比较，找反复出现的淡色覆盖，只作辅助', '同尺寸组有多张图时', 'default'),
    ('笔画补全', '沿已发现水印补回淡笔画，保留字内空洞和字间空白', '定位结果适合补全时', 'default'),
    ('修复前证据闸门', '选区为空、过小不自信、可信度太低或重绘范围过大时不动图', '每次修复前', 'default'),
    ('透明覆盖反解', '按估计的颜色和遮盖程度倒推被覆盖前的底色', '透明叠加可估计的水印', 'default'),
    ('周期纹理还原', '按测得的叠加关系还原小点/纹理，保留原亮度和纹理', '规则小点或重复纹理', 'default'),
    ('纸面/角标/重复字清理', '按路径估计的纸面底色清理', '简单纸面、角标或重复字', 'default'),
    ('照片细笔画就地修补', '用紧邻像素补齐较窄的缺口', '照片中的细小笔画', 'default'),
    ('LaMa 补图', '本地补图模型参考周围画面重画大区域，分块拼接', '照片大字及其他需要补图的区域', 'default'),
    ('通用混合路径', '先尝试透明反解，剩余部分再补图；范围过大时停止补图并提示复核', '各专门路径都不适用时', 'default'),
    ('自动检查', '检查耗时与选区周围异常改动；能估计才估计残留', '每张处理完后', 'default'),
    ('手动画笔修正', '添加/擦除完整选区后从原图重新修复，外扩为 0，越界拒绝', '人工显式操作，不参与自动检测', 'manual'),
    ('自动比较候选', '程序比较原方法与笔画补全再选路', '专用开关启动的试用入口', 'lab'),
    ('旧笔画保留候选', '补图路径中重验并保留旧明暗笔画', '专用开关，需重复字形等证据', 'lab'),
    ('文档错位点阵候选', '原规则点阵未找到时，测量错位小点周期再找', '研究工具显式启用', 'lab'),
    ('极淡区域染色反变换', '大面积低于常规检测下限的统一染色，整片代数逆变换', '用户选「极淡」样式提示或专用开关', 'lab'),
    ('实验性色彩细化', '颜色信号的多方向平衡检测', '默认关闭', 'lab'),
]

SECTIONS = [
    ('start', '从这里开始', '''
<p>这一页把当前去水印程序的<b>全部机制</b>汇总成一份内部能读懂的说明：每个机制<b>管什么事、什么时候触发、靠什么判断、保护什么、还不保证什么</b>。它只讲机制与边界，不写效果分数；分数与逐张证据在报告里，见「成绩在哪里看」。</p>
<div class="note">读法建议：第一节「一页看懂」用五分钟建立整体印象；后面每节对应一个机制，可以按需查阅；术语第一次出现时以词旁 <span class="qmark">❓</span> 浮层解释，点击弹出，再点一次、点页面空白处或按 Esc 收起。</div>
<p>页面上的机制分工以工程师维护的 <a href="去水印处理流程图.md">去水印处理流程图</a>（当前 V1.44）为准；那张图记录代码级细节与每次变更，这一页是它的通俗总览。两边说法不一致时，以流程图文档和实际代码为准，并应修订本页。</p>
'''),
    ('map', '一页看懂', '''
<p>一张图，从左到右就是程序处理一张图片的完整顺序：</p>
<div class="flow">
<div class="step"><b><span class="no">1</span>导入建批</b><span>选文件或整个文件夹，PDF 按页拆开，同批按尺寸分组排队</span></div><span class="arrow">→</span>
<div class="step"><b><span class="no">2</span>PDF 先试无损剥离</b><span>水印若是独立叠加对象，直接省略显示指令，不重画页面</span></div><span class="arrow">→</span>
<div class="step"><b><span class="no">3</span>找水印</b><span>两种方式二选一：自动识别（默认，内部按条件依次尝试多个通道）或提示词定位（本地模型按描述给框）</span></div><span class="arrow">→</span>
<div class="step"><b><span class="no">4</span>笔画补全与多图核对</b><span>沿已发现的水印补回淡笔画；同批重复出现的覆盖只作辅助，须核对后才能用</span></div><span class="arrow">→</span>
<div class="step"><b><span class="no">5</span>修复前证据闸门</b><span>选区为空、不可信或重绘范围过大：不动图，退回人工标记</span></div><span class="arrow">→</span>
<div class="step"><b><span class="no">6</span>按路径修复</b><span>透明反解、纹理还原、纸面清理、就地修补、LaMa 补图、通用混合，按定位结论分流，不是每条都跑</span></div><span class="arrow">→</span>
<div class="step"><b><span class="no">7</span>自动检查</b><span>查耗时与异常改动；能估计才估计残留；通过不等于去净</span></div><span class="arrow">→</span>
<div class="step"><b><span class="no">8</span>人工修正与导出</b><span>画笔补选区后从原图重修复；导出结果、选区和全部记录</span></div>
</div>
<h3>机制总表<span class="q">一行一个机制；「状态」区分默认启用、批次可选、显式人工和实验</span></h3>
__TABLE__
<p class="muted">上表就是当前的全部机制。带「实验·默认关闭」的一行一个机制，只在专用开关或研究工具里运行，不影响默认处理，详见「实验与试用机制」。</p>
'''),
    ('import', '一、导入与批次', f'''
<h3>解决什么问题</h3>
<p>把一批文件交给程序排队处理：支持多选文件或选择整个文件夹（递归包含所有子目录，按相对路径自然排序）；PDF 按页读取、图片保留彩色；同批文件尽量水印样式和位置相近，程序内部再按图片尺寸分组，同组共享部分分析结果以省时间。</p>
<h3>什么时候触发</h3>
<p>每次开始处理前。批次可指定处理范围（当前图片/当前批次/所有批次），参数默认在「自动」位，用户拖离自动位才算覆盖，只影响之后处理的图片。</p>
<h3>保护什么、不保证什么</h3>
<ul>
<li>暂停、停止在检查点生效，正在运行的模型调用不会立刻中断；停止保留已完成结果。</li>
<li>某张失败（模型缺失、文件损坏等）只记录原因，队列继续处理后面的。</li>
<li>界面的已处理张数、总用时、平均每张用时只统计最近一轮整批处理；暂停、排队、查看导出不计时。这组数字衡量速度，不代表效果。</li>
</ul>
'''),
    ('pdf', '二、PDF 先试无损剥离', f'''
<h3>解决什么问题</h3>
<p>{term('PDF 结构剥离', '不改动页面像素，而是找到 PDF 内部叠在内容上的独立文字/图片对象，把它的显示指令从这一页里省略掉；正文和底图保持原样，所以叫无损。')} PDF 里的水印常常是后叠上去的对象。程序先读这一页的内部结构，判断有没有「独立叠在内容上面、符合水印特征」的文字或图片对象；找到就只做省略显示指令的处理，不重画下面的内容。</p>
<h3>什么时候触发</h3>
<p>仅对来自原文件且未被修改的 PDF 页，在所有图片分析之前尝试。</p>
<h3>靠什么判断</h3>
<p>按透明度、几何位置和叠加关系判断对象是否独立叠加；注意这只证明「叠在上面」，不证明语义上一定是水印，所以处理后仍需人工复核是否删对。</p>
<h3>保护什么、不保证什么</h3>
<ul>
<li>嵌套对象等不支持的情况会自动转为图片方式处理。</li>
<li>结构剥离有结果后，这一页<b>不再</b>接着跑图片修复——页面上其他水印可能残留，需要人工查看。</li>
<li>手工编辑 PDF 页的选区时，从原始页面像素转入补图，不再是无损删除。</li>
</ul>
'''),
    ('detect', '三、找水印的两种方式', f'''
<p>按批次二选一，默认自动识别。</p>
<h3>方式一：自动识别（默认）</h3>
<p>结合原图的明暗、颜色和重复形状找水印。内部不是一套固定规则，而是<b>按条件依次尝试</b>的一组通道，得到适用结果通常就进入下一阶段，不是每张图都全部跑完：</p>
<ol>
<li><b>规则小点/重复纹理</b>：能验证规律小点或重复纹理时，确认点距、位置和形状在不同区域表现一致，再按叠加关系还原。</li>
<li><b>角落编号/深色大字</b>：能识别角落编号或适用的深色大字时，按位置、形状和画面证据确定水印区域。</li>
<li><b>{term('文字识别', '在本地读图里文字的位置、内容和倾斜方向的技术（常称 OCR）。这里只用它提供线索：读出的框只作搜索范围，不是最终水印选区。')}线索</b>：读出文字的位置、大小和倾斜方向，同时帮程序区分正文与水印。</li>
<li><b>图内重复字形</b>：图里有经过验证的重复字形时，比较同图多处，确认实际可见的笔画和位置。</li>
<li><b>重复文字水印</b>：文档里有适用的重复文字水印时，用较清楚的字形寻找其他重复位置，检查正文交叠处和图像边缘残笔。</li>
<li><b>透明叠加估计</b>：能从纸面、正文和水印颜色判断叠加关系时，估计水印的颜色和遮盖程度，为后续底色反推提供依据。</li>
<li><b>彩色大字与错位阴影</b>：纸面正文中出现大号彩色字且阴影可核对时，做配准与纸面核对；拟合收敛才{term('分层还原', '把彩色字形和它的错位阴影分别建模成两层，估计每层怎么叠上去，再按反方向一层层算回被覆盖前的样子。')}，验证失败就退回原阴影清理并记录原因。</li>
<li><b>日期/时间小行种子</b>：纸面页上文字识别读出日期/时间样式的小行时，框只作搜索范围；框内必须有实测染色笔画、不含深色墨迹，才把逐通道实测的水印颜色做{term('代数逆变换', '把「水印颜色按某个覆盖量盖在底色上」当作一条算式，实测出水印颜色和覆盖量后，反解出被盖之前的底色。只处理有实测证据的像素，不猜不填。')}修复。条件不达标自动放弃，页面行为和原流程一致。</li>
<li><b>通用明暗颜色兜底</b>：以上都不适用时，在明亮的纸面背景上估计纸张底色，寻找反复出现的灰度深浅，结合颜色、大小、形状避开识别到的正文和表格线；证据不充分时保留候选但标记低可信度。</li>
</ol>
<p class="muted">限定某种水印样式时，只尝试该样式允许的通道，全部失败可能返回空选区；文字识别模型缺失时，自动样式仍可走通用判断。</p>
<h3>方式二：提示词定位（批次可选）</h3>
<p>用户在批次里输入一句水印描述（如「验证素材 3409」），本地模型（WeVisOne）逐页给出水印实例的框。两个关键边界：</p>
<ul>
<li><b>模型框只作搜索范围</b>，不直接成为选区；默认的「按笔画」方式在框内由强笔画核做连通生长、选出淡笔画，并受模型软遮罩范围和正文保护区约束；粗大的深色笔画另有尺度通道处理，可延伸到框外。</li>
<li>「整框重绘」需用户显式选择：模型框并集就是重绘许可，框内正文会一并重绘，正文重叠较多时处理前明确警告。</li>
</ul>
<p>两种方式都先过修复前证据闸门，再用 LaMa 补图；模型服务异常会重启重试一次，仍失败则该页记为运行失败、队列继续。提示词模式跳过跨页辅助；PDF 结构剥离仍在最前。</p>
'''),
    ('crosspage', '四、多图辅助', f'''
<h3>解决什么问题</h3>
<p>{term('多图辅助', '同一批次、同一尺寸的图往往带同一水印。程序比较这些图相同位置的内容，找反复出现的淡色覆盖，帮单张图确认水印位置。')} 单张图看不清的淡水印，同批其他图可能看得更清。</p>
<h3>什么时候触发</h3>
<p>同批同尺寸组里有多张图时自动准备；结果可缓存复用，计算异常就放弃、退回单张分析。流式逐页处理和提示词定位不使用它。</p>
<h3>保护什么、不保证什么</h3>
<ul>
<li>页眉、表格、背景纹理同样会「每张图重复出现」，不能直接认作水印。</li>
<li>多图结果只作辅助：必须与当前图片自己的定位结果核对重合度、检查异常变大，核对通过才采用；没有独立水印证据时，不允许仅凭多图重复产生最终选区。</li>
</ul>
'''),
    ('recall', '五、笔画补全', f'''
<h3>解决什么问题</h3>
<p>{term('笔画补全', '定位阶段先找到的往往是水印里对比明显的部分。补全沿已发现的水印继续寻找颜色更淡的笔画，把漏掉的淡笔画补回选区。')} 水印笔画有深有浅，浅的容易漏。</p>
<h3>靠什么判断</h3>
<p>只按当前图的证据提取：文字框和推测的重复行只用来决定「去哪里找」，最终像素必须过颜色或明暗的画面证据；{term('字洞/字间空白', '文字笔画围出来的空心（字洞）和字与字之间的空隙。补全要求保留这些空洞，不允许把文字糊成实心块。')}必须保留。具体分支：</p>
<ul>
<li><b>斜向重复文字</b>：用原图把整条水印对齐，按多处重复笔画和可见边缘重新生成选区；补全或透明度拟合不成立时，退回「原定位选区 ∪ 对齐戳印」的并集并记录原因，不再整片丢弃已验证的弱实例；图边裁切的残笔按可见笔画支撑外推补回。</li>
<li><b>纸面重复文字</b>：并集补全只把局部对比证据漏掉的实例补回，自带实测颜色模型的图层不再被整框重绘替换；相邻行半周期错位时，用行内局部对比证据补回漏印的整行。</li>
<li>已有可信透明度还原或规则小点结果的图层不再被通用补全覆盖。</li>
</ul>
<h3>保护什么、不保证什么</h3>
<p>补全以「尽可能覆盖真实笔画、减少背景误选」为目标，但画面证据不足时仍可能漏掉极淡笔画或误选少量背景；补全失败的页面会在记录里写明本次未解决。</p>
'''),
    ('gate', '六、修复前证据闸门', f'''
<h3>解决什么问题</h3>
<p>{term('证据闸门', '动手修图之前先检查选区靠不靠谱：选区为空、选区过小且不自信、可信度低于 0.12、重绘范围超过整页四分之一，这四种情况一律跳过修复。')} 「该转人工的页不能先动手再标记」。</p>
<h3>保护什么、不保证什么</h3>
<ul>
<li>被闸门拦下的页输出与原图逐字节一致，不改动任何像素，只标记「需要人工选区」。</li>
<li>闸门只挡住明显不可靠的选区；通过闸门的选区仍需后续自动检查和人工复核。</li>
</ul>
'''),
    ('repair', '七、修复分流', f'''
<h3>解决什么问题</h3>
<p>定位阶段已经判断了水印类型，修复按结论分流到对应办法，<b>不是每张图依次执行所有修复</b>：</p>
<table>
<thead><tr><th>修复路径</th><th>什么时候用</th><th>做法</th></tr></thead>
<tbody>
<tr><td>透明覆盖反解</td><td>透明叠加关系可估计的水印</td><td>按估计的颜色和遮盖程度倒推被覆盖前的底色；适用路径中保护正文和表格</td></tr>
<tr><td>周期纹理还原</td><td>规则小点或重复纹理</td><td>按测得的叠加关系还原，尽量保留原亮度和纹理；正文交叠处按对应办法保护</td></tr>
<tr><td>纸面/角标/重复字清理</td><td>简单纸面、角标或重复字</td><td>按该路径估计的纸面底色清理，保留路径能识别的正常内容</td></tr>
<tr><td>阴影分层还原</td><td>通过验证的彩色阴影文字</td><td>用本图拟合的颜色与错位阴影模型，只在许可选区内写回反解结果</td></tr>
<tr><td>照片细笔画就地修补</td><td>照片中的细小笔画</td><td>用紧邻像素补齐，适合较窄的缺口</td></tr>
<tr><td>{term('LaMa 补图', '本地运行的一个图像修复模型：参考选区周围的画面，把选区内重画出来。适合大区域，但不保证画回的内容与原图一致。')}</td><td>照片大字及其他需要补图的区域</td><td>按用户「外扩」设置决定参考范围，小区域补边、大区域分块拼接；只把允许修复的位置写回原图</td></tr>
<tr><td>通用混合路径</td><td>各专门路径都不适用时</td><td>先尝试透明反解，剩余部分再补图；候选补图范围过大时停止补图、降低可信度并提示复核</td></tr>
</tbody></table>
<h3>保护什么、不保证什么</h3>
<ul>
<li>{term('外扩', '选区之外再让出的一圈像素，只作为补图时的参考画面，不等于检测到的水印。外扩为 0 时，选区外的输出必须与输入逐像素一致。')}由用户参数控制；外扩为 0 时选区外逐像素不动。</li>
<li>补图是「参考周围重画」，可能留下痕迹，尤其是纹理、表格和文字密集区域；重复照片反解只写有实测透明度模型的像素，选区内未实测的确认笔画会改走保守补图并双标注。</li>
</ul>
'''),
    ('check', '八、自动检查与待复核', f'''
<h3>解决什么问题</h3>
<p>每张处理完后做自动体检：处理耗时是否异常、选区周围是否出现意料外的改动；能估计时估计水印残留，<b>不能可靠估计的项目记为「未评估」而不是零</b>。</p>
<h3>保护什么、不保证什么</h3>
<div class="warn">普通桌面处理没有干净原图作答案，自动检查<b>无法证明</b>正文和背景已完整恢复，也不能证明水印全部去净。「自动检查通过」不等于「去净」；待复核项目不计为已验证成功。淡笔画、边缘残字、正文交叠、误选背景和补图痕迹仍需人工在查看界面复核。</div>
'''),
    ('brush', '九、手动画笔修正', f'''
<h3>解决什么问题</h3>
<p>自动结果不满意时的人工兜底：在选区视图上用画笔添加/擦除，形成一份完整选区后，程序从原图按这份选区重新修复（不重新检测）。同一像素最后一笔优先；字洞和字间空白不会被自动填充，画笔涂到背景也会获得修复权限。</p>
<h3>保护什么、不保证什么</h3>
<ul>
<li>这是自动处理后的<b>显式人工操作</b>，不参与默认全自动检测；未点击处理就切图/关闭，编辑直接丢弃，不静默保存。</li>
<li>手工选中不当作透明度依据，单独交给 LaMa；手工外扩固定为 0；修复越界即拒绝整次结果，不靠裁剪掩盖。</li>
<li>画笔能补自动漏检，但不能保证恢复被遮挡的正文；擦除交叠区域可能把水印一起留下。</li>
</ul>
'''),
    ('export', '十、导出与追溯', f'''
<h3>解决什么问题</h3>
<p>把「改了什么、凭什么改」完整存档：保存原图、处理结果、红色选区叠加图、黑白选区和选区强度；有相应数据时保存修复候选、正文保护区和手工完整选区（与实际修改范围分开保存）；记录处理参数、方法、耗时和异常。人工结果另存自动基线与每次完整选区，自动与人工分组统计。</p>
<h3>保护什么、不保证什么</h3>
<p>黑白选区和 <code>selection</code> 字段是完整修复权限；<code>mask_alpha</code> 只保留自动观测，不代表手工透明度。导出的是「实际修改范围」与「修复许可」两类概念，分开解释，不能互相顶替。</p>
'''),
    ('lab', '十一、实验与试用机制（默认关闭）', f'''
<p>以下机制真实存在，但<b>不在默认处理路径</b>里：它们只通过专用开关或研究工具运行，每条的完整验证结论以各自实验目录里的报告为准。共同点是：都有适用证据门槛、验证未通过或未完成前不接入默认、不更新观察基线。</p>
<table>
<thead><tr><th>实验机制</th><th>解决什么</th><th>入口与状态</th></tr></thead>
<tbody>
<tr><td>自动比较候选</td><td>由程序比较原方法与笔画补全再选路，通常只修复一次</td><td>桌面/CLI 的 <code>--experimental-auto-selection</code> 试用入口，未完成独立验证</td></tr>
<tr><td>旧笔画保留候选</td><td>补图路径中重验并保留旧的明暗笔画，减少补图损伤</td><td><code>--experimental-seed-preservation</code>，默认关闭，需重复字形等多重证据</td></tr>
<tr><td>文档错位点阵候选</td><td>原规则点阵找不到时，按错位小点的周期再找</td><td>研究工具显式启用；未达预登记门槛，未接入默认</td></tr>
<tr><td>极淡区域染色反变换</td><td>大面积低于常规检测下限的统一染色，整片代数逆变换</td><td>用户选「极淡」样式提示或专用开关；白纸纯度有硬门槛，候选仍为实验状态</td></tr>
<tr><td>实验性色彩细化</td><td>颜色信号的多方向平衡检测</td><td>默认关闭，未作为默认流程展示</td></tr>
</tbody></table>
<div class="note">登记这些实验不是承认它们可用：每一行都有「未通过/未完成」的门槛记录。默认路径上它们逐字节不受影响。</div>
'''),
    ('control', '十二、质量管控机制', f'''
<h3>解决什么问题</h3>
<p>回答「怎么知道这次修改没把别的样式改坏」。核心是<b>先冻结、再对比、门槛不让步</b>：</p>
<ul>
<li><b>分层检查</b>：日常改动先 <code>./start.sh test --plan</code> 预览范围，再 <code>./start.sh test</code> 按影响分层跑；局部通过不能更新默认基线。</li>
<li><b>发布放行</b>：<code>./start.sh test-release</code> 覆盖全部应用输入、历史样本与固定集的三模式完整对照；日常通过不能替代。</li>
<li><b>冻结与门槛</b>：验证前冻结代码、输入、模型、参数和验收规则；门槛预先登记，不允许看到结果后放宽、删失败样本或替换基线。</li>
<li><b>观察基线</b>：首次用 <code>./start.sh freeze-baseline</code> 建立，记录现状（包括缺陷）；只有完整门槛通过且显式 <code>control promote</code> 才更新指针，旧版本全部保留。</li>
<li><b>无水印对照</b>：保留在回归里专查误删；负例原本不变的输出不允许因新规则产生改动。</li>
<li><b>{term('独立验证', '用从未参与过调参的新样本做的一次性验证。按纪律，看过结果后该批样本就降级为开发/回归用途，不能再当作独立证据。')}纪律</b>：独立集一次性基线后锁柜；据其调参即降级，需另建独立集。</li>
</ul>
'''),
    ('scores', '十三、成绩在哪里看', f'''
<p>这一页只讲机制不讲分数；成绩和证据按下面的入口看，且都带当时的样本范围与条件：</p>
<ul>
<li><b><a href="../out/水印验证报告总目录.html">水印验证报告总目录</a></b>：按分类查看——「观察基线」是封存当时版本的表现，用于逐张比较；「项目管控」是冻结与防退化轮记录；「失败与诊断」是已知问题；「算法实验」含未通过实验的完整记录。</li>
<li><b>去水印失败案例记录</b>：历史与基线的代表性问题归纳，尚未解决的事项在这里登记；从总目录「失败与诊断」分类进入（原始页面为本机封存材料，不在仓库目录）。</li>
<li><b>独立验证集 v1 基线（2026-11-02）</b>：60 正例＋6 负例的一次性基线，查看后即锁柜；数字不代表当前算法已通过泛化验证。</li>
<li><b><a href="单图去水印流程演示_20261028.html">单图去水印流程演示</a></b>：一张建筑照片的真实运行记录，看实际输入、中间结果和输出长什么样（开发实例，不是独立验证）。</li>
</ul>
<div class="warn">口径提醒：历史封存的成绩对应当时的代码与样本，不是当前成绩；「测试通过」「回归零退化」只说明没把已有能力改坏，不等于去水印效果提升。当前总体结论以 README 为准：软件与管控已建立，自动去水印效果仍不稳定，没有独立泛化验证结论。</div>
'''),
    ('limits', '十四、已知限制汇总', '''
<ul>
<li>淡笔画、图像边缘残字、水印与正文交叠处容易漏选或残留；误选少量背景、照片补图痕迹仍会发生。</li>
<li>PDF 结构剥离成功的页不再跑图片修复，同页其他水印可能残留。</li>
<li>普通桌面处理没有干净原图作答案，任何「自动通过」都不等于去净；残留与识别质量无法可靠测量时保持「未评估」。</li>
<li>实验机制均未完成放行，默认路径不受影响；表格残影、正常彩色内容保护、补图耗时是已登记的待改进项。</li>
<li>程序的参数、阈值与通道顺序以流程图文档和代码为准，本页任何简化说法不用于验收。</li>
</ul>
'''),
    ('about', '十五、维护与版本', f'''
<ul>
<li>本页由 <code>tools/build_mechanism_overview.py</code> 生成，标题中的日期时间取自生成时刻；重新生成：<code>.venv/bin/python tools/build_mechanism_overview.py</code>。</li>
<li>修改导入、默认参数、PDF 判断、定位顺序、笔画补全、多图辅助、修复分流、队列控制、质量口径或导出行为时，须同步修订本页并重新生成（依据 <a href="去水印处理流程图.md">去水印处理流程图</a> 的修订核对）。</li>
<li>只重构而行为未变时，核对本页相关表述并记录「流程无变化」即可。</li>
<li>图示为机制结构示意，不冒充实际运行记录；实际运行的例子见单图流程演示。</li>
</ul>
'''),
]


def build_table():
    rows = ''.join(
        f'<tr><td><b>{escape(name)}</b></td><td>{escape(what)}</td><td>{escape(when)}</td><td>{BADGE[key]}</td></tr>'
        for name, what, when, key in MECHANISMS)
    return ('<div class="table-wrap"><table><thead><tr><th>机制</th><th>管什么</th>'
            '<th>什么时候触发</th><th>状态</th></tr></thead><tbody>' + rows + '</tbody></table></div>')


def render(now: datetime) -> str:
    stamp = now.strftime('%Y-%m-%d %H:%M')
    title = f'去水印机制与流程总览 {stamp}'
    nav = ' · '.join([
        '<a href="../README.md">项目首页</a>',
        '<a href="项目文档总览.html">文档总览</a>',
        '<a href="../out/水印验证报告总目录.html">报告目录</a>',
        '<a href="去水印处理流程图.md">处理流程图</a>',
    ])
    side = ''.join(f'<a href="#{sid}">{escape(label)}</a>' for sid, label in SIDEBAR)
    sections = []
    for sid, heading, body in SECTIONS:
        if '__TABLE__' in body:
            body = body.replace('__TABLE__', build_table())
        sections.append(f'<section id="{sid}"><h2>{escape(heading)}</h2>{body}</section>')
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{CSS}</style></head>
<body><aside><h2>去水印机制总览</h2>{side}</aside><main>
<nav>{nav}</nav>
<h1>去水印机制与流程总览</h1>
<p class="meta">生成时间 {stamp}（北京时间）· 面向内部阅读理解 · 只讲机制与能力边界，不写效果分数 · 与 <a href="去水印处理流程图.md">去水印处理流程图</a>（V1.44）配套，细节以流程图文档和实际代码为准</p>
{''.join(sections)}
<footer>本页自包含，离线可开。机制分工与边界以实际代码为准；历史成绩保留当时条件，不能当作当前成绩。发现本页与实际行为不符，请修订生成器后重新生成，不要手改页面。</footer>
</main><script>{JS}</script></body></html>
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT, help='输出 HTML 路径（默认 docs/去水印机制与流程总览.html）')
    args = ap.parse_args(argv)
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(now), encoding='utf-8')
    print(f'written: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
