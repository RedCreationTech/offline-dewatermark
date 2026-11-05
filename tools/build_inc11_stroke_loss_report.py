"""生成 inc11 笔画补全丢选区诊断报告(自包含 HTML, 图片 base64 内嵌)。

正文全平铺; 术语解释只通过词旁 ❓ 浮层提供, 不重复设词表。
用法: .venv/bin/python tools/build_inc11_stroke_loss_report.py
产出: out/诊断_2026-11-02_inc11笔画补全选区丢失/inc11笔画补全丢选区诊断报告_2026-11-02.html
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / 'out/诊断_2026-11-02_inc11笔画补全选区丢失'
OUT = DIR / 'inc11笔画补全丢选区诊断报告_2026-11-02.html'
PREVIEW = 800


def embed(path: Path, fmt: str = 'JPEG', quality: int = 85) -> str:
    im = Image.open(path)
    if max(im.size) > PREVIEW:
        im.thumbnail((PREVIEW, PREVIEW), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == 'JPEG':
        im.convert('RGB').save(buf, 'JPEG', quality=quality)
    else:
        im.save(buf, 'PNG')
    return f'data:image/{fmt.lower()};base64,' + base64.b64encode(buf.getvalue()).decode()


SHOTS = Path('/Users/a123/Pictures/去水印截图')

IMAGES = {
    'shot_off': embed(SHOTS / '关闭笔画补全.png'),
    'shot_on': embed(SHOTS / '开启笔画补全.png'),
    'crop': embed(DIR / '原图裁剪.png'),
    'structure': embed(DIR / '结构对照_OFF红_GT绿_黄绿=重叠.png', 'PNG'),
    'decomp': embed(DIR / '对照_阶段分解.png', 'PNG'),
    'anchor': embed(DIR / '对照_笔画锚定扩展.png', 'PNG'),
    'extrap': embed(DIR / '对照_几何外推.png', 'PNG'),
    'p1': embed(DIR / '原理1_色带的形成.png'),
    'p2': embed(DIR / '原理2_补全的工序.png'),
    'p3': embed(DIR / '原理3_两种背景对比.png'),
    'p3b': embed(DIR / '原理3b_匹配分数实测.png'),
    'd1a': embed(DIR / '原理1a_通道数学.png'),
    'd1b': embed(DIR / '原理1b_门槛关卡.png'),
    'd1c': embed(DIR / '原理1c_面积三叠.png'),
    'd1d': embed(DIR / '原理1d_像素传记.png'),
    'd1d2': embed(DIR / '原理1d2_全量审计.png'),
}

GLOSSARY = [
    ('水印选区', '程序认定"这里是水印"的那部分像素。修复只会重画选区内的内容，选区丢一块，水印就留一块。'),
    ('召回率', '真实水印笔画中，有多少比例被程序选中了。越高=漏掉的水印越少。'),
    ('精确率', '程序选中的像素里，有多少真的是水印。越低=误伤的正常画面越多。'),
    ('笔画补全', '"补全水印笔画"开关背后的功能：根据颜色关系和重复出现的字形，把被复杂背景"吃掉"的水印笔画找回来。'),
    ('base 掩码', '关闭笔画补全时，检测程序自己找到的水印区域。可以理解为"基础检测结果"。'),
    ('替换与并集', '替换=丢掉旧的只留新的；并集=新旧加起来。本次事故就是"该并集却替换了"。'),
    ('alpha 场', '一张"每个像素有多像水印"的数值图。本项目里大多数检测方法只输出"是/否"两种值。'),
    ('OCR', '文字识别：在图里找文字行并读出内容。本次用它定位水印文字行的位置。'),
    ('模板匹配', '拿一块已确认的水印笔画当"样板"，在全图滑动比对，找出相同水印的其他位置。'),
    ('tophat 变换', '一种图像处理手法，专门突出"比周围亮的小块"。水印笔画靠它在照片里显形。'),
    ('纹理区', '照片里细节杂乱的区域，比如菜、肉、酱汁。本次丢笔画重灾区。'),
    ('平铺水印', '同一段文字按固定规律重复铺满整图的水印。等距=间距固定；非等距=间距忽大忽小。'),
    ('回退(兜底)', '新方案验证不通过时，退回旧方案处理，并在记录里如实标注"本次未解决"。'),
    ('逐图路由', '每张图单独判断该走哪条处理路径，而不是全图统一用一种方法。'),
    ('覆盖率', '补全结果能解释基础选区的比例（重叠面积÷基础选区面积）。用来判断补全在这张图上是否失效。'),
]

GLOSS_JS = json.dumps([{'name': n, 'text': e} for n, e in GLOSSARY], ensure_ascii=False)
TERMS = {name: f'<span class="term" data-gid="{i}">{name}<span class="q">?</span></span>'
         for i, (name, _e) in enumerate(GLOSSARY)}


def t(name: str) -> str:
    return TERMS[name]


rows_paths = [
    ('现状: 补全关', '照片重复水印扩展定位', '34.92%', '82.24%', '35.65%', '选区是实心色带，误选多但结果干净'),
    ('现状: 补全开(默认)', '重复字形·笔画补全', '8.79%', '53.16%', '91.57%', '沿笔画、误选少，但近半水印丢了'),
    ('简单并集(已否决)', '—', '35.03%', '82.92%', '35.83%', '等于退回色带方案，违反选区规则'),
]

rows_decomp = [
    ('最终对比度门槛', '54 px', '0.1%', '这道门槛基本无罪'),
    ('模板字形覆盖不到', '13,226 px', '15.2%', '样板形状没传到这里'),
    ('行内匹配失败', '39,429 px', '45.3%', '纹理区相关系数过低'),
    ('OCR 根本没看到', '34,786 px', '39.9%', '整行漏检，搜索域外'),
]

rows_tries = [
    ('A 阈值分层保留', 'alpha 场是 0/1 二值，没有"强/弱"可分层', '不可行'),
    ('B 颜色方向准入', '菜上文字的颜色信号被纹理噪声淹没', '不可行(召回封顶63%)'),
    ('降匹配阈值 0.45', '纹理区相关系数就是过不去', '不够(召回64%/精确率90%)'),
    ('几何外推盲转移', '非等距平铺，外推位置大量落空', '不可行(精确率25%)'),
    ('笔画锚定弱证据', '弱笔画在纹理区真的没有任何信号', '不够(召回60%)'),
]

rows_cross = [
    ('food_soup__tiled_diagonal', '32.1%', '32.9%', '85%', '增强有效，保留'),
    ('food_dish__tiled_diagonal', '99.1%', '99.4%', '98%', '增强有效，保留'),
    ('food_dish__tiled_irregular(本案)', '82.2%', '53.2%', '59%', '增强失败，应回退'),
    ('outdoor__gradient', '0%', '0%', '—', '两路全漏，独立 bug'),
    ('cat_lowlight__line_net', '0%', '0%', '—', '两路全漏，独立 bug'),
]


def table(headers, rows):
    h = ''.join(f'<th>{c}</th>' for c in headers)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>inc11 笔画补全丢选区诊断报告 2026-11-02</title>
<style>
body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; margin: 0 auto; max-width: 980px; padding: 24px; color: #222; line-height: 1.7; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 18px; margin-top: 32px; padding-top: 12px; border-top: 2px solid #e3efe8; }}
.meta {{ color: #666; font-size: 13px; }}
.term {{ border-bottom: 1px dotted #2a7; cursor: pointer; }}
.term .q {{ display: inline-block; width: 14px; height: 14px; line-height: 14px; text-align: center;
  font-size: 10px; background: #2a7; color: #fff; border-radius: 50%; margin-left: 3px; vertical-align: super; }}
#gloss-pop {{ display: none; position: fixed; background: #fff; border: 1px solid #b8d8c6; border-radius: 8px;
  box-shadow: 0 6px 20px rgba(0,0,0,.18); padding: 10px 14px; z-index: 99; font-size: 13px; line-height: 1.6; }}
#gloss-pop.show {{ display: block; }}
#gloss-pop b {{ color: #175c3a; display: block; margin-bottom: 4px; }}
#gloss-pop p {{ margin: 0; color: #333; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
th {{ background: #f4f7f5; }}
.plain {{ background: #f0f7f3; border-left: 4px solid #2a7; padding: 8px 12px; margin: 8px 0; }}
.warn {{ background: #fdf5ec; border-left: 4px solid #e80; padding: 8px 12px; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin: 6px 0; }}
.figcap {{ font-size: 13px; color: #666; margin-top: 0; }}
code, pre {{ background: #f4f4f4; border-radius: 4px; font-size: 13px; }}
pre {{ padding: 10px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>inc11 笔画补全丢选区诊断报告</h1>
<p class="meta">版本 v1.5 ｜ 2026-11-02 ｜ 用途：记录"开启笔画补全反而丢水印选区"问题的完整诊断证据链与止血方案 ｜ 性质：算法实验诊断（最新），非放行结论<br>
阅读提示：术语旁的小问号 <span class="q" style="vertical-align:super;">?</span> 点击弹出解释，再点一次、点空白处或按 Esc 收起。</p>

<div class="plain">
<b>一句话结论：</b>"开启{t('笔画补全')}"在处理带纹理照片上的{t('平铺水印')}时，会<b>把基础检测已经找到的水印整盘替换掉、却只找回一半</b>，
导致近一半真实水印笔画丢失、处理后满图残留。修法不是硬找回丢失笔画（实验证明此路有硬上限），
而是给补全加一道"{t('逐图路由')}验证"：解释不了基础结果就{t('回退(兜底)')}，并如实标记未解决。
</div>

<h2>① 问题现象（用户截图对比）</h2>
<p class="plain">同一张照片、同一水印，唯一的区别是"补全水印笔画"开关。开关打开后（默认状态），
{t('水印选区')}从 34% 缩到 8.8%，处理后满图水印残留。</p>
<table><thead><tr><th>开关状态</th><th>定位方式</th><th>修复选区占比</th><th>处理后效果</th></tr></thead><tbody>
<tr><td>关闭</td><td>照片重复水印扩展定位</td><td>34.03%</td><td>干净</td></tr>
<tr><td>开启（默认）</td><td>重复字形·笔画补全</td><td><b>8.79%</b></td><td><b>满图水印残留</b></td></tr>
</tbody></table>
<p>左：补全关闭；右：补全开启（残留清晰可见）。</p>
<img src="{IMAGES['shot_off']}" alt="补全关闭截图">
<img src="{IMAGES['shot_on']}" alt="补全开启截图">

<h2>② 精确复现与定性（数字与截图逐一对上）</h2>
<p class="plain">用原始输入图在本地重新跑了两遍检测，结果与截图完全吻合。同时调出水印的精确真值（该样本自带），
确认丢的全是<b>真实水印笔画</b>，不是误选被正确剔除。</p>
{table(['模式', '检测方法', '选区占图', t('召回率'), t('精确率')], [r[:5] for r in rows_paths])}
<p>补充：丢失的 29 个百分点召回，经逐像素核对，包含淡笔画、抗锯齿边缘——正是项目规则要求必须覆盖的部分。
这是真实缺陷，定性为"补全开着的丢选区"。</p>
<p>下图中红色块是"补全关"的选区形态——它把整条文字行填成<b>实心色带</b>（字与字之间的空白也涂上了），
绿色是真实水印笔画（细细的字形）。可见两种方案的形态差异：色带覆盖全但误选多，笔画形态准但在这张图上找不全。</p>
<img src="{IMAGES['structure']}" alt="色带与笔画结构对比">
<p class="figcap">结构对比：红=补全关选区（实心色带） 绿=真实水印笔画 黄绿=重叠。左下角一整行文字完全落在色带之外，补全关其实也靠"猜"覆盖。</p>

<h2>③ 根因：一行"替换"写错了语义（代码证据）</h2>
<p class="plain">补全功能的入口代码里，遇到重复文字类水印时，会把{t('base 掩码')}<b>整个丢弃</b>，
只留补全自己找到的那部分。基础结果找到的 82% 覆盖里，凡是补全的严格验证"不认"的，全部丢掉。</p>
<pre>wm/recall.py 第 128-133 行（简化）：

if mask.method in ('photo_observed_repeats',      # 三种重复文字水印
                   'photo_complete_repeated_rows',
                   'repeated_oblique_text'):
    located = observed_text.repeated_photo(...)    # 补全重新找一遍笔画
    if located is not None:
        return located                             # ← 问题在这：直接返回替换，
                                                   #   应该与 mask 做并集或先验证</pre>
<p>正确语义应该是：<b>补全只能做加法，不能把基础检测已经确认的东西弄丢</b>。当前实现是"另起炉灶+整体替换"。</p>

<h2>④ 丢失的笔画死在哪一步？（阶段分解实验）</h2>
<p class="plain">把"丢失的真实笔画"按死亡位置分桶，找出真正卡脖子的环节：</p>
{table(['死在哪一步', '丢失像素', '占比', '说明'], rows_decomp)}
<p>结论：罪魁祸首不是最后一道门槛（只有 0.1%），而是<b>{t('模板匹配')}在{t('纹理区')}过不去（45%）</b>和
<b>{t('OCR')}根本没看到整行文字（40%）</b>。补全路径自身结构决定了它在这类图上必有大量丢失。</p>
<img src="{IMAGES['decomp']}" alt="阶段分解图">
<p class="figcap">阶段分解图：绿=成功找回 红=行内匹配失败 黄=OCR未观测 蓝=误选。匹配失败集中在菜、肉等纹理区。</p>

<h2>⑤ 修"补全自己"的五次尝试与数据（为什么走不通）</h2>
<p class="plain">目标形态：召回回到 82% 以上，同时精确率明显好于简单并集的 36%。五条路全部用数据验证过：</p>
{table(['尝试方向', '实验发现', '结论'], rows_tries)}
<p>关键证据：<b>水印在菜上和盘上的实际浓度完全一样</b>（真值 {t('alpha 场')} 中位数都是 0.412）——
不是水印变淡了，是检测用的图像变换（{t('tophat 变换')}等）在{t('纹理区')}失效。这个上限短期内破不了。</p>
<img src="{IMAGES['crop']}" alt="原图裁剪">
<p class="figcap">原图局部：斜向平铺的 "5328训练素材" 文字，跨越桌面（平滑）与菜肴（纹理）两种背景。</p>
<img src="{IMAGES['anchor']}" alt="笔画锚定扩展">
<p class="figcap">尝试"笔画锚定弱证据扩展"：绿=找回 红=仍丢失 蓝=误选。比原版多救回一些，但离目标很远。</p>
<img src="{IMAGES['extrap']}" alt="几何外推">
<p class="figcap">尝试"几何外推盲转移"：误选（蓝）爆炸。因为水印间距忽大忽小，按固定间距推算的位置大量落空。</p>

<h2>⑥ 原理图解（全部为真实样本的实际计算结果）</h2>
<p class="plain">三张图分别回答三个"为什么"：色带是怎么造出来的、补全的工序是什么、纹理区为什么找不到。每张图的面板都是这张真实照片在对应处理步骤的<b>真实计算产物</b>，不是示意图。</p>

<p><b>图 1 ｜ 关补全时，色带的形成（6 步）</b></p>
<img src="{IMAGES['p1']}" alt="色带的形成">
<p class="figcap">① 原图 → ② 对立通道（突出与水印同色的部分）→ ③ tophat 增强（笔画显形）→ ④ 增强图上 OCR 确认 14 行是水印（蓝框）→ ⑤ 行框内弱响应很稀疏且对不上字形 → ⑥ 7×7 膨胀连片成实心色带（红）。第 ⑤ 步的证据太少，第 ⑥ 步的色带主要是膨胀把稀疏点连成的。</p>

<p><b>图 1 细节系列（1a~1d：机制的最细还原，全部真实计算）</b></p>
<p><b>图 1a ｜ 通道数学：</b>一个真实笔画中心像素的完整运算（RGB→对立通道→tophat→判定），配穿过笔画的一维剖面曲线——tophat"原信号减局部最大"的数学含义一目了然。</p>
<img src="{IMAGES['d1a']}" alt="通道数学">
<p><b>图 1b ｜ 门槛关卡：</b>从原始 OCR 的 14 行到最终确认，每道门槛的真实通过数。通道 2 有 10 处观测但"分散度不够"被清零——每道关卡拦什么，数字说话。</p>
<img src="{IMAGES['d1b']}" alt="门槛关卡">
<p><b>图 1c ｜ 面积三叠：</b>同一窗口下三级面积对比——OCR 实心框 53.8 万像素 → 框内 tophat&gt;2 响应 23.9 万（仅占框 44%）→ 7×7 膨胀后色带 42.9 万（膨胀贡献约 2/3）。色带的大，主要是第三步胀出来的。</p>
<img src="{IMAGES['d1c']}" alt="面积三叠">
<p><b>图 1d ｜ 像素传记与全量审计：</b>三个真实像素各自的完整身世——P（字间空白，tophat=0，离响应点 1px，被膨胀带入）、Q（真实笔画，tophat=31，凭响应计入）、R（框内但离响应 6px，落选）。右图：色带 42.9 万像素逐类审计——真实笔画 36%、贴笔画误选 58%、远离误选 6%。</p>
<img src="{IMAGES['d1d']}" alt="像素传记">
<img src="{IMAGES['d1d2']}" alt="全量审计">

<p><b>图 2 ｜ 开补全时，补全的工序（4 步）</b></p>
<img src="{IMAGES['p2']}" alt="补全的工序">
<p class="figcap">① 色带作为"去哪找"的范围提示 → ② 从确认行切样板，多处实例取中位数去噪 → ③ 全图匹配热力图，白点=过了 0.62 及格线的位置，整图只有 37 处 → ④ 每个及格位置渲染验证过的笔画（亮绿）。精准，但纹理区大片空白=找不到。</p>

<p><b>图 3 ｜ 为什么纹理区找不到：痕迹一样深，分数过不去</b></p>
<img src="{IMAGES['p3']}" alt="两种背景对比">
<p class="figcap">左：亮底实例原样；中：暗底实例原样；右：水印造成的实际像素变化放大 8 倍——每一笔都真实存在、强度一致，包括菜上。水印没有变浅。</p>
<img src="{IMAGES['p3b']}" alt="匹配分数实测">
<p class="figcap">对每个真实水印字形实测"模板能拿到的最高分数"：空心红点=纹理上被丢掉的实例（171 个，集中在及格线 0.62 下方）；实心绿点=亮底及格的实例（100 个）。纹理区实例 53% 不及格，亮底 34%；丢失实例的 77% 在纹理上。</p>

<h2>⑦ 为什么不能两者结合（困境记录）</h2>
<p class="plain">老办法"圈得出范围但认不出笔画"，新办法"认得出笔画但找不到位置"——两个办法在同一片像素上<b>互补地失明</b>。
结合听起来天然合理，但三条结合路线全部走到死胡同：</p>
<table><thead><tr><th>结合路线</th><th>死在哪</th><th>性质</th></tr></thead><tbody>
<tr><td>简单并集（色带 ∪ 笔画）</td><td>色带本来就包住大部分笔画，并集≈色带原样，精确率砸回 36%</td><td>数据否决</td></tr>
<tr><td>色带圈范围 + 笔画模板圈内渲染（最理想的形态）</td><td>"范围"和"形状"都有，唯独"往哪放"的定位信号全灭：图像比对只到 64%、域内弱证据场在纹理区不响应、按间距推算因非等距精确率崩到 25%</td><td>三个零件实测断了两个</td></tr>
<tr><td>按 OCR 读出的文字内容排版渲染</td><td>OCR 认错一个字，整串模板错位，直接把正常画面涂进选区</td><td>项目规则 §2 明文禁止（绘字/推测选区）</td></tr>
</tbody></table>
<p><b>结论：</b>这不是"没想到结合"，而是结合只会把两种失败一起继承下来。唯一活路是<b>给"定位"造一个新的信号源</b>
——能穿透食物纹理的检测通道。这就是深层课题的核心命题：<b>"为纹理区水印笔画寻找新的定位证据"</b>，
属于新能力研发，不是调参或组合能解决的。</p>
<p class="warn">注意：下一节的"逐图验证回退"是<b>止血</b>不是治病——它保证默认输出不再比基础检测差，
但这类纹理照片最终走的仍是色带方案（选区形态违规、只是碰巧修得干净）。病根本节已记录，另独立立项。</p>

<h2>⑧ 交叉验证：一张图说了不算</h2>
<p class="plain">用同批其他样本检验"{t('覆盖率')}"判别是否靠谱（{t('覆盖率')} = 补全结果能解释多少基础选区）：</p>
{table(['样本', '补全关召回', '补全开召回', '覆盖率', '判定'], rows_cross)}
<p>覆盖率 85%~98% 的样本补全确实有效或无害；本案 59% 明确失败。判别器方向正确。
<b>注意：</b>这些是同批近亲样本，只能用于选方向；最终阈值要用固定回归集的全部重复文字类样本拟合，独立验证另算（项目规则 §5）。</p>

<h2>⑨ 止血方案（已实施 2026-11-02）：给补全加"逐图验证，不过关就{t('回退(兜底)')}"</h2>
<p class="plain">不硬救丢失笔画（此路有硬上限，见第⑥节困境），改为：<b>补全结果必须能"解释"基础检测已选中的内容</b>，
解释不了就说明补全在这张图上失效，退回基础结果并如实记录。它修的是"默认比关闭更差"这个事故，
<b>不解决纹理区的检测病根</b>。</p>
<pre>{t('覆盖率')} = |基础选区 ∩ (补全结果向外容差9px)| / |基础选区|

覆盖率 ≥ 阈值（暂定 0.75，用固定回归集拟定）
    → 采用补全结果（现状不变）
覆盖率 < 阈值
    → 保留基础结果，记录 enhancement=skipped_cannot_explain_base（如实标记未解决）</pre>
<ul>
<li>本案（59%）：回退到"补全关"结果，满图残留立即消失，用户截图里那个干净输出就是回退后的样子</li>
<li>tiled_diagonal 两例（85%/98%）：补全保留，不受影响</li>
<li>符合项目规则：逐图分流、不全局开关、验证失败如实标记（§4）</li>
</ul>
<p><b>实施与验证结果（2026-11-02）：</b></p>
<ul>
<li>代码：wm/recall.py 新增覆盖率验证（_explains_base，容差 9px，门槛 0.75 暂定）；回退时保留 base 并在 MaskLayer.skip_reason 记录 cannot_explain_base；wm/contract.py 新增该字段</li>
<li>单元测试 6/6 通过（tests/test_recall_explain_base.py：覆盖率判别三例 + 替换/回退/无关方法路径）</li>
<li>本案真实图像复验：回退触发，召回 82.24%（修复前 53.16%），skip_reason 标记正确；两个 tiled_diagonal 样本增强保留不受影响</li>
<li>全量工程测试 293/298 通过；5 项失败均为 test_kimi_image_tools 缺 KIMI_IMAGEGEN_CLI 脚本（既有环境问题，与本次无关）；check-data 通过</li>
<li>固定回归集图像回归与放行：计划今晚运行，结果出来后更新本报告</li>
<li>流程图已同步：docs/去水印处理流程图.md V1.26（S5 新增验证节点 + 实现边界记录）</li>
</ul>

<h2>⑩ 遗留问题登记（本次不修，另开任务）</h2>
<ul>
<li><b>深层课题（独立立项）：为纹理区水印笔画寻找新的定位证据。</b>现状：{t('纹理区')}笔画召回上限约 64%（笔画形态）/ 82%（色带形态但有违规）；
结合的三条路线已证死（见第⑥节），需要新的检测通道或模型能力，周期不确定。</li>
<li><b>两个样本两路全漏（0% 召回）</b>：outdoor__gradient、cat_lowlight__line_net 走 nonpaper_background 路径时完全没检测到水印，是另一个独立 bug，进失败案例库。</li>
<li><b>"补全关"的色带选区本身违规</b>：实心填充文字行、误选 27.6 万像素，只是结果碰巧干净。止血方案落地后它成为纹理照片的默认输出，这个违规形态要单独治理。</li>
</ul>

<h2>⑪ 验证计划与产物清单</h2>
<p><b>实施后的验证顺序（项目规则 §7）：</b></p>
<ol>
<li>单元测试：回退触发 / 不触发两条路径 + 记录标记正确</li>
<li>相关样式回归：重复文字类全部样本 + 固定保护样本 + 全量工程测试 + check-data</li>
<li>跨样式探针：确认其他水印样式的路由不受影响（§4）</li>
<li>通过后回到主线：参数面板改造 + 一次 freeze-baseline（修好的默认路径与新口径一起冻结）</li>
</ol>
<p><b>本报告引用的实验产物</b>（目录 out/诊断_2026-11-02_inc11笔画补全选区丢失/）：</p>
<ul>
<li>实验脚本：实验_门槛变体扫描.py、实验_丢失阶段分解.py、实验_匹配阈值与对齐先验.py、实验_笔画锚定弱证据扩展.py、实验_几何外推盲转移.py</li>
<li>对照图：对照_补全开.png、对照_补全关.png、对照_阶段分解.png、结构对照_OFF红_GT绿_黄绿=重叠.png 等</li>
</ul>
<p class="meta">修订记录：v1.0（2026-11-02）初版；v1.1（2026-11-02）正文改全平铺，术语解释统一为 ❓ 浮层，删除重复的词表小节；v1.2（2026-11-02）新增"为什么不能两者结合"困境记录，止血方案章节声明不解决深层病根；v1.3（2026-11-02）止血方案已实施并记录验证结果；v1.4（2026-11-02）新增"原理图解"四张图（真实计算产物：色带形成 6 步、补全工序 4 步、两背景对比、516 个实例的匹配分数实测散点），章节顺延重编号；v1.5（2026-11-02）色带机制更正并细化：经代码核实，色带来自"OCR 行框 ∩ 低门槛响应 + 7×7 膨胀"（非行级桥接），新增图 1 细节系列 1a~1d（通道数学一维剖面、门槛关卡漏斗、面积三叠、像素传记+全量审计），全部真实计算。</p>

<div id="gloss-pop" role="dialog"></div>
<script>
const GLOSS = {GLOSS_JS};
const pop = document.getElementById('gloss-pop');
let popFor = null;

function hidePop() {{ pop.classList.remove('show'); popFor = null; }}

function showPop(el) {{
  const g = GLOSS[el.dataset.gid];
  pop.innerHTML = '<b>' + g.name + '</b><p>' + g.text + '</p>';
  pop.classList.add('show');
  popFor = el;
  const r = el.getBoundingClientRect();
  const pw = Math.min(320, window.innerWidth - 20);
  pop.style.width = pw + 'px';
  const ph = pop.offsetHeight;
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
</html>
"""

OUT.write_text(html, encoding='utf-8')
print(f'已生成: {OUT} ({OUT.stat().st_size // 1024} KB)')
