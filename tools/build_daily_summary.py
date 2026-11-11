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
    ('独立验证集', '一组从未参与过调参的测试图（本项目为 66 张），专门用来阶段性证明"改动是真的进步"而不是只对调参过的样本有效。规则要求：看过并据其调过算法后，这组样本就要降级为开发用途，不能继续宣称盲测。'),
    ('开发集', '反复用来发现问题、调参数的少量样本（本项目 14 张真实图）。"小且有偏"是性质不是缺陷——调参只需要方向，不需要统计显著。'),
    ('固定回归集', '一组固定不变的测试输入，作用是"哨兵"：每次改动后重跑，发现"修好了 A 却修坏了 B"的退化。价值在固定二字，不随版本改动。'),
    ('召回率', '真实水印笔画中，有多少比例被程序选中了。越高 = 漏掉的水印越少。'),
    ('精确率', '程序选中的像素里，有多少真的是水印。越低 = 误伤的正常画面越多。'),
    ('负例', '明确不含水印的图片。处理这类图的正确行为是零改动；负例受损说明程序在凭空制造水印。'),
    ('基线封存', '用 freeze-baseline 把当前版本在全部样本上的表现完整跑一遍并封存，之后每次测试都与封存结果对比，回答"这次改动是变好还是变坏"。本机此前一直没有这份封存。'),
    ('笔画补全', '"补全水印笔画"功能：根据颜色关系和重复出现的字形，把被复杂背景"吃掉"的水印笔画找回来。桌面上默认开启。'),
    ('基础选区', '关闭笔画补全时，检测程序自己找到的水印区域。增强路径的结果不应比基础选区差，基础选区是召回的"地板"。'),
    ('逐图验证回退', '给笔画补全加的一道"对账"：补全结果能解释基础选区的比例够高才采用；不够就退回基础结果，并如实记录"这张图补全未帮上忙"。止血性质，不解决补全在纹理照片上找不到笔画的深层病根。'),
    ('色带选区', '把整条文字行涂成实心带状的选区形态。覆盖全但把字间空白也涂上了，按项目选区规则属于违规形态，只是部分照片碰巧修得干净。'),
    ('alpha 真值', '合成样本生成时记录的"每个像素被水印盖住多少"的参考答案，0–255。只在推理结束后评分时才能读，检测算法绝不允许偷看，否则等于抄答案。'),
    ('感知哈希', '一种对图片内容算"指纹"的方法：两张图肉眼看差不多，指纹就差不多。本次用这个方法确认新抓的底图与旧底图不是换皮重复。'),
    ('处理模式', '回归只跑桌面默认一种配置：文字识别与补全水印笔画始终开启，用户无需选择。命令行入口参数与桌面一致。'),
    ('隔离候选', '新算法做成默认关闭的实验开关：不动默认处理路径，只在开关打开时生效。验收时要求"默认路径结果逐字节不变、开关打开对负例和相邻样式零影响"，全部通过并保持实验状态，不宣称通用能力提升。'),
    ('正文保护', '检测程序里的一道拦截：被判定为正文文字的 OCR 行不许进入水印搜索，防止把文档正文当水印擦掉。本次时间戳归因发现它同时也拦住了真实的时间戳行——保护规则本身没错，缺的是"时间戳行作为水印被认领"的通道。'),
]

GLOSS_JS = json.dumps([{'name': n, 'text': e} for n, e in GLOSSARY], ensure_ascii=False)
TERMS = {name: f'<span class="term" data-gid="{i}">{name}<span class="q">?</span></span>'
         for i, (name, _e) in enumerate(GLOSSARY)}


def t(name: str) -> str:
    return TERMS[name]


# ---------------------------------------------------------------- 当日人工总结
# key 为日期字符串 'YYYY-MM-DD'。只增不改；当日事实以会话记录和磁盘产物为准。
DAILY_SUMMARIES: dict[str, dict] = {
    '2026-11-05': {
        'title': '去水印项目工作日报',
        'stat_cards': [
            ('关键产出', '7 项', '模版版基线双报告 / 输入去重 / 口径修正重跑 / 两类几何Logo裁撤 / 时间戳OCR种子候选 / 极淡候选提交 / 工具链提交'),
            ('遗留阻塞', '2 项', '时间戳 8 张未解决（OCR读不出等）/ 候选未接默认路径待独立验证'),
        ],
        'one_liner': '今天是「基线报告归位 + 数据裁撤定界 + 时间戳专项从归因到候选落地」的一天：'
                     '上午把常见水印基线双报告按定稿模版重做、修复总目录分类/排序/图片路径三处问题、'
                     '清掉应用测试输入 20 组重复；随后坐实昨晚基线"大面积退化"是脚本漏开笔画补全的假性信号，'
                     '修正口径重跑（优化 6／退化 5／持平 141，整体持平）；'
                     '用户裁决裁撤透明与不透明两类几何 Logo 样本（常见清单 11→10 种，56 张输入＋202 条训练清单归档可回滚）；'
                     '下午主攻日期时间戳——修正归因（不是 OCR 读不出，而是读出的行被' + t('正文保护') + '拦下没进入水印搜索）、'
                     '做出' + t('隔离候选') + '「OCR 种子染色」，document_029/030/031 从 0% 召回到 ≈1.0、'
                     '水印区误差降 74–95%、基线逐字节与负例零影响；'
                     '用户截图发现粉色残留后当天修复（水印色逐通道实测＋透明度逐像素），'
                     '全套验收通过；8 张 OCR 真读不出的样本如实登记未解决。',
        'threads': [
            {
                'name': '① 基线报告归位与工具修复（上午）',
                'points': [
                    '总目录三处修复全部写进 build_report_index.py（重建不回退）：常见水印类报告关键词→显式分类覆盖表'
                    '（基线类进"基线"tab、报告类进"项目管控"tab，存量卡片同样生效）；每个 tab 内部按日期时间倒序、'
                    '最新在前（修排序正则：标题里"测试"二字导致时间被丢弃）；修复返回总目录报错。',
                    '双报告对齐 11-03 定稿模版：改用 build_control_reports.py 模版本体生成（适配器接入常见基线数据），'
                    '结论置顶＋对比横幅＋问题最大样式表＋6 tab＋上次基线对照图（152 张全部引用成功）；'
                    '摘要报告 100% base64 自包含，标题统一带测试当时的起止时间。',
                    '明细报告图片不显示修复：根因是报告目录与图片目录平级、相对路径缺 ../ 前缀；'
                    '修生成工具后 649 张图片全部校验存在、0 失效。',
                    '退化/优化数量不一致的口径解释：横幅数"处理记录行"、tab 数"唯一图卡片"'
                    '（同图两条记录折叠成一张卡），两层各自自洽，不动模版。',
                ],
            },
            {
                'name': '② 基线口径坐实与修正重跑（上午～中午）',
                'points': [
                    '昨晚"大面积退化"定性为<b>假性信号</b>：逐层排除（输入哈希一致、 dpi 96/200 结果相同、'
                    '排除未提交改动）后定位真凶——专项基线脚本用 Thresholds() 裸默认值 '
                    'watermark_recall_priority=False，而桌面默认基线是 True（笔画补全常开），'
                    '<b>是基线跑错了口径，不是算法回退</b>。顺带确认极淡通道 5 个文件全有开关门、默认路径零影响。',
                    '口径修正重跑（10:39–11:28，158 输入＝135 正例＋23 ' + t('负例') + '）：配对 152 页次——'
                    '<b>优化 6、退化 5、持平 141</b>、未配对 6，整体大体持平。',
                    '修正后成绩亮点：居中大字 79.2%、斜向平铺 78.5%、网址 60.0%（32→60）；'
                    '仍差：不透明 Logo 1.3%、透明 Logo 15.0%、时间戳 21.4%。',
                    '病例 A（inc11_cat_lowlight__local_text 100%→0%）为输入真的变了'
                    '（11-04 可见度重生成，绿字→近白字），同输入连跑两次逐位一致，路由稳定无随机性。',
                ],
            },
            {
                'name': '③ 样本裁撤：两类几何 Logo 退出考核（用户裁决）',
                'points': [
                    '不透明几何 Logo 移除：回归输入 14 张、训练清单 101 条、生成器与报告工具样式清单同步移除；'
                    '归档 data/_archive/不透明几何Logo移除_20261105/ 带 retirement.json（含裁决日期与理由）。',
                    '透明几何 Logo 同规格移除（14 输入＋101 训练条），全库现 347 输入／0 张 Logo；'
                    'PDF 结构测试夹具里的 logo 图（不是 Logo 样本）保留不动。',
                    '常见清单 <b>11 种 → 10 种</b>；历史封存报告中的 Logo 页保留为历史数据，不算当前成绩；'
                    'check-data 与 295 个单元测试全过。',
                ],
            },
            {
                'name': '④ 日期时间戳专项：归因修正 → 隔离候选落地（下午主线）',
                'points': [
                    '<b>归因修正（重要）</b>：原结论"11 张全死在 OCR 没读出"被探针修正——document_029–032、'
                    'charts_015 的 OCR <b>读出了</b>时间戳且命中真值，0% 召回的真机制是读出的行被 '
                    'body_text_boxes 当正文护住、没进入水印搜索；另有 5 张增强二次 OCR 也读不出'
                    '（food_dish／outdoor_building／textures_015/016／charts_016），'
                    'indoor_signage 增强可补读（登记候选）。',
                    '<b>候选 wm/ocr_seed_stain.py</b>（开关 experimental_ocr_line_seed，默认关）：'
                    'OCR 行只作搜索范围，掩码来自框邻域实测染色证据，含深色墨迹的框（真实正文）拒绝；'
                    '只承诺日期/时间样式行（正则定义，不靠固定文案），其他浅灰小字一律不动——登记为候选边界。',
                    '尝试过程全程留痕（进展记录＋尝试 1–7 写进报告）：字号筛选不可靠 → 染色带上限 14 级过严'
                    '（实测偏离 38–126 级）→ 逐级放宽＋去 paperish 条件＋修正中性灰假设（这批是浅彩色，离散 12–38 级）'
                    '→ 种子碎片合并过松收紧。',
                    '<b>尝试 6（用户截图驱动）</b>：用户发现"修复后只是变颜色、水印仍清晰"——根因是逆变换用'
                    '名义灰色 W=75，减得掉亮度差减不掉色相。修复：水印色从框内深核逐通道实测、透明度逐像素'
                    '按偏离量缩放；复测逐通道残差 → 0.00–0.02 级，输出图逐张目视确认无残留。',
                    '<b>最终验收（17:34 全部通过，预登记门槛逐项核对）</b>：document_029 召回 1.000、'
                    '030 为 0.975、031 召回 1.000，水印区误差降 74–95%、区域外改动均为 0；'
                    + t('基线封存') + '逐字节对照 47/47、候选开启零影响 42/42（' + t('负例') + ' 26＋相邻样式 16，'
                    '曾误触发的 core_036 已消除）；单测 6/6＋全套 295 项＋check-data 全过；'
                    '8 张未解决样本局部图按用户要求内嵌进报告（17:38 版）。',
                    '<b>登记未解决（如实）</b>：document_032（偏离 126 级与真实墨迹物理不可分，深色墨迹门是'
                    '它的保护而非误杀）、charts_015（灰底图表）、food_dish／outdoor_building／textures_015/016／'
                    'charts_016（OCR 读不出）、indoor_signage（增强再读候选，未实施）；'
                    '029–031 同位置跨页先验另开候选。',
                    '<b>边界声明</b>：候选不接默认路径、保持实验状态；14 张属' + t('开发集') + '，'
                    '不宣称通用能力提升，' + t('独立验证集') + '另行安排。'
                    '<b>教训落档</b>：效果验收必须逐张目视实际输出图，不能只看数值。',
                ],
            },
            {
                'name': '⑤ 入库提交（3 个）',
                'points': [
                    '6bcc24b：' + t('固定回归集') + '回归与独立验证对齐桌面默认口径（笔画补全常开）。',
                    '15f1345：极淡区域一致染色反变换' + t('隔离候选') + '（默认关闭）＋分层配置路径修正。',
                    '24340a3：报告与管控工具链（基线对比三件套、总目录旧版标注、日报术语浮层、图解与样本构建工具）。',
                    '未提交（实验进行中）：ocr_seed_stain.py 及配套验收/报告工具，待候选评审后决定。',
                ],
            },
        ],
        'metrics': [
            ('口径修正基线', '优化6/退化5/持平141', '158 输入；居中大字 79.2%、斜向平铺 78.5%、网址 60.0%（32→60）'),
            ('时间戳候选', '召回 1.000/0.975/1.000', 'document_029/030/031 从 0% 修复；误差降 74–95%；区域外改动 0'),
            ('候选隔离性', '47/47 + 42/42', '基线逐字节不变；负例＋相邻样式零影响'),
            ('样本裁撤', '28 输入＋202 训练条', '两类几何 Logo 归档可回滚；常见清单 11→10 种'),
            ('输入去重', '移除 20 组', '364 输入中 28 组重复；real_samples 8 组因配对引用保留'),
            ('回归范围', '347 输入', '375→361→347；check-data＋295 单元测试全过'),
            ('未解决登记', '8 张', '时间戳：OCR 读不出或物理不可分，原因逐张写明'),
            ('基线最差（修正后）', '不透明Logo 1.3%', '该样式当日已被裁撤退出考核；透明 Logo 15.0% 成唯一图形类'),
        ],
        'todo': [
            '时间戳候选（OCR 种子染色）保持实验状态：待跨样式回归＋独立验证后决定接不接默认路径。',
            '8 张未解决时间戳：indoor_signage 走"增强二次 OCR 再读"候选；其余 5 张需新定位证据（登记待立项）。',
            '网址 60%、白色半透明大字 35.5%、局部文字/小角标修正后回落（37→29、42→28）逐张复核是否真退化。',
            '修正后基线 vs 11-04 封存基线是"非受控对照"（混有代码与环境变化），下一轮起与本机封存基线对比才是验收口径。',
            t('负例') + ' 22/23 安全：1 张负例受损待查。',
        ],
    },
    '2026-11-04': {
        'title': '去水印项目工作日报',
        'one_liner': '今天项目干了两层事。研究线上并行推进了五条线索：'
                     '一是笔画补全守卫与过选问题的根因追查到 photo_021（退化图）——新加的逐图验证闸门在个别图上'
                     '反而保住了 30.7 万像素的基础色带坏选区；tiny_corner 的退化则另有一桩已闭环的原因——'
                     '11-02 的"四角 OCR 拼接提速"（四角各识别一次合并成一次）导致图表底图小角标漏检召回归零，'
                     '处理方式是撤销提速、恢复逐角 OCR，昨晚基线中三张角标图已去水印成功；'
                     '二是极淡水印专项走通完整闭环——从发现去不了、竞品 watermarkremover.io 能去，'
                     '到确认竞品是整页重绘的生成式路线、自研选定区域染色反变换方案，首轮效果不达预期'
                     '（水印尾部数字漏选），下午和强哥同步以后决定暂时停止这种极端水印，而是处理常见水印；'
                     '三是笔画补全的逐图验证闸门、扩展选区的 9px 误差带判定原理彻底吃透并生成图解，'
                     '基础选区与扩展选区的优化已排入推进清单（由于方向转为处理常见水印，暂不处理）；'
                     '四是报告与工具改进落地——明细报告补退化/优化数量与历史对比、总目录按时间倒序、'
                     '修复返回报错、术语禁裸写英文；'
                     '五是外部技术判断核实后并入方案抉择。决策线上，开发方向正式切为'
                     '"先把 11 种常见水印处理好"，此前暂定方案全部挂起，同时拍板不补真实样本'
                     '（相机拍摄带时间水印这种）、命名维持中英分工；样本审查坐实"白纸压白字、暗底压暗字"'
                     '两类失真后，当天完成 112 张样本重生成、新增灰黑半透明大字样式、固定回归输入同步，'
                     '1111 张常见样本全部达到可见度门槛、负例零改动、295 个测试全过，'
                     '今晚会跑这个常见的水印的处理结果。明天开始处理这些常见水印产生的问题。',
        'threads': [
            {
                'name': '① 方向与决策',
                'points': [
                    '多轮讨论后正式裁决：此前暂定方案全部止步挂起（极淡染色候选、墨色一致性关卡、低频合成样式），'
                    '唯一目标改为：<b>先把常见类型水印处理好</b>。',
                    '提出"先分常见和极端"的分类框架并确认了 11 种常见清单：局部文字、日期时间戳、网址、'
                    '透明/不透明 Logo、居中大字、白色半透明大字（限深色底）、灰黑半透明大字（文档）、'
                    '斜向平铺、高清小角标、模拟签名。',
                    '对"白色半透明大字算常见吗"提出质疑——讨论后分清了两个场景：照片类 App 常见、'
                    '文档场景不常见；最终决策："样式保留，样本须调整，文档底上不出现白色半透明水印"。',
                    '明确<b>不补真实样本</b>（原话："没啥差别"），数据集维持现有合成口径——这个决定'
                    '直接划定了本轮工作的边界。',
                    '命名口径经比较后维持现状：展示层中文、文件名英文机器 ID，不做全中文改名'
                    '（避免数据登记推倒重来）。',
                ],
            },
            {
                'name': '② 审查与口径',
                'points': [
                    '应"确认常见水印集真的可用"的要求做了逐张可见度体检，坐实两类失真样本：'
                    '白纸压白字（最低 0 级）、暗底压暗字（最低 5 级），以及"灰黑半透明大字"的覆盖缺口。',
                    '围绕数据口径连续追问（"真的是常见类型吗""为什么这么多""包含关系""都是合理的吗"），'
                    '澄清了 1010 常见 / 2553 总量 / 145 固定回归的包含关系和统计单位差异。',
                    '日报统计口径两次修正：会话先改按最后一条消息归日、再改为按每条消息自身时间归日——'
                    '今天实际 8 个会话、94 轮，归日逻辑已固化进工具。',
                    '版式三处调整：去掉工作时段、去掉会话纪要、补充决策工作记录——'
                    '本报告的三层结构（决策/审查/实施）由此而来。',
                ],
            },
            {
                'name': '③ 样本体检结论（支撑决策的证据）',
                'points': [
                    '白色半透明大字 9 张固定回归样本里 6 张是白纸/浅图表压白字（0–3.7 级，物理不可见，非真实场景）；',
                    '真实照片底上 8 张常见类 &lt;10 级——生成器随机暗色从不看底图（实测水印与背景亮度差仅 3–23 级，'
                    '真实 App 在暗底用浅色水印）；',
                    '覆盖缺口确认：灰黑半透明大字（文档上真实存在的半透明形态）整套样本一种都没有。',
                    '结论：这批样本把"算法失败"和"样本本身不可见"混在一起，不修正则失败归因会出错。',
                ],
            },
            {
                'name': '④ 实施落地（当日完成，只动样本侧，算法零改动）',
                'points': [
                    '产出《项目现状与开发方向整理》（自包含 HTML、进总目录），生成逻辑固化在 '
                    'tools/build_project_status_direction.py，可重跑。',
                    '生成器加可见度兜底：中位可见度 &lt;8 级自动按底图重染颜色（暗底浅色/亮底深色，两种极性取更优）；'
                    'center_white 浅底自动路由新增的 <b>center_gray（灰黑半透明大字）</b>样式；'
                    '随机流保持一致，未触发样本逐字节不变。',
                    '存量重生成：v1 改 107 张（47 张路由 center_gray、60 张重染），增量改 5 张；'
                    '旧文件全部归档 replaced_20261104/ 带 receipt.json，可回滚。',
                    '固定回归输入同步：应用测试输入原地更新 10 张、迁移更名 6 张（新目录'
                    '"02b_灰黑半透明大字"），输入来源与校验.json 重算 16 条哈希；'
                    'coverage.json 的 validation split pin 更新并留痕。',
                ],
            },
            {
                'name': '⑤ 验收门槛逐条核对（预登记，不看结果后改）',
                'points': [
                    '✅ 常见类可见度：两数据集 1111 张常见样本全部 ≥8 级（修正前最低 0 级）；',
                    '✅ ' + t('负例') + '零改动：哈希抽查全部一致；',
                    '✅ check-data 通过（269 文件核对）、295 个单元测试全过；',
                    '⏳ 重封存' + t('基线封存') + '未执行（423 页全量约 2 小时），等待启动指令，新旧基线并存不混比。',
                ],
            },
            {
                'name': '⑥ 数据口径沉淀（备查）',
                'points': [
                    '常见水印样本量：1010 张（v1 960＝96 底图×10 种；增量 50＝5 照片底×10 种）；'
                    '全量 2553 条（常见 1010＋非常见 1212＋' + t('负例') + ' 133＋PDF 198）。',
                    '非常见类只登记未修（如约 6 张 &lt;8 级，含 indoor_signage 非等距密集文字 5.3 级）；'
                    '极淡是有意 &lt;5 级的边界样本，不算失真。',
                    'PDF 类走对象删除，不适用可见度标准；矢量路径（5% 召回）是结构类的已知缺口。',
                ],
            },
            {
                'name': '⑦ 当日研究线（8 个会话并行，按线索归并）',
                'points': [
                    '<b>线索一：笔画补全守卫与过选问题</b>（贯穿 3 个会话）——photo_021 过选根因查清：'
                    '新加的笔画补全逐图验证闸门（解释不了基础选区就不允许替换）在 photo_021 上算出远低于 75% '
                    '门槛，反而保留了 30.7 万像素的坏选区。'
                    'tiny_corner 属另一桩已闭环的退化：11-02 的"四角 OCR 拼接提速"（四角各识别一次合并为一次）'
                    '导致图表底图小角标漏检（召回归零）；恢复方式为撤销提速、恢复逐角 OCR，'
                    '以角标页 1.9s→4.2s 的耗时回退换取检出正确性（提交 dc0a65f/2c99e71），'
                    '昨晚基线中三张角标图已去水印成功。'
                    '进一步追问退化归因四类情况是否同根，确认情况一已修、其余指向同一修改。'
                    '<b>根治方案 A 定型</b>："整行涂满"通道的输出必须再过一道笔画证据检查'
                    '（模板命中率 ≥75% 且原图明暗突变 >30）才允许留在最终选区——不是继续打补丁。'
                    '期间还讨论了"纸质重复叠加"通道为什么在多选 2.6 万像素的情况下胜出、统一质检的原理、'
                    '以及"根本解法是不是训练模型"的方向之问（结论已登记待立项）。',
                    '<b>线索二：极淡水印专项</b>（完整走通一轮研究闭环后暂停）——从"极淡有没有解决方案"开始，'
                    '要求一份完整记录：发现去不了→竞品 watermarkremover.io 能去→为什么→严格要求是否过严→'
                    '优化路线的报告（已生成自包含 HTML 并进总目录）。关键辨析：竞品是整页重绘（Inpainting '
                    '生成式模型，追求视觉干净而非文档保真），自研方向是区域级一致染色反变换（数学减法、'
                    '不动正文）；"只限白纸是否太严"引出四级参照面梯子。选定自研方案后跑首轮，'
                    '效果不达预期（document_030/031 水印尾部数字漏选）。'
                    '<b>下午与强哥同步后决定：暂时停止这类极端水印，转攻常见水印</b>，本线索转入挂起。',
                    '<b>线索三：笔画补全逐图验证闸门与 9px 误差带原理吃透</b>——守卫判定公式的细节追问：'
                    '9×9 膨胀为什么每边只放宽 4px、基础选区与扩展结果谁覆盖谁（例子图是否画反）、'
                    '横向笔画是否同样适用，直至生成图解确认理解无误；结论：基础选区与扩展选区都还需要继续优化，'
                    '已排入推进清单。<b>由于方向转为处理常见水印，本项暂不处理。</b>',
                    '<b>线索四：报告与工具改进</b>——基线明细报告需求定稿：要显示退化/优化数量、'
                    '能展开历史基线的三档＋多选背景＋退化优化数据做逐日对比；总目录全部入口按时间倒序、'
                    '最新在前；修复"返回总目录报错 ERR_FILE_NOT_FOUND"；删除非真实时间的旧报告只保留 0224 版。'
                    '版式规范新增：极淡报告图片只显示水印区域（整图看不出边界）、'
                    '术语禁止裸写英文代码名（必须带解释）。',
                    '<b>线索五：外部技术判断核实</b>——引入外部意见：存在保真度更高的去水印方式，'
                    'watermarkremover.io 很可能基于 Inpainting 生成式模型；核实要点有二：'
                    '竞品是否真为整页重绘（出图很快是否符合该判断）、合成图验证竞品是否有影响。'
                    '核实结论已并入线索二的方案抉择。',
                    '<b>另一条登记在案</b>：单模式 423 页新封存替换三模式 1,266 页封存的口径问题，'
                    '登记待拍板，未并入上述五条线索。',
                ],
            },
        ],
        'metrics': [
            ('方向与关键决策', '6 项', '方向换挡 / 11 种常见清单 / 白色半透明场景辨析 / 不补真实样本 / 命名维持中英分工 / 样本调整方案批准'),
            ('失真样本修正', '112 张', 'v1 107（47 路由＋60 重染）＋增量 5；全部归档可回滚'),
            ('常见样本可见度', '1111/1111 ≥8 级', '修正前最低 0 级（白纸压白字）'),
            ('新增样式', 'center_gray', '灰黑半透明大字，v1 47 张；文档真实形态补空白'),
            ('回归输入同步', '16 张', '原地 10＋迁移更名 6（02b_灰黑半透明大字）'),
            ('check-data', '通过', '269 文件核对；coverage pin 已留痕更新'),
            ('单元测试', '295/295 通过', '应用算法零改动'),
            ('常见类当前最差（旧基线口径）', '时间戳 20% 召回', '主攻靶子；新基线待重封存'),
        ],
        'todo': [
            '<b>今晚</b>：跑常见水印的处理结果（423 页基线），出分后对比旧基线。',
            '<b>明天起</b>：处理常见水印暴露的问题（主攻时间戳、网址、Logo、白色半透明大字的漏检归因）。',
            'center_gray 建议下轮补进最小回归集（coverage 仍记 22/22）。',
            '非常见类排一轮同样的可见度体检（已登记未修样本约 6 张起）。',
            '挂起项保持挂起：极淡染色候选、墨色一致性关卡、闸门与选区优化，方向回转时再评审。',
            '登记待拍板：单模式封存指针替换三模式封存的口径问题。',
        ],
    },
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
    """读取当天全部会话的统计与对话摘要。

    归属口径（2026-11-04 二次修正）：**消息级归日**——逐条按消息自身时间戳的
    本地区间日期过滤，只把当天发出的提问与回答计入当天；同一会话跨多天
    时，每天各计各的消息。会话被计入的条件：当天至少有一条消息。
    """
    from zoneinfo import ZoneInfo
    local = ZoneInfo('Asia/Shanghai')

    def _day(ts: str):
        try:
            return datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone(local).date().isoformat()
        except (ValueError, AttributeError):
            return None

    cwd_dir = SESSIONS_ROOT / ('--' + PROJECT_CWD.strip('/').replace('/', '-') + '--')
    stats = []
    for path in sorted(cwd_dir.glob('*.jsonl')):
        users, assistants = [], []
        first_ts = last_ts = None
        with path.open(encoding='utf-8') as fp:
            for line in fp:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = obj.get('timestamp')
                if not ts or _day(ts) != day:
                    continue
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
                first_ts = first_ts or ts
                last_ts = ts
                if msg.get('role') == 'user':
                    if content.startswith('<') or 'tool_result' in content[:30]:
                        continue
                    users.append(content)
                elif msg.get('role') == 'assistant':
                    assistants.append(content)
        if not last_ts:
            continue
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

    default_cards = [
        ('关键产出', '6 项', '方向决策 / 样本审查口径 / 体检结论 / 生成器修正 / 112 张重生成 / 今日日报'),
        ('遗留阻塞', '1 项', '423 页新基线重封存待启动（等发令）'),
    ]
    stat_cards = [('会话数', f'{len(sessions)}', '今日 pi 会话文件数'),
                  ('对话轮次', f'{total_turns}', '用户提问/指令总数')]
    for k, v, s in summary.get('stat_cards', default_cards):
        stat_cards.append((k, v.format(sessions=len(sessions), turns=total_turns), s))
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

<h2>四、待办与阻塞</h2>
<ul>{todo}</ul>

<p class="meta" style="margin-top:36px">修订记录：v1.1（{day}）按消息时间归日、不再展示逐会话明细；v1.0（{day}）初版，由 tools/build_daily_summary.py 生成。<br>
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
