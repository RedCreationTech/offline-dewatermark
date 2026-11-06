"""生成本周工作总结 HTML（2026-11-02 ~ 11-05，含流程改进与优先项建议）。

内容依据：out/日报/工作总结_2026-11-02/03/05.md、当天会话、git 提交（26 个）、
各日期产物目录。性质：工作纪要与管理建议，非效果放行报告。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out/周报'
STAMP = datetime.now().strftime('%Y-%m-%d %H:%M')
REPORT = OUT / f'本周工作总结与优先项建议_20261102-1105_{datetime.now().strftime("%H%M")}.html'

GLOSSARY = {
    '召回率': '真实水印笔画中被选区命中的比例。只报告选区覆盖率不算数。',
    '精确率': '选区像素里真正属于水印笔画的比例。精确率低 = 把背景也圈进来了。',
    'alpha>0': '真值标注里"任何透明度大于零"的水印像素，含肉眼几乎看不见的边缘。分母用它，淡边缘漏检就藏不住。',
    '负例': '无水印的对照图。在负例上产生任何选区或改动都算安全失败。',
    '逐字节一致': '两次运行的输出文件每个字节都一样，证明改动没有带来任何影响。',
    '开发集': '已经反复看过、调过参的样本。在它上面的改善不能当作通用能力证据。',
    '独立集': '没参与过调参的样本，用来检验"没见过的东西行不行"。看一次就烧一次，不能反复调。',
    '止血': '不追求完美修复，先用低风险措施阻止最坏情况（比如"宁可不修也不许改坏"的退回闸门）。',
    '转正': '候选从"实验开关后面"进入默认处理流程的审批流程：独立验证 + 全量回归放行。',
    'test-release': '全量放行：约 395 个场景全部重跑一遍，确认默认路径没有回归，才允许更新基线。',
    '死因聚类': '不按"哪张图失败"修，而按"哪种死法"分桶计数（如 OCR 读不出×5、被正文保护×4），按桶的频次和严重度排优先级。',
}


def g(t):
    return f'<span class="term">{t}<span class="tip">{GLOSSARY[t]}</span></span>'


def main():
    OUT.mkdir(exist_ok=True)
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>本周工作总结与优先项建议（11-02~11-05） {STAMP}</title>
<style>
body{{font:15px/1.85 system-ui,'PingFang SC',sans-serif;background:#f1f4f8;color:#25394e;margin:0;padding:26px;max-width:1150px}}
h1{{font-size:22px}} h2{{font-size:17px;border-left:4px solid #2c6e9b;padding-left:10px;margin-top:30px}}
table{{border-collapse:collapse;background:#fff;width:100%;font-size:14px;margin:12px 0}}
td,th{{border:1px solid #d5dfe8;padding:7px 11px;text-align:left;vertical-align:top}} th{{background:#eef3f8}}
.note{{background:#fff8e6;border:1px solid #e8d9a0;border-radius:8px;padding:11px 16px;font-size:14px}}
.warn{{background:#fdecea;border:1px solid #e6b0aa;border-radius:8px;padding:11px 16px;font-size:14px}}
.good{{background:#eafaf1;border:1px solid #b7e4c7;border-radius:8px;padding:11px 16px;font-size:14px}}
.ok{{color:#1c7a4d;font-weight:600}} .bad{{color:#c0392b;font-weight:600}} code{{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:13px}}
.meta{{color:#7a8a94;font-size:13px}}
.term{{border-bottom:1px dashed #2c6e9b;cursor:help;position:relative}}
.tip{{display:none;position:absolute;left:0;top:1.5em;z-index:9;width:340px;background:#fff;border:1px solid #b9cbda;border-radius:8px;
box-shadow:0 4px 14px rgba(30,60,90,.18);padding:10px 13px;font-size:13px;line-height:1.7;color:#33475b;cursor:default;font-weight:400}}
.term.open .tip{{display:block}}
.pri{{display:inline-block;min-width:64px;text-align:center;border-radius:6px;padding:2px 8px;font-size:13px;font-weight:600}}
.p1{{background:#c0392b;color:#fff}} .p2{{background:#e67e22;color:#fff}} .p3{{background:#2c6e9b;color:#fff}} .px{{background:#95a5a6;color:#fff}}
</style></head><body>
<h1>本周工作总结与优先项建议（2026-11-02 ~ 11-05）</h1>
<p class="meta">{STAMP} ｜ 依据：日报三篇（11-02/03/05）、26 个 git 提交、各日期产物目录 ｜
性质：工作纪要与管理建议，<b>不是效果放行报告</b>。时间戳候选的全量放行截至本报告时仍在跑（21/93 批）。</p>

<h2>一句话结论</h2>
<p class="note">本周是「测量地基 + 管控定型 + 一个候选打透」的一周：独立集建成并基线化、回归与报告体系定型、
inc11 缺陷止血、日期时间戳候选从归因修正到默认开启提交。成果是实的，但<b>默认路径上的效果改进集中于一个候选</b>，
流程繁琐与返工成本已经肉眼可见——建议从下周起切换到「两车道 + 按批转正」的工作方式（见第四节）。</p>

<h2>一、本周到底干了什么（按天）</h2>
<table><tr><th style="width:90px">日期</th><th>主要内容</th><th>关键数字 / 产物</th></tr>
<tr><td><b>11-02</b><br/>周一</td><td>独立验证集建成并基线化；inc11 补全丢笔画缺陷实锤并止血；检测提速；全量基线封存</td><td>独立集 66 条/26 新底图/三重去重；首轮基线 {g('召回率')} 33.9%、{g('精确率')} 67.3%、负例 6/6 零改动；inc11 复现 82.2%→53.2%；角标页耗时 4.2s→1.9s；基线封存 300 场景</td></tr>
<tr><td><b>11-03</b><br/>周二</td><td>基线报告模版定型；色带原理图解；产品方案落地（删开关、单模式口径）；一处提速回退（正确性优先）</td><td>删开关后 OCR/补全永远开启；回归收敛桌面默认单模式；恢复 corner_id 逐角 OCR（修图表小角标漏检）；当天 7 提交</td></tr>
<tr><td><b>11-04</b><br/>周三</td><td>极淡区域反染色候选两轮（测量式反变换，不用 OCR）；常见水印基线；基础选区漏检诊断（四类根因登记）</td><td>极淡候选二轮探针：document_029–032 召回 1.000、误差 0.09–0.16 级——但随后全量验证发现真实管线被 {g('止血')}闸门拦住、修复根本没执行（登记待处理，见优先项 4）</td></tr>
<tr><td><b>11-05</b><br/>周四</td><td>基线口径坐实（昨晚"大面积退化"是脚本漏开笔画补全的<b>假性信号</b>，修正后优化 6/退化 5/持平 141）；两类几何 Logo 样本裁撤（11→10 种）；<b>日期时间戳专项</b>：归因修正 → 候选落地 → 独立集转正 → 默认开启提交</td><td>候选尝试 7 次：document_029/030/031 残差 → 0.0x 级；基线 47/47 {g('逐字节一致')}、零影响 42/42；独立集 v1：61/61 零影响、xdoc_002 独立新底图 {g('召回率')} 0→1.000；8 张未解决样本如实登记</td></tr></table>
<p class="meta">全周 26 个提交。应用实际行为变化三处：能力开关删除（11-03）、Logo 两类退出考核（11-05）、
时间戳通道默认开启（11-05 提交，全量放行 <code>test-release</code> 21/93 批进行中，完成后才生效）。</p>

<h2>二、实事求是：本周的问题</h2>
<div class="warn"><b>1. 净效果集中，产出大头是工具与管控。</b>26 个提交里，默认路径的效果改进实质只有一个候选
（时间戳文档底：开发集 3 张 + 独立集 1 张）。其余是报告体系、验收运行器、巡检工具——必要，但用户感知不到"去水印变好了"。</div>
<div class="warn"><b>2. 返工吃掉了近一半时间。</b>时间戳候选尝试 6 次，其中 3 次是"想当然→被打脸"：
字号筛选假设错误、证据带上限拍脑袋、中性灰假设错误；<b>粉色残留是用户看报告时抓出来的</b>——验收只看数值没逐张看输出图。
inc11 修复探索 5 条路线全部被数据证伪后才收敛为止血方案。11-04 晚"大面积退化"花了半天排查，结论是基线脚本口径 bug，与算法无关。</div>
<div class="warn"><b>3. 流程繁琐已影响节奏。</b>一个候选走完探针→47 页验收→HTML 报告→独立集→test-release（93 批、数小时），
才能让用户真正用上；独立集"看一次烧一次"的语义让每次验证都像考试，不敢迭代。</div>

<h2>三、这些问题的根因（不是人的问题，是流程粒度问题）</h2>
<table><tr><th>现象</th><th>根因</th></tr>
<tr><td>每天很忙、见效少</td><td>每个候选都按"默认路径转正"的标准办婚礼；实验没有快车道，大多数候选本可以在一天内被证伪或晋级</td></tr>
<tr><td>不稳定、返工</td><td>修"哪张图死了"而不是修"哪种死法"；缺少"逐张看实际输出"的硬约束执行</td></tr>
<tr><td>独立集浪费</td><td>单候选单烧独立集；应该攒一批候选一次验</td></tr>
<tr><td>测试报告信息量低</td><td>只报"哪张没过"，不报"死法分布及趋势"</td></tr></table>

<h2>四、建议的新流程（两车道 + 按批转正 + 日报三板块）</h2>
<div class="good"><b>实验车道（快，每天跑）</b>：探针 → 修 → 开发集 + 47 页回归，一天内闭环。
只留进展记录 md，不出 HTML、不进总目录、不碰独立集。大多数候选死在这里，死得快就是赚。<br/>
<b>转正车道（慢，按批做）</b>：攒一批晋级的候选，一次独立集验收、一次 test-release、一次正式报告（如每周五评审）。
样式提示触发的接入当天可用，不算默认路径变更。</div>
<p><b>日报固定三板块</b>（保证没修东西也有的报）：① {g('死因聚类')}表（各死法桶的数量与昨日对比）；
② 在途候选状态机（探针中/调参中/待回归/已毙/已晋级）；③ 稳定性指标（回归通过率、零影响页数、耗时）。
允许"平静的一天"：无变化就如实写无变化，不硬凑动作。</p>

<h2>五、优先优化清单（按见效速度排序，都有本周实测证据）</h2>
<table><tr><th style="width:76px">优先级</th><th>项目</th><th>证据与预期</th><th>工作量</th></tr>
<tr><td><span class="pri p1">已办待放行</span></td><td>时间戳·文档底（默认开启）</td><td>已提交，test-release 跑完即生效</td><td>—</td></tr>
<tr><td><span class="pri p1">下周第一</span></td><td>时间戳·照片/图表底<b>可见档</b></td><td>独立集证明：默认路径对 alpha_max=178 的<b>显眼</b>时间戳也 0%（5 张里 4 张），死法疑似与文档底同源（OCR 读出后被正文保护拦下）；照片路径现有 local_lines 通道可诊断。收益：独立集时间戳 1/5 → 潜在 4/5</td><td>半天~1 天诊断</td></tr>
<tr><td><span class="pri p2">快</span></td><td>时间戳成功样本的多选问题</td><td>3 张成功样本 {g('精确率')} 仅 0.17–0.21（背景多选 6–10k 像素），收紧颜色通道选区即可，指标直接变好看</td><td>1 天内</td></tr>
<tr><td><span class="pri p2">快</span></td><td>极淡通道闸门豁免</td><td>已验证的修复（连带 ≤0.23 级）被 repaint_too_broad 闸门拦截、修复未执行；豁免改动小，解 document_029–032 __faint 4 张，随下批转正</td><td>1 天</td></tr>
<tr><td><span class="pri p3">中</span></td><td>白色半透明大字</td><td>独立集 {g('召回率')} 最差三类之首（Logo 裁撤后它是最大短板）</td><td>需诊断后估</td></tr>
<tr><td><span class="pri p3">中</span></td><td>indoor_signage 增强再读候选</td><td>探针已证可行（增强 OCR 补读、命中真值 3–5px），未实施</td><td>1 天</td></tr>
<tr><td><span class="pri px">不承诺</span></td><td>OCR 读不出的 5 张（照片/纹理底极淡）</td><td>增强再读也 0 行，物理上难，保持登记，不投入</td><td>—</td></tr>
<tr><td><span class="pri px">另立项</span></td><td>document_032（时间戳深到与墨迹不可分）</td><td>需跨页先验候选（029–031 同位置配准），单独评估</td><td>—</td></tr></table>

<h2>六、下周建议行动</h2>
<table><tr><th style="width:24px">#</th><th>行动</th></tr>
<tr><td>1</td><td>完成 test-release 复核并 promote 时间戳通道（放行已 21/93）</td></tr>
<tr><td>2</td><td>启动"时间戳·照片/图表底可见档"诊断（下周第一优先）</td></tr>
<tr><td>3</td><td>攒第一批转正候选：极淡闸门豁免 + 多选收紧，周五首次按批评审</td></tr>
<tr><td>4</td><td>日报切换三板块模板，首次执行"平静的一天"规则</td></tr>
<tr><td>5</td><td>规划独立集 v2（加厚 + 补亚阈值时间戳 + 含真实日期的文档负例），供下批转正用</td></tr></table>

<p class="meta">依据文件：out/日报/工作总结_2026-11-02.md、_2026-11-03.md、_2026-11-05.md；
out/日期时间戳OCR种子候选_20261105/（进展记录、验收证据、报告 17:38 版）；
out/极淡区域反染色验收_20261105/；out/极淡区域反染色候选_20261104/。</p>
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
