"""周报（专业+易懂合并重写版）：一份文档同时满足"有数字有依据"和"不看代码能看懂"。

写法：每节先一句大白话定调，再给专业表格与数字；技术名词首次出现即用括号解释，
同时保留 ❓ 浮层给想看更细的人。白话分册与正式分册保持不动，本版取代它们成为对外版本。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out/周报'
STAMP = datetime.now().strftime('%Y-%m-%d %H:%M')
REPORT = OUT / f'本周工作总结_20261102-1105_{datetime.now().strftime("%H%M")}.html'

GLOSSARY = {
    '召回率': '真实水印笔画中被选中的比例。分母是全部水印像素，不是选区大小——只报选区覆盖率会掩盖漏检。',
    '精确率': '选中的像素里真正属于水印的比例。精确率低 = 把背景误圈进来了。',
    'alpha>0': '真值里"任何透明度大于零"的水印像素，含肉眼几乎看不见的边缘。用它当分母，淡边缘漏检就藏不住。',
    '负例': '无水印的对照图。在负例上产生任何选区或改动都算安全失败。',
    '逐字节一致': '两次运行输出文件的每个字节完全相同，证明改动没有造成任何影响。',
    '独立集': '建造和调参都没参与过的测试样本（看一次烧一次，防的是"背答案"）。在它上面验证过，才说明功能对没见过的东西有效。',
    '开发集': '已经反复看过、调过参的样本。在它上面的改善不能当作通用能力证据。',
    '转正': '候选功能从"实验开关后面"进入默认处理流程：要过独立验证 + 全量回归两道关。',
    'test-release': '全量放行：约 395 个输入场景全部重跑，确认默认流程没有被改坏，才允许更新基线。本轮 93 个批次。',
    '回归': '重跑已有测试，确认新改动没有把原本好的功能改坏。',
    '止血': '不追求完美，先用低风险措施阻止最坏情况（例：修复证据不足就原样退回，宁可不修也不改坏）。',
}


def g(t):
    return f'<span class="term">{t}<span class="tip">{GLOSSARY[t]}</span></span>'


def lead(text):
    return f'<p class="lead">{text}</p>'


def main():
    OUT.mkdir(exist_ok=True)
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>本周工作总结（11-02~11-05） {STAMP}</title>
<style>
body{{font:15px/1.9 'PingFang SC','Hiragino Sans GB',system-ui,sans-serif;background:#eef2f6;color:#25394e;margin:0;padding:40px 20px}}
.wrap{{max-width:980px;margin:0 auto;background:#fff;border-radius:14px;box-shadow:0 2px 18px rgba(30,60,90,.08);padding:46px 54px;box-sizing:border-box}}
h1{{font-size:23px;text-align:center;margin:6px 0 4px;color:#1b3a54;letter-spacing:.5px}}
.subtitle{{text-align:center;color:#7a8a94;font-size:13.5px;margin-bottom:24px}}
h2{{font-size:17.5px;color:#1b3a54;border-left:5px solid #2c6e9b;padding-left:12px;margin:40px 0 12px}}
.lead{{font-size:15.5px;background:#f2f7fb;border-left:4px solid #7aa8c4;border-radius:0 8px 8px 0;padding:12px 16px;margin:12px 0;line-height:1.9}}
table{{border-collapse:collapse;background:#fff;width:100%;font-size:13.8px;margin:14px 0;border-radius:8px;overflow:hidden}}
td,th{{border:1px solid #e2e8ef;padding:9px 13px;text-align:left;vertical-align:top}}
th{{background:#f2f6fa;color:#1b3a54;font-weight:600}}
tr:nth-child(even) td{{background:#fafcfe}}
tr:hover td{{background:#f0f6fb}}
.warn{{background:#fdf3f2;border:1px solid #e6c4c0;border-radius:10px;padding:13px 17px;font-size:14px;margin:12px 0;line-height:1.85}}
.good{{background:#f0faf4;border:1px solid #bfe3cc;border-radius:10px;padding:13px 17px;font-size:14px;margin:12px 0;line-height:1.85}}
.ok{{color:#1c7a4d;font-weight:600}} .bad{{color:#c0392b;font-weight:600}}
code{{background:#eef1f4;padding:1px 6px;border-radius:5px;font-size:12.8px}}
.meta{{color:#7a8a94;font-size:13px;text-align:center;margin-top:36px}}
.term{{border-bottom:1px dashed #2c6e9b;cursor:help;position:relative}}
.tip{{display:none;position:absolute;left:0;top:1.5em;z-index:9;width:330px;background:#fff;border:1px solid #b9cbda;border-radius:8px;
box-shadow:0 4px 14px rgba(30,60,90,.18);padding:10px 13px;font-size:13px;line-height:1.7;color:#33475b;cursor:default;font-weight:400}}
.term.open .tip{{display:block}}
.pri{{display:inline-block;min-width:60px;text-align:center;border-radius:6px;padding:2px 8px;font-size:13px;font-weight:600;white-space:nowrap}}
.p1{{background:#c0392b;color:#fff}} .p2{{background:#e67e22;color:#fff}} .p3{{background:#2c6e9b;color:#fff}} .px{{background:#95a5a6;color:#fff}}
</style></head><body><div class="wrap">
<h1>本周工作总结与下周优先项（2026-11-02 ~ 11-05）</h1>
<p class="subtitle">{STAMP} ｜ 依据：日报三篇、26 个代码提交、各日期产物目录 ｜ 性质：工作纪要，非效果放行报告</p>

<h2>一句话总结</h2>
{lead('这一周干了很多事，但<b>真正让软件去水印变好的只有一件</b>：图角落那种"2026-11-25 16:05"日期时间戳水印，以前基本去不掉，现在能去掉了（正在跑全量放行，93 个批次检完 21 个）。其余工作是打地基：建了一套"没做过的考卷"来测真实水平、定了成绩单格式、给两个已知缺陷止血。')}

<h2>一、这周每天发生了什么</h2>
{lead('四天的主线：<b>周一量出真实水平 → 周二定规矩 → 周三试新药被门卫拦住 → 周四新药治好时间戳</b>。下面每行都可以点开 ❓ 名词看解释。')}
<table><tr><th style="width:88px">日期</th><th>事情</th><th>数字</th></tr>
<tr><td><b>11-02</b></td><td>建成<b>独立测试集</b>（全新底图、没参与过任何调参），第一次测出软件的真实水平；抓到并止血一个真 bug（"自动补全"会把已找到的水印弄丢）；处理速度翻倍</td><td>独立集 66 题：{g('召回率')} 33.9%（十挑三）、{g('精确率')} 67.3%（网里三成是背景）、无水印对照图 6/6 零误伤；bug 复现：补全开关一开 {g('召回率')} 82.2%→53.2%；角标页耗时 4.2 秒→1.9 秒</td></tr>
<tr><td><b>11-03</b></td><td>成绩单（报告）格式定稿；按你的裁决简化产品（删掉两个技术开关，用户只管"给图出净图"）；发现前一天一个提速改动漏检了小水印，当天撤销——宁可慢不可错</td><td>回归口径收敛为单模式；恢复逐角检测修复图表小角标漏检；当天 7 个代码提交</td></tr>
<tr><td><b>11-04</b></td><td>"极淡水印"新解法实验室满分，但正式流程里被安全闸门误拦（修复根本没执行），登记待放行；做了一次全面体检，登记基础选区漏检的四种病根</td><td>实验室 4 张满分、误差 0.09–0.16 级——但正式工位被 {g('止血')}闸门拦截，未生效</td></tr>
<tr><td><b>11-05</b></td><td>前晚"大面积退步"查实为统计脚本自己忘了开开关（虚惊）；按你裁决两类几何 Logo 退出考核；<b>时间戳专项打透</b>：查明病根是"水印被正文保护误当正文护住"，开新通道解决，并在独立考卷的新题上验证成功，提交默认开启</td><td>修正口径后真实成绩：6 张变好/5 张变差/141 张不变；Logo 清单 11→10 种；时间戳：3 张开发样本残差降到 0.0x 级、47 页回归 {g('逐字节一致')}、独立考卷 61/61 零误伤、新题 {g('召回率')} 0→1.000</td></tr></table>

<h2>二、实事求是说问题</h2>
{lead('三条自我批评，都给了实例，不是套话。')}
<div class="warn"><b>① 忙而见效少。</b>26 个代码提交里，默认流程的效果改进只有时间戳这一个功能。其余是考卷、成绩单、监考规则——必要，但用户感受不到"去水印变好了"。</div>
<div class="warn"><b>② 近一半时间在返工。</b>时间戳功能试了 6 次，3 次是"我以为行"→一测不行→重来；其中<b>水印去掉却留粉色残留</b>是你看报告时发现的（当时只跑了分数没逐张看图）。周三晚"大面积退步"查了半天，结论是统计脚本口径错误，与算法无关。教训已立规：看效果必须逐张亲眼对图。</div>
<div class="warn"><b>③ 流程繁琐。</b>一个功能从做好到你能用上要过五道仪式（探针→47 页回归→正式报告→独立考卷→全量放行几小时），且独立考卷"看一次烧一次"，迭代起来像高考。</div>

<h2>三、问题根因与流程改进（下周五起执行）</h2>
{lead('病根不是人不认真，是<b>每个想法都按"上户口"的标准办婚礼</b>。改成两条车道：平时在试验田快进快出，攒一批挑一天统一上户口。')}
<div class="good"><b>实验车道（每天跑）</b>：新想法当天在练习题 + 47 页回归上验证，行→攒着，不行→当天埋掉只留一句话。只记进展，不出正式报告。<br/>
<b>转正车道（攒批办，如每周五）</b>：一批验证过的功能统一考独立卷、统一全量放行、统一出正式报告。<br/>
<b>日报三块</b>：死因榜（哪种失败多少例、比昨天增减）＋ 在途想法清单（试炼中/已阵亡/待上户口）＋ 健康指标。允许"平静的一天"：无变化就如实写，不为交差硬改。</div>
<p style="font-size:14px">你的体验变化：平时每天有进展感（试了什么、成几个、死几个），每周五有获得感（一批功能真正进软件）。见效频率从"一个功能拖九天"变成"每周一批"，安全措施一分不少。</p>

<h2>四、下周优先做什么（按见效速度排序）</h2>
{lead('排序逻辑：<b>投入小、救回的图多、证据现成</b>的排前面；"物理上无解"的不投入，如实登记。')}
<table><tr><th style="width:78px">优先级</th><th>项目</th><th>为什么</th><th>预估</th></tr>
<tr><td><span class="pri p1">已办待放行</span></td><td>时间戳·文档底默认开启</td><td>已提交，全量放行跑完即生效</td><td>—</td></tr>
<tr><td><span class="pri p1">下周第一</span></td><td><b>照片/图表上"看得见的"时间戳也 0 分</b></td><td>独立考卷发现：连肉眼清晰可见的时间戳，默认流程也一道不对（5 题挂 4 题），病根疑似与刚治好的文档时间戳同源，而药方可能现成</td><td>半天~1 天诊断</td></tr>
<tr><td><span class="pri p2">快</span></td><td>时间戳"成功样本"误伤背景</td><td>3 张成功的图每张多圈 6 千~1 万个背景像素（{g('精确率')} 仅 0.17–0.21），收紧选区即可</td><td>1 天内</td></tr>
<tr><td><span class="pri p2">快</span></td><td>极淡水印的安全闸门放行</td><td>周四实验室满分的修复被闸门误拦，豁免是小改动，一管 4 张图，随下批转正</td><td>1 天</td></tr>
<tr><td><span class="pri p3">中</span></td><td>白色半透明大字</td><td>Logo 退出考核后独立考卷最差的一类</td><td>诊断后估</td></tr>
<tr><td><span class="pri p3">中</span></td><td>室内标牌（indoor_signage）补读</td><td>路子已探明可行，差动手</td><td>1 天</td></tr>
<tr><td><span class="pri px">不承诺</span></td><td>5 张"程序真读不出"的图 + document_032</td><td>物理上无法与背景区分，硬解必误伤原图，违背"不误伤"铁规矩</td><td>—</td></tr></table>

<h2>五、下周行动清单</h2>
<table><tr><th style="width:26px">#</th><th>行动</th></tr>
<tr><td>1</td><td>完成全量放行复核并正式启用时间戳通道（当前 21/93 批）</td></tr>
<tr><td>2</td><td>诊断"照片/图表可见时间戳 0 分"（下周第一优先）</td></tr>
<tr><td>3</td><td>攒第一批转正：闸门豁免 + 选区收紧，周五首次按批评审</td></tr>
<tr><td>4</td><td>日报切换三块模板</td></tr>
<tr><td>5</td><td>规划加厚版独立考卷 v2（补极淡场景、含真实日期的对照文档），供后续批次用</td></tr></table>

<p class="meta">依据：out/日报/工作总结_2026-11-02/03/05.md；out/日期时间戳OCR种子候选_20261105/；out/极淡区域反染色候选_20261104/ 与验收目录。带 ❓ 的名词可点击浮层查看解释，再点一次或按 Esc 收起。</p>
<script>
document.addEventListener('click',function(e){{
  var t=e.target.closest('.term');
  document.querySelectorAll('.term.open').forEach(function(x){{if(x!==t)x.classList.remove('open');}});
  if(t)t.classList.toggle('open');
}});
document.addEventListener('keydown',function(e){{if(e.key==='Escape')document.querySelectorAll('.term.open').forEach(function(x){{x.classList.remove('open');}});}});
</script>
</div></body></html>"""
    REPORT.write_text(html, encoding='utf-8')
    print(f'written: {REPORT} ({len(html) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
