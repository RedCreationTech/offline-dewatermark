"""周报的白话详解版（个人临时查看，不入总目录）。

把「本周工作总结与优先项建议」逐节翻译成大白话：每个数字是什么意思、
问题为什么重要、建议改了之后你每天的体验会变成什么样。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out/周报'
STAMP = datetime.now().strftime('%Y-%m-%d %H:%M')
REPORT = OUT / f'周报白话详解_20261102-1105_{datetime.now().strftime("%H%M")}.html'

def build_html():
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>周报白话详解（11-02~11-05） {STAMP}</title>
<style>
body{{font:16px/2 'PingFang SC','Hiragino Sans GB',system-ui,sans-serif;background:#eef2f6;color:#25394e;margin:0;padding:40px 20px}}
.wrap{{max-width:880px;margin:0 auto;background:#fff;border-radius:14px;box-shadow:0 2px 18px rgba(30,60,90,.08);padding:44px 52px;box-sizing:border-box}}
h1{{font-size:24px;text-align:center;margin:6px 0 4px;color:#1b3a54}}
.subtitle{{text-align:center;color:#7a8a94;font-size:13.5px;margin-bottom:26px}}
h2{{font-size:18px;color:#1b3a54;border-left:5px solid #2c6e9b;padding-left:12px;margin:40px 0 14px}}
h3{{font-size:15.5px;color:#2c6e9b;margin:22px 0 6px}}
.plain{{background:#f7fafc;border:1px solid #e2e8ef;border-radius:10px;padding:14px 18px;margin:12px 0;font-size:15px}}
.ana{{background:#fff8e6;border:1px solid #e8d9a0;border-radius:10px;padding:14px 18px;margin:12px 0;font-size:15px}}
table{{border-collapse:collapse;width:100%;font-size:14.5px;margin:14px 0}}
td,th{{border:1px solid #e2e8ef;padding:10px 13px;text-align:left;vertical-align:top}}
th{{background:#f2f6fa;color:#1b3a54}}
em{{color:#c0392b;font-style:normal;font-weight:600}}
strong{{color:#1b3a54}}
</style></head><body><div class="wrap">
<h1>周报白话详解（11-02 ~ 11-05）</h1>
<p class="subtitle">{STAMP} ｜ 这是对隔壁那份《本周工作总结与优先项建议》的逐句翻译，写给不看代码的人<br/>看到"召回率""逐字节"这类词不用怕，每个都会用大白话解释</p>

<h2>〇、先记住三个比喻</h2>
<div class="ana">
<strong>比喻一：捞鱼。</strong>图片是一片池塘，水印是鱼，程序是一张网。<br/>
<strong>捞回率（召回率）</strong>= 塘里 100 条鱼捞回来几条。捞回率 33.9% 就是"十挑三，七成水印还在图上"。<br/>
<strong>误捞率（精确率）</strong>= 网里捞上来的东西有几成真是鱼。误捞率 67.3% 就是"网里三分之一是水草石头"——水草石头就是被你误伤的原图背景。<br/>
<strong>这张网的两条铁规矩</strong>：宁可漏捞（水印去不干净，图还在），不可误捞（把原图内容擦了，救不回来）。
</div>
<div class="ana">
<strong>比喻二：考试。</strong><br/>
<strong>开发集</strong>= 做过的练习题（答案都背熟了，考高分不算本事）。<br/>
<strong>独立集</strong>= 没见过的考卷（考一次烧一次，不能拿来反复练）。一份没见过的考卷上考好了，才说明真学会了。
</div>
<div class="ana">
<strong>比喻三：看病。</strong><br/>
<strong>止血</strong>= 病人大出血时先扎住，不追求根治。<br/>
<strong>死因聚类</strong>= 不按"哪个病人死了"逐个治，而是统计"哪种死法最多"（心梗 5 个、中风 3 个），先把最大的死因干掉。
</div>

<h2>一、这份周报在说什么（一句话版）</h2>
<div class="plain">这一周干了很多事，但<strong>真正让软件去水印变好的只有一件</strong>：某种"日期时间戳"水印（图角落的 2026-11-25 16:05 那种），以前完全去不掉，现在能去掉了。其他工作是在<strong>修度量衡、立规矩、擦屁股</strong>——这些不直接变好看，但没有它们，变好看了也不知道、也不敢说。</div>

<h2>二、每天到底干了啥（大白话）</h2>
<table><tr><th style="width:100px">日子</th><th>用三句话说清楚</th></tr>
<tr><td><strong>11-02 周一</strong></td><td>① 造了一摞"没见过的考卷"（独立集，66 道题），第一次考试：十挑三（捞回率 33.9%），而且网里三成是水草（误捞率 67.3%）。这就是现在的真实水平。<br/>② 抓到一个真 bug：有些图开"自动补全"反而把已经找到的水印弄丢了大半（捞回 82%→53%），当天做了"止血"：补全不合格就退回原样。<br/>③ 处理速度翻了一倍（4.2 秒→1.9 秒），没影响效果。</td></tr>
<tr><td><strong>11-03 周二</strong></td><td>① 把"成绩单"的格式定下来了（以后每张卷子长什么样）。<br/>② 按你的裁决改了产品：以前界面上有两个技术开关，现在删了，用户只管"给一张图、出一张净图"。<br/>③ 发现前一天的一个提速改动漏掉了一种小水印（图表角落的小字），当天撤销——宁可慢，不可错。</td></tr>
<tr><td><strong>11-04 周三</strong></td><td>① 试了一种"极淡水印"的除法（水印淡到 OCR 都读不出），实验室里四张全满分，但随后发现<strong>正式工位上有个门卫把修理工拦在了门外</strong>——修复压根没执行。这条先记着，下周放行卫。<br/>② 做了一次全面体检，登记了"基础选区漏检"的四种病根。</td></tr>
<tr><td><strong>11-05 周四</strong></td><td>① 前一天成绩单上"大面积退步"虚惊一场：查了半天，是<strong>抄分数的人抄错了栏</strong>（脚本忘了开一个开关），不是真退步。修正后：6 张变好、5 张变差、141 张不变。<br/>② 按你的裁决，两类"几何图形 Logo"水印退出考核（清单从 11 种减到 10 种）。<br/>③ <strong>本周主菜</strong>：那种"日期时间戳"水印，以前 14 张里 3 张能去，现在研究透了——不是程序瞎，是水印被"正文保护"功能误当成正文护住了。给程序开了条新通道，3 张从 0 分到满分，还在没见过的考卷上验证了一道新题也满分。最后提交了"默认开启"，正在跑全厂体检（93 个车间检完 21 个）。</td></tr></table>

<h2>三、周报里"自我批评"那三条，翻译成人话</h2>
<div class="plain"><h3>批评一：忙了一周，用户能感知到的变好只有一件</h3>
26 次代码存档里，大部分是"成绩单、体检表、监考规则"。这些必须有人做，但你作为用户感受不到。所以后面要改工作方式（见第四节），让"能感知的变好"变多。</div>
<div class="plain"><h3>批评二：近一半时间在返工</h3>
举两个你亲历的例子：<br/>
· 时间戳那个新功能，试了 6 次才成——中间有 3 次是"我以为这样可以"，一测不行，重来。其中最丢人一次：<strong>水印是去掉了，但留下一圈粉色</strong>——这个是你看报告时发现的，光跑数字没看出来。后来查明是"调色配方想当然"，改成实测每种颜色的配方后，残留归零。<br/>
· 周三晚上成绩单说"好多图退步了"，吓得半死，查了半天发现是<strong>统计脚本自己忘了开开关</strong>，白紧张。<br/>
教训立了规矩：以后看效果必须<strong>一张一张亲眼看图</strong>，不能只看分数。</div>
<div class="plain"><h3>批评三：流程太繁琐，一个功能从做好到能用要过五道仪式</h3>
探针 → 47 张回归 → 正式报告 → 没见过的考卷 → 全厂体检（几小时）。每一道都有正当理由（怕误伤你的图），但五道加起来，一个功能要拖好几天才能到你手里。而且"没见过的考卷看一次烧一次"，搞得每验证一次像高考，不敢迭代。</div>

<h2>四、建议的新工作方式（改了之后你每天体验到什么）</h2>
<div class="ana"><h3>两车道</h3>
<strong>试验田（每天跑）</strong>：新想法先在练习题和 47 张回归图上快速试，一天内就知道行不行。行→攒着；不行→当天埋掉，只留一句话记录。大部分想法死在这里——死得快也是省钱。<br/>
<strong>上户口（攒一批办一次）</strong>：攒够几个验证过的功能，<strong>挑一天统一</strong>考试（没见过的考卷）、统一全厂体检、统一出正式报告。比如每周五办一次。<br/>
<strong>你的体验变化</strong>：平时每天能看到"今天试了什么、成了几个、死了几个"（有进展感）；每周五能看到一批功能真正进软件（有获得感）。比现在"一个功能拖九天"快 3~5 倍，安全措施一分不少。
</div>
<div class="ana"><h3>日报三块内容（保证没修好也有的报）</h3>
① <strong>死因榜</strong>：今天各种失败各有多少个、比昨天多还是少——就算一行代码没写，这张榜也在动；<br/>
② <strong>在试的想法清单</strong>：每个想法处于"试炼中/已阵亡/待上户口"哪个状态；<br/>
③ <strong>健康指标</strong>：回归通过率、速度。<br/>
另外立一条规矩：<strong>允许"平静的一天"</strong>——没变化就写"今天没变化"，不许为了交差硬做改动。
</div>

<h2>五、优先清单逐个翻译（为什么这样排）</h2>
<table><tr><th style="width:110px">排序</th><th>项目</th><th>为什么排这里</th></tr>
<tr><td><strong style="color:#c0392b">下周第一件</strong></td><td>照片/图表上的"看得见的"日期时间戳</td><td>考卷上发现一件怪事：<strong>连清清楚楚、肉眼一眼可见的时间戳</strong>（不是极淡那种），现在的软件也 0 分！5 道这样的题挂了 4 道。死法和刚解决的文档时间戳疑似同一个病根，而药方很可能现成（软件里已有类似通道）。半天到一天能确诊，一旦治好，这类题从"5 对 1"变成潜在"5 对 4"。投入小、见效大，所以排第一。</td></tr>
<tr><td><strong>快</strong></td><td>时间戳去掉了但网捞了太多水草</td><td>那 3 张"成功"的图，每张都多圈了 6 千~1 万个背景像素。收紧网的口子就行，分数立刻变好看。</td></tr>
<tr><td><strong>快</strong></td><td>极淡水印的门卫放行</td><td>周三实验室满分的那款修复，是被安全闸门误拦的（闸门怕大改，但这款修复恰好命中"大改"的字面定义、实际改得很精细）。放行卫是小改动，一管 4 张图。</td></tr>
<tr><td><strong>中</strong></td><td>白色半透明大字</td><td>考卷上最差的一类（图形 Logo 已按你裁决退出考核之后，它就是倒数第一）。值得治，但得先诊断再估工期。</td></tr>
<tr><td><strong>中</strong></td><td>indoor_signage（室内标牌）那类</td><td>已经探明路子可行，就差动手，一天左右的活。</td></tr>
<tr><td><strong style="color:#95a5a6">不承诺</strong></td><td>5 张"程序真瞎"的图 + document_032</td><td>有些水印淡到/深到和背景物理上无法区分，硬要区分就会误伤原图——违背"不误捞"的铁规矩。如实登记，不投钱。</td></tr></table>

<h2>六、总结成三句话</h2>
<div class="plain">
1. <strong>这一周</strong>：修好了"日期时间戳"这一类水印（等体检走完就生效），同时把考试制度和成绩单制度建起来了；<br/>
2. <strong>问题</strong>：仪式太多、返工太多，你感知到的进展太慢——下周起改两车道；<br/>
3. <strong>下周最该做的一件事</strong>：把"看得见的照片时间戳也 0 分"这个怪病治好，投入一天，可能多救回一类图。
</div>
<p class="subtitle" style="margin-top:40px">本文档为个人查看版，未入报告总目录 ｜ 正式版见同目录《本周工作总结与优先项建议》</p>
</div></body></html>"""


def main():
    REPORT.write_text(build_html(), encoding='utf-8')
    print(f'written: {REPORT} ({len(build_html()) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
