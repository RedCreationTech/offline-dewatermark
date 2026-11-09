"""生成周报 HTML：汇总本周工作进展、关键指标和下周计划。

用法：
    .venv/bin/python tools/build_weekly_report.py                  # 本周
    .venv/bin/python tools/build_weekly_report.py --week 2026-11-03  # 指定周（以周一为起点）

约定：
    - 使用 tools/templates/report_base.html 统一模板
    - 输出到 out/周报/周报_YYYY-MM-DD.html
    - 数据从 tools/weekly_data.json 读取（手动维护或从会话记录提取）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.template_engine import write_report


def get_week_range(week_start: date) -> tuple[date, date]:
    """返回指定周的起止日期（周一到周日）。"""
    start = week_start - timedelta(days=week_start.weekday())
    end = start + timedelta(days=6)
    return start, end


def load_weekly_data(data_path: Path, week_start: date) -> dict:
    """加载指定周的数据。"""
    if not data_path.exists():
        return {
            "week_start": week_start.isoformat(),
            "summary": "暂无数据",
            "highlights": [],
            "metrics": {},
            "next_week": [],
        }

    all_data = json.loads(data_path.read_text(encoding="utf-8"))
    week_key = week_start.isoformat()
    return all_data.get(week_key, {
        "week_start": week_start.isoformat(),
        "summary": f"第 {week_key} 周数据未录入",
        "highlights": [],
        "metrics": {},
        "next_week": [],
    })


def render_content(data: dict, week_start: date, week_end: date) -> str:
    """渲染报告正文 HTML。"""
    highlights_html = ""
    if data.get("highlights"):
        items = "".join(f"<li>{h}</li>" for h in data["highlights"])
        highlights_html = f"<div class='section'><h2>本周亮点</h2><ul>{items}</ul></div>"

    metrics = data.get("metrics", {})
    metrics_html = ""
    if metrics:
        rows = "".join(
            f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
            for k, v in metrics.items()
        )
        metrics_html = f"<div class='section'><h2>关键指标</h2><table>{rows}</table></div>"

    next_week_html = ""
    if data.get("next_week"):
        items = "".join(f"<li>{n}</li>" for n in data["next_week"])
        next_week_html = f"<div class='section'><h2>下周计划</h2><ul>{items}</ul></div>"

    return f"""
<a class="back" href="../水印验证报告总目录.html">← 返回报告总目录</a>
<h1>工作周报 {week_start.isoformat()} ~ {week_end.isoformat()}</h1>
<p class="byline">生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

<div class="section">
<h2>本周总结</h2>
<p>{data.get("summary", "暂无总结")}</p>
</div>

{highlights_html}
{metrics_html}
{next_week_html}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--week", type=str, default=None,
                    help="周报起始日期（YYYY-MM-DD，周一），默认本周")
    ap.add_argument("--data", type=Path, default=Path("tools/weekly_data.json"),
                    help="周报数据文件路径（默认：tools/weekly_data.json）")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出文件路径（默认：out/周报/周报_YYYY-MM-DD.html）")
    args = ap.parse_args()

    if args.week:
        week_start = date.fromisoformat(args.week)
    else:
        today = date.today()
        week_start = today - timedelta(days=today.weekday())

    week_start, week_end = get_week_range(week_start)
    data = load_weekly_data(args.data, week_start)

    content = render_content(data, week_start, week_end)

    if args.out:
        output_path = args.out
    else:
        output_dir = Path("out/周报")
        output_path = output_dir / f"周报_{week_start.isoformat()}.html"

    result = write_report(
        output_path=output_path,
        template="report_base.html",
        title=f"工作周报 {week_start.isoformat()}",
        content=content,
        data=data,
        footer=f"周报数据源：{args.data}",
    )

    print(f"已生成周报：{result}")


if __name__ == "__main__":
    main()
