"""报告模板引擎：加载 HTML 模板并注入数据。

用法：
    from tools.template_engine import render_report

    html = render_report(
        template="report_base.html",
        title="测试报告 2026-11-09",
        content="<h1>报告内容</h1>",
        data={"key": "value"},
        footer="生成时间：2026-11-09 17:00"
    )
"""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_report(
    template: str = "report_base.html",
    title: str = "报告",
    content: str = "",
    data: dict | list | None = None,
    footer: str = "",
) -> str:
    """渲染报告 HTML。

    Args:
        template: 模板文件名（在 tools/templates/ 下）
        title: 页面标题
        content: HTML 正文内容
        data: 注入到 __DATA__ 占位符的数据（会序列化为 JSON）
        footer: 页脚文本

    Returns:
        完整的 HTML 字符串
    """
    template_path = TEMPLATES_DIR / template
    if not template_path.exists():
        raise FileNotFoundError(f"模板不存在：{template_path}")

    html = template_path.read_text(encoding="utf-8")

    html = html.replace("__TITLE__", title)
    html = html.replace("__CONTENT__", content)
    html = html.replace("__FOOTER__", footer)

    data_json = json.dumps(data or {}, ensure_ascii=False, indent=2)
    html = html.replace("__DATA__", data_json)

    return html


def write_report(
    output_path: Path,
    **kwargs,
) -> Path:
    """渲染并写入文件。

    Args:
        output_path: 输出文件路径
        **kwargs: 传给 render_report 的参数

    Returns:
        输出文件路径
    """
    html = render_report(**kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
