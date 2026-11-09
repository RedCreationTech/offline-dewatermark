#!/usr/bin/env python3
"""将剩余问题总结 markdown 转为 HTML + 重命名图片，打包 zip。"""

import os, re, shutil, zipfile
from pathlib import Path

PROJECT = Path("/Users/kevinli/Desktop/挑战赛需求/offline-dewatermark-ai")
MD_FILE = PROJECT / "docs" / "剩余问题总结_20261109.md"
OUT_DIR = PROJECT / "out" / "剩余问题总结_20261109"
HTML_FILE = OUT_DIR / "剩余问题总结_20261109.html"
ZIP_FILE = OUT_DIR / "剩余问题总结_20261109.zip"
IMG_DIR = OUT_DIR / "images"

RENAME_MAP = {
    "原始截图_08_建筑照片点阵_20261028.png":          "01_偏移点阵_建筑照片.png",
    "001_地铁指示牌_复测结果.png":                      "02_边缘漏选_地铁指示牌.png",
    "原始截图_03_表格文档_水印选区_20261028.png":       "03_正文保护不修复_表格文档选区.png",
    "原始截图_04_表格文档_处理结果_20261028.png":       "03_正文保护不修复_表格文档结果.png",
    "原始截图_05_小车间_水印选区_20261028.png":         "04_补全替换选区_小车间.png",
    "原始截图_01_高清车间_处理结果_20261028.png":       "07_颜色优先失配_高清车间.png",
    "照片失败_结果.png":                                "08_LaMa黑斑_修复结果.png",
    "照片失败_原图.png":                                "08_LaMa黑斑_原图对照.png",
    "002_汤碗_复测结果.png":                            "09_纹理重建失真_汤碗结果.png",
    "002_汤碗_干净参考.png":                            "09_纹理重建失真_汤碗参考.png",
    "HCimage局部图像对照_20261030.png":                 "10_交叠文字丢失_局部对照.png",
    "s040_heavy_dark_medium_photo.png":                 "11_暗色水印遮盖_测试样本.png",
    "样式_叠加.png":                                    "14_印章平色叠加_样式.png",
    "用户提供_处理耗时完整截图_20261030.png":            "15_性能超时_耗时截图.png",
}

def resolve_img_path(src: str) -> Path | None:
    docs_dir = PROJECT / "docs"
    candidate = (docs_dir / src).resolve()
    if candidate.exists():
        return candidate
    return None

def renamed_filename(original_name: str) -> str:
    return RENAME_MAP.get(original_name, original_name)

def convert_md_to_html(md_text: str) -> str:
    lines = md_text.strip().split("\n")
    html_parts = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("# "):
            html_parts.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_parts.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("> "):
            html_parts.append(f'<p class="meta">{line[2:]}</p>')
        elif line.startswith("---"):
            html_parts.append("<hr>")
        elif line.startswith("| "):
            table_lines = []
            while i < len(lines) and lines[i].startswith("| "):
                table_lines.append(lines[i])
                i += 1
            i -= 1
            html_parts.append(render_table(table_lines))
        elif line.startswith("**") and line.endswith("**"):
            html_parts.append(f'<p class="section-label"><strong>{line.strip("*")}</strong></p>')
        elif line.startswith("!["):
            m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if m:
                alt, src = m.group(1), m.group(2)
                img_path = resolve_img_path(src)
                if img_path:
                    new_name = renamed_filename(img_path.name)
                    html_parts.append(f'<p class="img-block"><img src="images/{new_name}" alt="{alt}"></p>')
                else:
                    html_parts.append(f'<p class="img-missing">[图片缺失: {src}]</p>')
        elif line.startswith("#") and "—" in line:
            html_parts.append(f'<p class="img-caption">{line}</p>')
        elif line.strip() == "":
            pass
        else:
            html_parts.append(f"<p>{line}</p>")
        i += 1

    body = "\n".join(html_parts)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>剩余问题总结 2026-11-09</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; background: #f5f6fa; color: #2d3436; padding: 24px; line-height: 1.6; }}
  .container {{ max-width: 1100px; margin: 0 auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 16px rgba(0,0,0,0.08); padding: 36px 40px; }}
  h1 {{ font-size: 26px; color: #2d3436; border-bottom: 3px solid #0984e3; padding-bottom: 10px; margin-bottom: 8px; }}
  h2 {{ font-size: 20px; color: #0984e3; margin-top: 32px; margin-bottom: 12px; padding-left: 12px; border-left: 4px solid #0984e3; }}
  .meta {{ color: #636e72; font-size: 14px; margin-bottom: 20px; }}
  .section-label {{ font-size: 15px; color: #2d3436; margin-top: 16px; margin-bottom: 8px; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0 16px; font-size: 14px; }}
  th {{ background: #0984e3; color: #fff; padding: 10px 12px; text-align: left; font-weight: 500; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #dfe6e9; vertical-align: top; }}
  tr:nth-child(even) td {{ background: #f8f9fa; }}
  tr:hover td {{ background: #eef5ff; }}
  .img-block {{ margin: 12px 0; text-align: center; }}
  .img-block img {{ max-width: 100%; height: auto; border: 1px solid #dfe6e9; border-radius: 6px; box-shadow: 0 1px 6px rgba(0,0,0,0.1); }}
  .img-caption {{ font-size: 13px; color: #636e72; margin: 6px 0 4px; font-weight: 500; }}
  .img-missing {{ color: #d63031; font-style: italic; font-size: 13px; }}
  hr {{ border: none; border-top: 1px solid #dfe6e9; margin: 24px 0; }}
  p {{ margin: 4px 0; }}
  .note {{ background: #ffeaa7; border-left: 4px solid #fdcb6e; padding: 12px 16px; border-radius: 0 6px 6px 0; margin-top: 20px; font-size: 14px; }}
</style>
</head>
<body>
<div class="container">
{body}
</div>
</body>
</html>"""

def render_table(table_lines: list[str]) -> str:
    if len(table_lines) < 2:
        return ""
    def parse_row(line):
        cells = [c.strip() for c in line.strip("|").split("|")]
        return cells
    headers = parse_row(table_lines[0])
    data_rows = []
    for line in table_lines[2:]:
        if not re.match(r'^\|[\s\-|]+\|$', line):
            data_rows.append(parse_row(line))
    html = "<table><thead><tr>"
    for h in headers:
        h_clean = h.replace("**", "")
        html += f"<th>{h_clean}</th>"
    html += "</tr></thead><tbody>"
    for row in data_rows:
        html += "<tr>"
        for cell in row:
            html += f"<td>{cell}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

def collect_image_paths(md_text: str) -> list[tuple[str, Path]]:
    pairs = []
    for m in re.finditer(r'!\[[^\]]*\]\(([^)]+)\)', md_text):
        src = m.group(1)
        img_path = resolve_img_path(src)
        if img_path:
            pairs.append((src, img_path))
    return pairs

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if IMG_DIR.exists():
        shutil.rmtree(IMG_DIR)
    IMG_DIR.mkdir()

    md_text = MD_FILE.read_text(encoding="utf-8")
    md_text = md_text.replace(
        "**备注**：",
        '<div class="note"><strong>备注</strong>：'
    ).rstrip()
    if not md_text.endswith("</div>"):
        md_text += "</div>"

    html = convert_md_to_html(md_text)
    HTML_FILE.write_text(html, encoding="utf-8")
    print(f"HTML 已生成: {HTML_FILE}")
    print(f"HTML 大小: {HTML_FILE.stat().st_size / 1024:.0f} KB")

    img_pairs = collect_image_paths(md_text)
    print(f"\n复制并重命名 {len(img_pairs)} 张图片到 images/")
    for src, img_path in img_pairs:
        new_name = renamed_filename(img_path.name)
        dst = IMG_DIR / new_name
        shutil.copy2(img_path, dst)
        print(f"  {img_path.name}")
        print(f"    → {new_name}  ({dst.stat().st_size / 1024:.0f} KB)")

    print(f"\n打包 ZIP...")
    with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(HTML_FILE, HTML_FILE.name)
        for img_file in sorted(IMG_DIR.iterdir()):
            zf.write(img_file, f"images/{img_file.name}")

    print(f"ZIP 已打包: {ZIP_FILE}")
    print(f"ZIP 大小: {ZIP_FILE.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
