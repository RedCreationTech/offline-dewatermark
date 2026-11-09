"""笔刷收尾（黑斑补涂）GUI 自动化验证：真实走通 ImageViewer 鼠标事件链路。

在 real_010 第1页上：提示词定位 → 处理结果视图进入编辑 → 大笔刷涂黑斑整片
暗区 → 按当前选区处理 → 输出前后对照图。用于验证"笔刷收尾"交互与
纸面中位色填补（f02eac6）的实际效果。

用法：set WM_WEVISONE_DEC_BATCH=4 && .venv/Scripts/python.exe tools/run_brush_touchup_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "out" / "笔刷收尾验证_20261109"
SAMPLE = ROOT / "data/watermark_regression_min_v1/inputs/real/real_010.pdf"
# 黑斑中心页坐标（历次实测确定，约"预计工时 3 小时"右侧）
STAIN = (952, 926)


def main() -> int:
    from wm.app import MainWindow, make_app
    from wm.wevisone_client import Service
    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    app = make_app()
    win = MainWindow()
    win.show()
    app.processEvents()

    # 1) 导入 PDF（同 GUI"新增批次"）
    win.open_file(str(SAMPLE))
    app.processEvents()
    # 2) 提示词模式 + 参数（同面板设置）
    win.mode_box.setCurrentIndex(1)          # 提示词定位
    win.service_box.setCurrentIndex(1)       # 按需启动
    win.prompt_edit.setText('斜向黑字"内部演示稿"')
    app.processEvents()
    # 3) 只处理第 1 页（同"处理/重处理当前图片"）
    win.start_run(only_page=0)
    if win.worker is None:
        print("队列未启动", file=sys.stderr)
        return 1
    win.worker.wait(300_000)
    app.processEvents()
    result = win.batch.results.get(0)
    if result is None:
        print("处理失败", win.batch.errors.get(0), file=sys.stderr)
        return 1
    before = result.output.image.copy()
    Image.fromarray(before[..., :3]).save(OUT / "1_笔刷前_处理结果.png")
    print("提示词处理完成，已进入笔刷收尾")

    # 4) 编辑水印选区（同按钮）
    win.assist_select_btn.click()
    app.processEvents()
    # 5) 切到处理结果视图并调大笔刷
    win.set_view_mode("处理结果")
    win.brush_size.setValue(220)
    app.processEvents()

    # 6) 页坐标 → 画布坐标，涂一笔盖住黑斑整片暗区
    rect = win.canvas.image_rect()
    def to_canvas(px, py):
        return QPointF(rect.left() + px / 1654 * rect.width(),
                       rect.top() + py / 2339 * rect.height())
    cx = to_canvas(*STAIN)
    QTest.mousePress(win.canvas, Qt.LeftButton, pos=cx.toPoint())
    QTest.mouseMove(win.canvas, (cx + QPointF(6, 0)).toPoint())
    QTest.mouseRelease(win.canvas, Qt.LeftButton, pos=(cx + QPointF(6, 0)).toPoint())
    app.processEvents()
    print(f"已涂抹：{win.selection_notice.text()}")

    # 7) 按当前选区处理（同按钮），等待辅助线程
    win.assist_confirm_btn.click()
    if win.assist_worker is not None:
        win.assist_worker.wait(120_000)
    app.processEvents()
    result = win.batch.results.get(0)
    after = result.output.image
    Image.fromarray(after[..., :3]).save(OUT / "2_笔刷后_处理结果.png")

    # 8) 前后对照（黑斑区放大）
    x0, y0 = 850, 800
    crop_b = before[y0:y0 + 340, x0:x0 + 460]
    crop_a = after[y0:y0 + 340, x0:x0 + 460]
    joined = np.vstack([crop_b, np.full((8, 460, 3), 40, np.uint8), crop_a])
    Image.fromarray(joined).save(OUT / "3_前后对照_上前下后.png")
    region = after[y0:y0 + 360, x0:x0 + 420].astype(np.float32).mean()
    print(f"黑斑区平均灰度：笔刷前 {before[y0:y0+360, x0:x0+420].astype(np.float32).mean():.1f} "
          f"→ 笔刷后 {region:.1f}（255=纯白）")
    print(f"产物目录：{OUT}")
    win.wevisone.stop()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # 离屏 Qt 清理会偶发挂起，验证脚本直接退出
