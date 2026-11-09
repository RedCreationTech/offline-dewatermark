模型包（2026-10-27）

将 ZIP 的内容解压到项目根目录，得到 models/；不要多套一层目录。

正式推理：
models/inpainting_lama_2025jan.onnx：LaMa，已复制实际文件，非本机缓存符号链接。wm/lama.py会自动读取。
models/ocr/models/：RapidOCR 1.2.3 的检测、方向分类、文字识别三个ONNX模型备份；config.yaml同时保留。
现有 wm/ocr.py 默认读取已安装 rapidocr_onnxruntime 包内模型。安装匹配版本通常已带权重；若缺失，在已激活的项目环境执行：
python models/check_models.py --restore-ocr
该选项会把本包权重复制到当前环境的rapidocr包models目录，仅适用于1.2.3。

检查模型包里的四个ONNX都能加载：
python models/check_models.py
正式程序模型检查：
HF_HUB_OFFLINE=1 .venv/bin/python -c "from wm import lama, ocr; print('LaMa:', lama.is_available(), 'OCR:', ocr.available())"

研究备份：
models/research/slbr_256_devicefree.pt：现有SLBR TorchScript研究权重，未接入正式处理；有干净图误改问题，不应直接启用。
models/research/ 包含原作者README、推理网络源码和已有验证来源/指标，方便继续筛选。
旧中间训练检查点、重复权重和临时图片不包含；本包不是全体实验历史备份。

本包只提供模型与配置，不是Python/驱动/依赖安装包。不同操作系统须先准备相应依赖；严格断网机器需另备对应平台的离线依赖。
模型来源、SHA256和用途见 models/package_manifest.json。原项目源码及Git暂存区未修改。
