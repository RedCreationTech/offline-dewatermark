# 安装与换机说明

更新日期：2026-11-02。程序运行和质量边界以 [README](README.md) 及 [实际流程](docs/去水印处理流程图.md) 为准。

## 取得程序

```bash
git clone git@github.com:RedCreationTech/offline-dewatermark.git
cd offline-dewatermark
./start.sh setup
```

本机若无法访问 SSH 22 端口，可使用同一仓库的 443 端口地址：

```bash
git clone ssh://git@ssh.github.com:443/RedCreationTech/offline-dewatermark.git
```

首次准备可能联网。严格断网部署需在有网机器准备相同操作系统及架构的 Python 依赖包、模型和字体，再在目标机器安装验证。不要直接复制其他操作系统的 `.venv/`。

## 准备资源

按 [数据与模型放置说明](资源放置说明.md) 放好 `data/`、`models/`；旧资源包不等于当前完整清单，不能覆盖已有数据。

```bash
./start.sh check-data
./start.sh gui
```

LaMa 权重缺失或加载失败会显式报错，不以传统修复冒充同一模型结果。`WM_LAMA_ONNX` 可指定本地模型文件；`WM_LAMA_CUDA=1` 请求 GPU 后端，是否可用须以目标机器的实际加载与运行验证为准。

中文合成工具使用已安装的中文字体，必要时设置 `WM_FONTS_DIR`。重建训练语料时优先使用资源包中的原语料字体，缺失时显式设置 `WM_CORPUS_FONT` 指向同一字体并保留许可，不能静默换字体后声称结果完全可复现。历史导出报告工具另使用 `WM_REPORT_FONT` 指定中文字体文件。

Windows 请在适配环境中使用 Python 模块入口；`.venv/bin/python` 对应 `.venv\Scripts\python.exe`。无桌面测试可设 `QT_QPA_PLATFORM=offscreen`，但不能以离屏检查替代真实窗口验收。

## 换机需要另行备份

Git 不提供本地测试报告、样本展示图片、完整封存基线、原始输入导出、内部方案和个人配置。需要保留时单独备份对应目录；个人密钥不要放入共享备份。详见 [仓库范围](docs/仓库内容与本地资料说明.md)。

`.regression-control/` 保存本机基线指针，不能把旧机器的绝对路径直接作为新机器的可比较基线。没有可用基线时显式建立观察基线，它记录缺陷，不代表质量合格。

## 验证

```bash
./start.sh test --plan
./start.sh test
```

发布或更新默认基线前另运行 `./start.sh test-release`。模型、数据、基线或平台条件不一致时，先解决条件差异；不能替换基线或放宽门槛制造通过。完整工程测试、局部图片通过均不能证明通用去水印效果达标，当前尚无独立泛化验证结论。
