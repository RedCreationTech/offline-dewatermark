# 离线智能去水印原型

本地处理图片与 PDF，提供桌面界面、批量处理、手动画笔选区和结果导出。
**软件与测试管控已经建立，自动去水印效果仍不稳定；目前没有独立泛化验证结论。**

## 项目入口

- [架构与代码说明](docs/knowme.md)
- [实际处理流程与已知边界](docs/去水印处理流程图.md)
- [去水印机制与流程总览](docs/去水印机制与流程总览.html)（通俗汇总所有机制，细节以上一条为准）
- [安装与换机说明](MIGRATION.md)
- [数据与模型放置说明](资源放置说明.md)
- [仓库内容与本地资料说明](docs/仓库内容与本地资料说明.md)

仓库保存程序、工程测试、通用维护工具和必要技术说明。测试报告、输入输出图片、样本展示素材、历史封存、内部任务和个人助手配置仅保留在本机，不随新版本提交。详情见仓库内容说明。

## 启动

```bash
git clone git@github.com:RedCreationTech/offline-dewatermark.git
cd offline-dewatermark
./start.sh setup       # 首次准备依赖与模型，可能需要联网
./start.sh gui         # 打开桌面程序
```

Windows（PowerShell/cmd）使用等价的 `start.bat`（或 `start.ps1`），子命令与 `./start.sh` 完全一致，例如 `start.bat gui`；脚本内部已处理虚拟环境路径与 UTF-8 编码。

日常使用已有本地依赖和模型；严格离线部署须提前准备资源和对应平台的 Python 依赖。图片输入与评分参考分开，干净图、alpha 和标注不能用于决定当前单图选区或处理结果。

数据与模型从本仓库的 [2026年10月27日资源包](https://github.com/RedCreationTech/offline-dewatermark/releases/tag/resources-2026-10-27) 下载。2026-11-02 已复制原 Release 的六个附件，文件名、大小与 SHA256 保持一致，原仓库 Release 继续保留。资源仍属于 2026-10-27 快照，不能直接视为当前清单，现有工作目录不要被旧包覆盖。

## 数据范围

<!-- current-dataset-counts:start -->
| 数据 | 当前数量 | 用途 |
|---|---|---|
| 分层数据集 | data/分层数据集/（斜向平铺_白底 13 张起） | 单变量变化生成，效果测试与定档来源 |
| 效果分类数据集 | 效果好／提示词可处理／待解决 三目录 | 人工定档归档，防退化回归对象 |
| 独立验证集 | data/watermark_independent_v1（锁柜） | 一次性基线后冻结，看过即降级 |

固定回归集与旧训练语料已于 2026-11-23 退役并直接删除（登记见 config/regression.json 的 retirements 节）；历史成绩作废，当前口径以分层控制轮为准。
旧应用输入已删除；完整清单检查不再适用。
<!-- current-dataset-counts:end -->

上述数量是已有本地清单的记录，不代表数据随 Git 提供。实际来源、清单和封存证据需另外备份。旧 real_samples、syn、公开40对样本及旧入口已经退役；三张手写样本的移除范围见 [回归配置](config/regression.json) 的 `retirements.previous`，不得通过恢复旧包重新加入当前回归。

## 测试与质量边界

```bash
./start.sh test         # 工程单元测试全量（约 340 项）
# 图像效果回归：跑分层数据集控制轮 → 报告里人工定档 → 归档（见下「效果分类数据集」）
```

首次建立观察基线使用 `./start.sh freeze-baseline`，它只记录现状，包括缺陷。[回归配置](config/regression.json) 的 `control` 节定义控制轮放行门槛。数据、模型和可比较基线缺失时不能宣称验收完成。

有 NVIDIA 显卡的机器可在任意子命令后加 `--gpu`：WeVisOne 分割探针用 `cuda:0`；LaMa 仅当主环境装有可用的 CUDA 版 onnxruntime 时才走 `WM_LAMA_CUDA=1`，否则保持 CPU 并提示。示例：`./start.sh gui --gpu`（或 `start.bat gui --gpu`）；`./start.sh wevisone --gpu [提示词] [图片]` 单独跑 WeVisOne 单图分割探针（默认 cpu），输出到 `out/WeVisOne环境检查_NN`。注意：现网 LaMa 量化模型在 CUDA 12 版 onnxruntime 上缺少量化算子支持、DirectML 亦不支持其 FFC 结构，本机经实测只能在 CPU 运行；未来如升级驱动与 CUDA 13 运行时，该模型在 CUDA 13 版 onnxruntime 上的可用性需重新实测。

工程单元测试、分层控制轮单组通过与局部实验不能替代完整放行（全部变量组控制轮通过）或独立验证。

纸面彩色阴影文字分层反解已在 2026-10-29 接入默认处理；表格残影、正常彩色内容保护和耗时问题仍未解决。部分照片和淡字存在漏检，正文、数字、纹理可能受损；无可靠参考时保持待复核，不将未知质量计为成功。已有实验细节保留在本地报告中，实际行为以长期流程说明和源码为准。

## 实验与本地报告

分层合成测试可分别观察透明度、颜色和字号对斜向平铺文字的影响。每轮使用新目录，已有样本可直接从第二条命令开始。

```bash
.venv/bin/python tools/layered_samples.py prepare --out out/斜向平铺分层样本_新轮次
HF_HUB_OFFLINE=1 .venv/bin/python tools/layered_samples.py run --dataset-dir out/斜向平铺分层样本_新轮次
```

第二条命令按桌面默认参数冻结并运行样本，将逐页结果写入 `out/项目防退化控制/`，并更新本地报告总目录。合成测试用于核对样本表现和报告链路，独立泛化验证另行开展。

**效果分类数据集**：每轮测试后在明细报告里人工定档（卡片按钮或按 1/2/3 选 效果好／提示词可处理／待解决，机器只按 召回≥98% 且 精确率≥90% 预筛候选），点「导出任档结果」后归档进三个数据集：

```bash
.venv/bin/python tools/archive_classified.py --json 效果定档_*.json   # 在浏览器下载目录里找到导出的 JSON
```

三个数据集在 `data/效果分类数据集/`（效果好／提示词可处理／待解决），每样本附判定记录。定档标准与流程见该目录 README.md。

自动比较处理方法候选默认关闭，尚未通过完整放行与独立来源验证，不建议作为默认处理：

```bash
./start.sh gui --experimental-auto-selection
.venv/bin/python -m wm.cli run 输入图片 --experimental-auto-selection --out out/自动比较试用
```

本地报告索引为 `out/水印验证报告总目录.html`，只有具备相应本地产物时才可查看。报告生成器仍保留在 `tools/`，生成内容受 `.gitignore` 管理；无需为了查看或开发应用，把私人报告重新加入 Git。

面向用户的报告版式约定：正文全平铺展示，不用折叠列表藏内容；专业名词和代码段必须配解释，术语解释只通过词旁 ❓ 浮层提供（点击弹出，再点一次、点空白处或按 Esc 收起），不另设重复词表。约束细则见 `AGENTS.md` 第 8 节。

```bash
.venv/bin/python tools/organize_project_docs.py         # 从本地清单生成说明
.venv/bin/python tools/organize_project_docs.py --check # 核对本地生成内容与链接
.venv/bin/python tools/build_report_index.py            # 更新本地报告索引
```

HTML 报告约定（2026-11-02 起）：面向用户的报告图片一律 base64 内嵌，单个 HTML 发出即可完整查看；所有新建报告都要能通过 `out/水印验证报告总目录.html` 分类跳转到达；总目录与报告之间只用相对路径，交付前压缩整个目录实际解压验证。详见[代码与工具用途说明](docs/代码与工具用途说明.html)中的“HTML 报告编写与分享约定”。

这些工具依赖本地数据或封存记录；仅克隆仓库不保证能重新生成全部历史报告。通用语料工具缺字体时需设置 `WM_CORPUS_FONT`，历史导出报告工具使用 `WM_REPORT_FONT`，不依赖个人主目录。
