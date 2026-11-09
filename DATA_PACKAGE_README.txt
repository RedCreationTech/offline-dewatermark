数据与模型发布包（2026-11-06）

本包只包含"无法从仓库脚本重建"的数据与模型；训练语料不在包内，按需自行重建（见文末）。
每个 zip 都把内容放在 `data/<名称>/` 或 `models/` 顶层目录：将 zip 内容解压到项目根目录
（与 README.md、start.sh、wm/ 平级）即落位。不要多套一层目录。
发布目录内 SHA256SUMS.txt 是各 zip 的校验和；每个数据集目录内还有逐文件 SHA256SUMS.txt。

本包含 5 个 zip（约 830 MB）：
1. watermark_regression_min_v1.zip（约 51 MB）：固定哨兵回归集，46 个输入、62 张/页。
   价值在"字节固定"——任何重建都会让历史封存基线失去可比性，因此随包发布。
2. watermark_independent_v1.zip（约 139 MB）：锁定的独立验证集（66 条记录）。
   抓取脚本按设计永远抓新图，不可能重建出同一套；锁定集的字节一致性就是它的意义。
3. 应用测试输入.zip（约 286 MB）：用户素材整理出的应用测试输入（快速20张／人工增量110张／
   基础测试202张／real_samples），无任何生成脚本，必须随包分发。
4. watermark_training_increment_v1_1.zip（约 163 MB）：用户扫描件增量集
   （111 强记录＝105 图片＋6 PDF 页，3 弱配对，5 未配对参考；129 训练任务）。
5. models.zip（约 173 MB）：正式推理模型（LaMa＋RapidOCR 1.2.3 备份）与研究权重，详见 MODEL_PACKAGE_README.txt。

不包含：watermark_training_v1 训练语料（3.0 GB，可自行重建，见下）、data/_archive 退役数据、
用户原始收件目录与受限原件、out/ 全部运行产物。使用范围与许可记录随各数据集保留；
用户素材没有因此取得公开再分发许可，本包仅供本项目授权参与者使用，不要公开转发。

验证（解压到仓库根目录后执行）：
.venv/bin/python tools/run_min_regression.py --validate-only
.venv/bin/python data/watermark_training_increment_v1_1/scripts/load_manual_increment.py \
    --root data/watermark_training_increment_v1_1 --view strict --limit 1
HF_HUB_OFFLINE=1 .venv/bin/python -c "from wm import lama, ocr; print('LaMa:', lama.is_available(), 'OCR:', ocr.available())"
校验 zip 完整性：shasum -a 256 -c SHA256SUMS.txt（在发布目录内）

训练语料重建指南（watermark_training_v1，不在本包）：
1. 抓取 CC0 照片底图：.venv/bin/python tools/fetch_cc0_training_bases.py
2. 生成语料（固定种子 SEED=20261026，可用 WM_DATASET_ROOT 指定输出目录）：
   .venv/bin/python tools/build_training_corpus.py
3. 可见度口径修正（与 2026-11-04 数据状态一致）：
   .venv/bin/python tools/regen_low_visibility_samples.py
注意：底图来自联网抓取的公开图库，重建得到的是同分布新语料，不保证与任何历史版本逐字节一致；
做训练集没有问题，但不要拿重建语料冒充固定回归/独立验证集（那两个只认随包发布的字节）。
