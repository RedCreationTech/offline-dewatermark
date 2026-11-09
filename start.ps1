# 离线智能去水印 — 一键启动脚本（Windows PowerShell 版，与 start.sh 等价）
# 用法：
#   .\start.ps1              等价于 .\start.ps1 test
#   .\start.ps1 setup        只装环境（磁盘预检 + venv + 依赖 + LaMa 模型 + 字体检查）
#   .\start.ps1 test         按修改影响选择日常检查；--plan 只预览范围和耗时
#   .\start.ps1 test-release 工程预检后完整三模式对照，供版本放行
#   .\start.ps1 freeze-baseline 冻结当前代码、资源、参数并运行，建立首次观察基线
#   .\start.ps1 control-inventory 校验全部应用输入、参考和既有固定集
#   .\start.ps1 control ...  防退化控制工具（resume/compare/promote 等）
#   .\start.ps1 check-data   校验固定回归集，不运行模型
#   .\start.ps1 gui [文件]   启动桌面前端
#   .\start.ps1 wevisone [--gpu] [query] [image]  单图分割探针（默认 cpu，--gpu 走 cuda:0）
#   任意子命令后加 --gpu     LaMa 走 WM_LAMA_CUDA=1，WeVisOne 走 cuda:0
#   .\start.ps1 eval         仅跑固定核心回归集
#   .\start.ps1 report       显示最近一次最小回归报告路径
# 若提示“禁止运行脚本”，可用 powershell -ExecutionPolicy Bypass -File .\start.ps1 <子命令>
# 或直接运行 start.bat（内部已带 Bypass）。

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'  # Windows GBK 代码页下保证 Python 按 UTF-8 读写项目数据
Set-Location $PSScriptRoot

$PY = $env:PYTHON
$VENV = '.venv'
$MIN_FREE_GB = 5
$VenvPython = Join-Path $VENV 'Scripts\python.exe'
$VenvPip    = Join-Path $VENV 'Scripts\pip.exe'

function Say($msg)  { Write-Host "[start] $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[start] $msg" -ForegroundColor Yellow }
function Die($msg)  { Write-Host "[start] $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 磁盘空间预检
function Check-Disk {
    $drive = (Get-Location).Drive.Name
    $freeGB = [math]::Floor((Get-PSDrive -Name $drive).Free / 1GB)
    Say "磁盘可用 ${freeGB}GB（要求 ≥${MIN_FREE_GB}GB）"
    if ($freeGB -lt $MIN_FREE_GB) { Die "磁盘空间不足 ${MIN_FREE_GB}GB，请先清理" }
}

# ---------------------------------------------------------------- Python 选择
function Pick-Python {
    if ($PY) { return }
    # Windows 优先用 py 启动器锁定版本，再退回 PATH 里的 python
    $localPythons = @()
    foreach ($v in @('Python314', 'Python313', 'Python312')) {
        $pp = Join-Path $env:LOCALAPPDATA "Programs\Python\$v\python.exe"
        if (Test-Path $pp) { $localPythons += , @($pp) }
    }
    foreach ($cand in @($localPythons + @(@('py', '-3.14'), @('py', '-3.13'), @('py', '-3.12'), @('python')))) {
        $exe = $cand[0]
        $extra = @()
        if ($cand.Count -gt 1) { $extra = @($cand[1]) }
        & $exe @extra -V 2>$null
        if ($LASTEXITCODE -eq 0) { $script:PY = "$exe $($extra -join ' ')"; break }
    }
    if (-not $PY) { Die "找不到 python（可先安装 Python 3.14 或用 `$env:PYTHON 指定）" }
    Say "使用解释器：$PY（$(Invoke-Expression "& $PY -V 2>&1")）"
}

# ---------------------------------------------------------------- 依赖安装
function Setup-Venv {
    if (-not (Test-Path $VenvPython)) {
        Pick-Python
        Say "创建虚拟环境 $VENV"
        Invoke-Expression "& $PY -m venv $VENV"
    }
    if (Test-Path (Join-Path $VENV '.audit_deps_ok')) {
        Say "依赖已安装（跳过；删除 $VENV\.audit_deps_ok 可强制重装）"
        return
    }
    Say "安装依赖（先校验本平台能否装 requirements.lock 原样清单）"
    # lock 是 Windows/Python3.14 机生成的 freeze；装不上时按 MIGRATION.md §5
    # 退回 requirements.txt，仅放宽装不上的包，其余仍按钉死版本安装。
    & $VenvPip install --dry-run -q -r requirements.lock 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        & $VenvPip install -r requirements.lock
    } else {
        Warn "requirements.lock 原样装不上（平台 wheel 差异），改用 requirements.txt 并放宽 onnxruntime"
        Get-Content requirements.txt | Where-Object { $_ -notmatch '^onnxruntime==' } |
            Set-Content 'req_local.txt'
        & $VenvPip install -r 'req_local.txt'
        & $VenvPip install onnxruntime
        if ($LASTEXITCODE -ne 0) { Die "onnxruntime 安装失败" }
        Remove-Item 'req_local.txt' -ErrorAction SilentlyContinue
        $ortVer = (& $VenvPip show onnxruntime | Select-String '^Version:').Line -replace 'Version:\s*', ''
        Say "onnxruntime 放宽到 ${ortVer}（其余按 requirements.txt 钉死）"
    }
    New-Item -ItemType File -Path (Join-Path $VENV '.audit_deps_ok') -Force | Out-Null
}

function Test-Deps {
    Say "依赖自检"
    & $VenvPython -c "import numpy, cv2, skimage, fitz, PIL, onnxruntime; print('  核心依赖 OK')"
    & $VenvPython -c "import PySide6" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  PySide6 OK（GUI 可用）"
    } else {
        Warn "PySide6 不可用，gui 子命令将失败"
    }
}

# ---------------------------------------------------------------- LaMa 模型
function Ensure-Model {
    & $VenvPython -c "from wm.lama import is_available; exit(0 if is_available() else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Say "LaMa 模型就绪"
        return
    }
    Say "首次运行：下载 LaMa ONNX 模型（~93MB，来自 HuggingFace opencv/inpainting_lama）"
    Check-Disk
    $code = @'
import numpy as np, cv2
from wm.lama import inpaint
img = np.full((512, 512, 3), 255, np.uint8)
mask = np.zeros((512, 512), np.uint8); mask[128:384, 128:384] = 255
inpaint(img, mask)  # 触发下载并做一次热身推理
print("LaMa 下载并预热完成")
'@
    $env:HF_HUB_OFFLINE = '0'
    $code | & $VenvPython -
    $env:HF_HUB_OFFLINE = '1'
}

# ---------------------------------------------------------------- 字体检查
function Check-Fonts {
    $check = @'
from wm.synth import _resolve_font
_resolve_font('hei'); _resolve_font('body'); _resolve_font('head')
'@
    $check | & $VenvPython - 2>$null
    if ($LASTEXITCODE -eq 0) {
        Say "中文字体 OK"
        return
    }
    # Windows：微软雅黑/等线通常已随系统提供，先提示环境变量方式
    Warn "未找到中文字体：合成语料/评测会报错。请安装 Noto Sans CJK 或用 `$env:WM_FONTS_DIR 指定字体目录"
}

# ---------------------------------------------------------------- 各子命令
function Invoke-Setup {
    Check-Disk; Setup-Venv; Test-Deps; Ensure-Model; Check-Fonts
    Say "环境就绪。可以 .\start.ps1 test 或 .\start.ps1 gui"
}

function Invoke-Eval {
    Say "固定核心回归：36 个输入文件 / 48 张或页"
    $env:HF_HUB_OFFLINE = '1'
    & $VenvPython tools/run_min_regression.py --suite core
}

function Invoke-Report {
    $code = @'
from pathlib import Path
reports=list(Path('out/项目防退化控制').glob('*/项目防退化控制报告.html'))
reports+=list(Path('out/防退化对照').glob('*/项目防退化控制报告.html'))
reports+=list(Path('out/min_regression').glob('*/report.html'))
if not reports:
    raise SystemExit('尚无固定最小回归结果；请先执行 ./start.ps1 test 或 ./start.ps1 eval。')
print(max(reports,key=lambda p:p.stat().st_mtime_ns).resolve())
'@
    $code | & $VenvPython -
}

function Invoke-Gui {
    Invoke-Setup
    & $VenvPython -m wm.app @Args
    exit $LASTEXITCODE
}

$sub = 'test'
$rest = @($args)
if ($rest.Count -gt 0) { $sub = $rest[0]; $rest = $rest[1..($rest.Count - 1)] }

# --gpu 标志（可出现在任意位置）：LaMa 用 WM_LAMA_CUDA=1，WeVisOne 用 cuda:0
$Device = 'cpu'
$gpu = $false
$kept = @()
foreach ($a in $rest) {
    if ($a -eq '--gpu') { $gpu = $true } else { $kept += $a }
}
$rest = $kept
if ($gpu) {
    $Device = 'cuda:0'
    $cudaOk = & $VenvPython -c "import onnxruntime as o; print('CUDAExecutionProvider' in o.get_available_providers())" 2>$null
    if ($cudaOk -contains 'True') {
        $env:WM_LAMA_CUDA = '1'
        Say "GPU 模式：LaMa 使用 WM_LAMA_CUDA=1，WeVisOne 使用 cuda:0"
    } else {
        Warn "GPU 模式：主环境未安装可用的 CUDA 版 onnxruntime，LaMa 保持 CPU；WeVisOne 使用 cuda:0"
    }
}

switch ($sub) {
    'setup'  { Invoke-Setup }
    'eval'   { Invoke-Setup; Invoke-Eval }
    'test'   {
        Say "日常按影响分层检查；局部通过不能更新默认基线"
        $env:HF_HUB_OFFLINE = '1'; $env:QT_QPA_PLATFORM = 'offscreen'
        & $VenvPython tools/daily_check.py @rest
    }
    'test-release' {
        $env:HF_HUB_OFFLINE = '1'; $env:QT_QPA_PLATFORM = 'offscreen'
        & $VenvPython tools/daily_check.py @rest --level full
    }
    'freeze-baseline' {
        $env:HF_HUB_OFFLINE = '1'; $env:QT_QPA_PLATFORM = 'offscreen'
        & $VenvPython tools/regression_control.py freeze @rest
    }
    'control-inventory' {
        $env:HF_HUB_OFFLINE = '1'; $env:QT_QPA_PLATFORM = 'offscreen'
        & $VenvPython tools/regression_control.py inventory @rest
    }
    'control' {
        $env:HF_HUB_OFFLINE = '1'; $env:QT_QPA_PLATFORM = 'offscreen'
        & $VenvPython tools/regression_control.py @rest
    }
    'wevisone' {
        $wvPy = 'out/third_party/WeVisOne-VL/.venv/Scripts/python.exe'
        if (-not (Test-Path $wvPy)) { Die 'WeVisOne 环境未安装，请先按配置文档第 06 节准备 out/third_party/WeVisOne-VL' }
        $env:HF_HUB_OFFLINE = '1'
        & $wvPy out/试跑_wevisone.py $Device @rest
    }
    'report'     { Invoke-Report }
    'check-data' { & $VenvPython tools/run_min_regression.py --validate-only }
    'gui'        { Invoke-Gui }
    default      { Die "未知子命令：$sub（可用：setup/check-data/eval/test/test-release/freeze-baseline/control-inventory/control/report/gui/wevisone）" }
}
exit $LASTEXITCODE
