"""图 1 细节系列(1a~1d): 色带形成机制的最细还原, 全部真实计算。

1a 通道数学(一维剖面+单像素运算)  1b 门槛关卡漏斗(逐条真实数字)
1c 面积三叠(框→稀疏响应→膨胀色带)  1d 像素传记(三个像素的完整身世+全量审计)
产物: out/诊断_2026-11-02_inc11笔画补全选区丢失/原理1{a,b,c,d}_*.png
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

font_manager.fontManager.addfont('/System/Library/Fonts/Hiragino Sans GB.ttc')
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

from wm import ocr, masks
from wm.observed_text import _positive_ink_view
from wm.contract import Thresholds

ROOT = 'data/watermark_training_increment_v1_1/images/inc11_food_dish__tiled_irregular/'
OUT = 'out/诊断_2026-11-02_inc11笔画补全选区丢失/'

img = np.array(Image.open(ROOT+'watermarked.png').convert('RGB'))
gt = np.array(Image.open(ROOT+'mask.png').convert('L')) > 127
clean = cv2.resize(np.array(Image.open('data/watermark_training_increment_v1_1/bases/food_dish.png')
                             .convert('RGB')), (1280, 960))
clean_gray = clean.mean(2)

h, w = img.shape[:2]
rgb = img.astype(np.float32)


def opp_of(c, src=None):
    s = src if src is not None else rgb
    return np.uint8(np.clip(128+s[..., c]-(s[..., (c+1) % 3]+s[..., (c+2) % 3])/2, 0, 255))


feat = cv2.morphologyEx(opp_of(0), cv2.MORPH_TOPHAT, np.ones((17, 17), np.uint8)).astype(np.float32)
view = np.uint8(255-np.clip(feat*12, 0, 255))

# ---------- 共用数据 ----------
lines_raw = ocr.read_lines(img) or []
geo = [(box, t, s, ocr._geometry(box)) for box, t, s in lines_raw]


def gate_stats():
    """逐条门槛的真实通过数(候选行筛选, 与 observed_text 同规则)"""
    rows = []
    cur = geo
    rows.append(('原始 OCR 行', len(cur)))
    cur = [g for g in cur if g[2] >= .65 and len(g[1].strip()) >= 4]
    rows.append(('门槛1 置信度≥0.65 且 ≥4字', len(cur)))
    cur = [g for g in cur if .025*min(h, w) <= g[3][0] <= .10*min(h, w)]
    rows.append(('门槛2 字高占图短边 2.5%~10%', len(cur)))
    cur = [g for g in cur if g[3][1] > 2.5*g[3][0]]
    rows.append(('门槛3 行长>2.5倍字高', len(cur)))
    cur = [g for g in cur if 15 < g[3][2] < 55]
    rows.append(('门槛4 倾角 15°~55°', len(cur)))
    return rows, cur


rows_funnel, candidates = gate_stats()
height = float(np.median([g[3][0] for g in candidates]))
tilt = float(np.median([g[3][2] for g in candidates]))
cand2 = [g for g in candidates if .65*height < g[3][0] < 1.5*height and abs(g[3][2]-tilt) < 6]
rows_funnel.append(('门槛5 字号/倾角与全图中位一致', len(cand2)))

# 每通道增强观测的逐层计数
ch_stats = []
choices = []
for ch in range(3):
    op = opp_of(ch)
    ft = cv2.morphologyEx(op, cv2.MORPH_TOPHAT, np.ones((17, 17), np.uint8)).astype(np.float32)
    vw = np.uint8(255-np.clip(ft*12, 0, 255))
    obs_all = ocr.read_lines(np.repeat(vw[..., None], 3, axis=2)) or []
    a = [o for o in obs_all if o[2] >= .65 and len(o[1].strip()) >= 3
         and .6*height < ocr._geometry(o[0])[0] < 1.6*height
         and ocr._geometry(o[0])[1] > 2*ocr._geometry(o[0])[0]
         and abs(ocr._geometry(o[0])[2]-tilt) < 7]
    b = a
    if len(a) >= 3:
        centres = np.array([o[0].mean(0) for o in a])
        if not (np.ptp(centres[:, 0]) >= 3*height and np.ptp(centres[:, 1]) >= 2*height):
            b = []
    c = b
    if b and not _positive_ink_view(ft, b):
        c = []
    ch_stats.append((ch, len(obs_all), len(a), len(b), len(c)))
    if c:
        choices.append((len(c), ch, ft, vw, c))
_, CH, feat_ch, view_ch, observations = max(choices, key=lambda x: x[0])

# 矫正图 OCR（与 observed_text 相同：转正后再读一遍，补充范围）
vectors = [g[0][1]-g[0][0] for g in cand2]
angle = float(np.median([np.degrees(np.arctan2(v[1], v[0])) for v in vectors]))
matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
rw = int(abs(matrix[0, 0])*w+abs(matrix[0, 1])*h)
rh = int(abs(matrix[0, 0])*h+abs(matrix[0, 1])*w)
matrix[:, 2] += [(rw-w)/2, (rh-h)/2]
upright = cv2.warpAffine(view_ch, matrix, (rw, rh))
lines2 = ocr.read_lines(np.repeat(upright[..., None], 3, axis=2)) or []
deskew_ok = [o for o in lines2 if o[2] >= .60 and len(o[1].strip()) >= 3
             and ocr._geometry(o[0])[2] <= 8
             and .55*height < ocr._geometry(o[0])[0] < 1.65*height
             and ocr._geometry(o[0])[1] >= 2*ocr._geometry(o[0])[0]]

# 框/响应/色带三级面积(真实)
domains = np.zeros((h, w), np.uint8)
for box, _, _ in observations:
    cv2.fillConvexPoly(domains, np.rint(box).astype(np.int32), 1)
for box in [o[0] for o in deskew_ok]:
    cv2.fillConvexPoly(domains, np.rint(box).astype(np.int32), 1)
domains = cv2.dilate(domains, np.ones((5, 5), np.uint8)) > 0
fill = domains & (feat_ch > 2)
m_off = masks.estimate_single_page(img, Thresholds(watermark_recall_priority=False), close_px=11, use_ocr=True)
band = m_off.binary(0.08) > 0
ring = band & ~fill

# ============ 图1a 通道数学 ============
# 选亮底上的一个笔画中心（提高亮度门槛确保落在盘上）
ys_p, xs_p = np.where(gt & (clean_gray > 190) & (feat_ch > 8))
cy, cx = int(np.median(ys_p)), int(np.median(xs_p))
ang = np.radians(-tilt)          # 文字方向
direction = np.array([np.cos(ang), np.sin(ang)])
normal = np.array([-direction[1], direction[0]])
TS = np.arange(-48, 49)


def sample(field):
    out = []
    for t in TS:
        p = np.array([cx, cy], np.float64)+t*normal
        x0, y0 = int(np.floor(p[0])), int(np.floor(p[1]))
        dx, dy = p[0]-x0, p[1]-y0
        patch = field[y0:y0+2, x0:x0+2].astype(np.float64)
        out.append((1-dx)*(1-dy)*patch[0, 0]+dx*(1-dy)*patch[0, 1]
                   + (1-dx)*dy*patch[1, 0]+dx*dy*patch[1, 1])
    return np.array(out)


prof_opp = sample(opp_of(CH).astype(np.float64))
prof_feat = sample(feat_ch)

ZW = 120
z = img[cy-ZW//2:cy+ZW//2, cx-ZW//2:cx+ZW//2].copy()
scale = 3
z = cv2.resize(z, (ZW*scale, ZW*scale), interpolation=cv2.INTER_NEAREST)
p0 = (np.array([cx, cy])-normal*48 - [cx-ZW//2, cy-ZW//2])*scale
p1 = (np.array([cx, cy])+normal*48 - [cx-ZW//2, cy-ZW//2])*scale
cv2.line(z, tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int)), (255, 0, 0), 2)

fig = plt.figure(figsize=(17, 5.6))
gs = GridSpec(1, 3, width_ratios=[1.15, 1.5, 1.25], wspace=0.22)
ax = fig.add_subplot(gs[0])
ax.imshow(z)
ax.set_title('红线=取样线（垂直穿过一笔画）\n中心亮底真实笔画，放大3倍', fontsize=12)
ax.axis('off')

ax = fig.add_subplot(gs[1])
ax.plot(TS, prof_opp, color='#888', label='对立通道值（原信号）')
ax.plot(TS, prof_feat, color='#c22', linewidth=2.2, label='tophat 之后（抠出的小鼓包）')
ax.axhline(2, color='k', linestyle='--', linewidth=1)
ax.text(-46, 5, '及格线 field>2', fontsize=10)
ax.axvspan(-6, 6, color='#2a7', alpha=.15)
ax.text(0, ax.get_ylim()[1]*.72, '笔画宽度\n≈12像素', ha='center', fontsize=10)
ax.set_xlabel('沿取样线的位置（像素）', fontsize=11)
ax.set_ylabel('数值', fontsize=11)
ax.legend(fontsize=10, loc='upper right')
ax.set_title('一维剖面：tophat = 原信号 - 局部最大\n笔画=小鼓包，平缓背景被减成≈0', fontsize=12)

ax = fig.add_subplot(gs[2])
ax.axis('off')
r, g, b_ = img[cy, cx]
op_v = float(opp_of(CH)[cy, cx])
ft_v = float(feat_ch[cy, cx])
txt = (f'中心像素的真实数值\n\n'
       f'RGB = ({r}, {g}, {b_})\n'
       f'↓ 对立通道{CH}\n'
       f'128 + {r} - ({g}+{b_})/2 = {op_v:.0f}\n\n'
       f'↓ tophat（减17×17局部最大）\n'
       f'{op_v:.0f} - 局部最大 = {ft_v:.1f}\n\n'
       f'判定：{ft_v:.1f} > 2 ？ {"通过（计入）" if ft_v > 2 else "不通过"}')
ax.text(0.02, 0.97, txt, fontsize=12.5, va='top',
        bbox=dict(boxstyle='round', facecolor='#f4f7f5', edgecolor='#2a7'))
ax.set_title('单像素运算传记（笔画中心像素）', fontsize=12)
fig.savefig(OUT+'原理1a_通道数学.png', dpi=110, bbox_inches='tight')
plt.close(fig)

# ============ 图1b 门槛关卡漏斗 ============
fig, ax = plt.subplots(figsize=(11.5, 7.2))
labels = [r[0] for r in rows_funnel]+[f'增强图通道{ch}（门槛6/7 后）' for ch, _, _, _, c in ch_stats]
vals = [r[1] for r in rows_funnel]+[c for *_, c in ch_stats]
notes = ['']*len(rows_funnel)+[f'原始{all_}→基本门槛{a}→分散度{b}' for _, all_, a, b, c in ch_stats]
y = np.arange(len(labels))[::-1]
colors = ['#2a7']*len(rows_funnel)+['#48b' if c else '#bbb' for *_, c in ch_stats]
ax.barh(y, vals, color=colors, height=.62)
for yi, v, n in zip(y, vals, notes):
    ax.text(v+.3, yi, f'{v}'+(f'   ({n})' if n else ''), va='center', fontsize=11)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11.5)
ax.set_xlim(0, 30)
ax.set_title(f'行确认的关卡漏斗（真实数字）\n'
             f'通道{CH} 胜出：{len(observations)} 处观测；矫正图 OCR 再补 {len(deskew_ok)} 行进范围',
             fontsize=13)
ax.set_xlabel('通过的行/观测数', fontsize=11)
fig.savefig(OUT+'原理1b_门槛关卡.png', dpi=110, bbox_inches='tight')
plt.close(fig)

# ============ 图1c 面积三叠 ============
# 选包含亮底与文字行的窗口
wy, wx = int(np.median([o[0][:, 1].mean() for o in observations[:4]]))-90, 150
WW_H, WW_W = 240, 760
sl = (slice(wy, wy+WW_H), slice(wx, wx+WW_W))


def lay(mask_img, color):
    base = img[sl].copy()
    ov = np.zeros_like(base)
    ov[mask_img[sl]] = color
    return cv2.addWeighted(base, 0.45, ov, 0.55, 0)


fig, axes = plt.subplots(3, 1, figsize=(14.5, 9.2))
fig.subplots_adjust(hspace=0.32)
axes[0].imshow(lay(domains, (80, 160, 255)))
axes[0].set_title(f'① OCR 确认的实心行框（{len(observations)} 行 + 矫正补 {len(deskew_ok)} 行）'
                  f'　面积 {domains.sum():,} 像素', fontsize=13)
axes[1].imshow(lay(fill, (255, 255, 255)))
axes[1].set_title(f'② 框内 tophat>2 的低门槛响应　面积 {fill.sum():,} 像素'
                  f'（仅占框的 {fill[domains].mean()*100:.0f}%——稀疏的笔画+噪声点）', fontsize=13)
axes[2].imshow(lay(band, (255, 60, 60)))
ovg = np.zeros_like(img[sl]); ovg[gt[sl]] = (0, 200, 0)
axes[2].imshow(cv2.addWeighted(axes[2].images[0].get_array().astype(np.uint8)
                               if False else lay(band, (255, 60, 60)), 0.8,
                               np.dstack([ovg[..., 1]]*3).astype(np.uint8)*0, 0, 0)) if False else None
axes[2].imshow(lay(band, (255, 60, 60)))
gtvis = lay(band, (255, 60, 60))
gmask = gt[sl]
blend = gtvis.copy()
blend[gmask] = [0, 220, 0]
axes[2].imshow(blend)
axes[2].set_title(f'③ 7×7 膨胀连片 = 最终色带（红）　面积 {band.sum():,} 像素'
                  f'　其中 {ring.sum()/band.sum()*100:.0f}% 是膨胀这一步造出来的；绿=真实笔画', fontsize=13)
for a_ in axes:
    a_.axis('off')
fig.savefig(OUT+'原理1c_面积三叠.png', dpi=110, bbox_inches='tight')
plt.close(fig)

# ============ 图1d 像素传记 ============
# P: 字间空白、膨胀环进来的亮底像素
dist_fill = cv2.distanceTransform((~fill).astype(np.uint8), cv2.DIST_L2, 5)
cand_P = np.argwhere(ring & (clean_gray > 140) & ~gt & (dist_fill <= 3) & (feat_ch <= 2))
py_, px_ = cand_P[len(cand_P)//2]
# Q: 笔画上的响应像素
cand_Q = np.argwhere(gt & (feat_ch > 2) & (clean_gray > 140))
qy, qx_ = cand_Q[len(cand_Q)//2]
# R: 框内但没被选中（离响应点>3px）
cand_R = np.argwhere(domains & ~band & (clean_gray > 140) & (dist_fill > 3))
ry, rx_ = cand_R[len(cand_R)//2]


def bio(y_, x_):
    r, g, b_ = img[y_, x_]
    op_v = float(opp_of(CH)[y_, x_])
    ft_v = float(feat_ch[y_, x_])
    in_fill = bool(fill[y_, x_])
    in_band = bool(band[y_, x_])
    d = float(dist_fill[y_, x_])
    return (f'RGB=({r},{g},{b_})\n对立{CH}={op_v:.0f}\ntophat={ft_v:.1f}\n'
            f'在框内: {"是" if domains[y_,x_] else "否"}\n'
            f'field>2: {"是→直接计入" if in_fill else "否"}\n'
            f'离响应点: {d:.0f}px\n'
            f'进色带: {"是" if in_band else "否"}'
            + (f'\n身世: 膨胀带入' if in_band and not in_fill else
               (f'\n身世: 响应计入' if in_fill else '')))


# 全量审计
inter = int((band & gt).sum())
fp = band & ~gt
gt_d3 = cv2.dilate(gt.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
near3 = int((fp & gt_d3).sum())
gt_d9 = cv2.dilate(gt.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0
near9 = int((fp & gt_d9).sum())

ZW2 = 64


def zoom(y_, x_):
    z = img[max(0, y_-ZW2//2):y_+ZW2//2, max(0, x_-ZW2//2):x_+ZW2//2].copy()
    z = cv2.resize(z, (ZW2*4, ZW2*4), interpolation=cv2.INTER_NEAREST)
    cv2.circle(z, (ZW2*2, ZW2*2), 26, (255, 0, 0), 3)
    return z


fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2))
fig.subplots_adjust(wspace=0.25, top=0.72)
for a_, (y_, x_, name, sub) in zip(axes, [
        (py_, px_, 'P：字间空白像素', '膨胀带入的 75% 之一'),
        (qy, qx_, 'Q：真实笔画像素', '凭响应计入'),
        (ry, rx_, 'R：框内但落选像素', '离响应太远，膨胀也够不到')]):
    a_.imshow(zoom(y_, x_))
    a_.set_title(f'{name}\n{sub}\n\n{bio(y_, x_)}', fontsize=11, loc='left')
    a_.axis('off')
fig.suptitle('三个像素的完整身世（亮底窗口各 64×64，红圈=主角）', fontsize=13)
fig.savefig(OUT+'原理1d_像素传记.png', dpi=110, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10.5, 5.2))
cats = ['真实笔画', '误选-贴笔画3px内', '误选-3~9px', '误选->9px']
vals_a = [inter, near3, near9-near3, int(fp.sum())-near9]
cols = ['#2a7', '#e90', '#e60', '#c22']
bars = ax.bar(cats, vals_a, color=cols, width=.55)
for b_, v in zip(bars, vals_a):
    ax.text(b_.get_x()+b_.get_width()/2, v+4000, f'{v:,}\n({v/band.sum()*100:.0f}%)', ha='center', fontsize=10.5)
ax.set_title(f'色带 {band.sum():,} 像素的全量审计\n来源：框内响应 {fill.sum()/band.sum()*100:.0f}% + 膨胀环 '
             f'{ring.sum()/band.sum()*100:.0f}%', fontsize=13)
ax.set_ylabel('像素数', fontsize=11)
ax.set_ylim(0, max(vals_a)*1.25)
fig.savefig(OUT+'原理1d2_全量审计.png', dpi=110, bbox_inches='tight')
plt.close(fig)

print('图1系列已生成: 1a 通道数学 / 1b 门槛关卡 / 1c 面积三叠 / 1d 像素传记+全量审计')
print(f'关键数字: 框{domains.sum()} 响应{fill.sum()}({fill[domains].mean()*100:.0f}% of 框) '
      f'色带{band.sum()} 环{ring.sum()}({ring.sum()/band.sum()*100:.0f}%)')
print(f'P=({py_},{px_}) Q=({qy},{qx_}) R=({ry},{rx_})')
