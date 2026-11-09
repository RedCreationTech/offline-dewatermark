"""生成"原理图解"三张图: 用 inc11 真实样本实际计算每一步中间产物。

图1 色带的形成(关补全)  图2 补全的工序(开补全)  图3 信噪比死锁(为什么找不到)
所有面板均为真实样本的真实计算结果, 无示意伪造。
产物: out/诊断_2026-11-02_inc11笔画补全选区丢失/原理*.png
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

font_manager.fontManager.addfont('/System/Library/Fonts/Hiragino Sans GB.ttc')
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

from wm import ocr, masks
from wm.observed_text import _stroke_template, _positive_ink_view
from wm.contract import Thresholds

ROOT = 'data/watermark_training_increment_v1_1/images/inc11_food_dish__tiled_irregular/'
OUT = 'out/诊断_2026-11-02_inc11笔画补全选区丢失/'

img = np.array(Image.open(ROOT+'watermarked.png').convert('RGB'))
gt = np.array(Image.open(ROOT+'mask.png').convert('L')) > 127
clean = np.array(Image.open('data/watermark_training_increment_v1_1/bases/food_dish.png').convert('RGB'))
if clean.shape != img.shape:
    clean = cv2.resize(clean, (img.shape[1], img.shape[0]))
clean_gray = clean.mean(2)

h, w = img.shape[:2]
rgb = img.astype(np.float32)


def opponent_channel(c):
    op = 128 + rgb[..., c] - (rgb[..., (c+1) % 3]+rgb[..., (c+2) % 3])/2
    return np.uint8(np.clip(op, 0, 255))


def tophat(op):
    return cv2.morphologyEx(op, cv2.MORPH_TOPHAT, np.ones((17, 17), np.uint8)).astype(np.float32)


# ---------- 共用: 通道选择与候选行(与 observed_text 同规则) ----------
lines_raw = ocr.read_lines(img) or []
candidates = []
for box, text, score in lines_raw:
    height, length, tilt = ocr._geometry(box)
    if score >= .65 and len(text.strip()) >= 4 and .025*min(h, w) <= height <= .10*min(h, w) \
            and length > 2.5*height and 15 < tilt < 55:
        candidates.append((box, float(height), float(tilt)))
height = float(np.median([a[1] for a in candidates]))
tilt = float(np.median([a[2] for a in candidates]))
candidates = [a for a in candidates if .65*height < a[1] < 1.5*height and abs(a[2]-tilt) < 6]

best = None
for ch in range(3):
    op = opponent_channel(ch)
    feat = tophat(op)
    view = np.uint8(255-np.clip(feat*12, 0, 255))
    observations = []
    for box, text, score in ocr.read_lines(np.repeat(view[..., None], 3, axis=2)) or []:
        bh, bl, bt = ocr._geometry(box)
        if score >= .65 and len(text.strip()) >= 3 and .6*height < bh < 1.6*height \
                and bl > 2*bh and abs(bt-tilt) < 7:
            observations.append((box, float(bh), float(score)))
    if len(observations) >= 3:
        centres = np.array([a[0].mean(0) for a in observations])
        if np.ptp(centres[:, 0]) >= 3*height and np.ptp(centres[:, 1]) >= 2*height:
            if not _positive_ink_view(feat, observations):
                continue
            if best is None or len(observations) > best[0]:
                best = (len(observations), ch, op, feat, view, observations)
n_obs, CH, op_best, feat_best, view_best, observations = best

m_off = masks.estimate_single_page(img, Thresholds(watermark_recall_priority=False), close_px=11, use_ocr=True)
band = m_off.binary(0.08) > 0

# 亮底/纹理底的水印实例: 从真值连通域里选"最亮实例"和"最暗实例"
n_lab, labels = cv2.connectedComponents(gt.astype(np.uint8), 8)
best_b = worst_b = None
for lab in range(1, n_lab):
    ys, xs = np.where(labels == lab)
    if len(xs) < 200:
        continue
    b = float(clean_gray[ys.mean().astype(int), xs.mean().astype(int)])
    cand = (b, int(ys.mean()), int(xs.mean()))
    if best_b is None or b > best_b[0]:
        best_b = cand
    if worst_b is None or b < worst_b[0]:
        worst_b = cand
_, py, px = best_b
_, fy, fx = worst_b
plate_win = (slice(max(0, py-30), min(h, py+30)), slice(max(0, px-80), min(w, px+80)))
food_win = (slice(max(0, fy-30), min(h, fy+30)), slice(max(0, fx-80), min(w, fx+80)))

CROP = (slice(360, 680), slice(60, 1100))


def crop(a):
    return a[CROP]


def panel(ax, image, title, cmap=None, vmin=None, vmax=None):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13)
    ax.axis('off')


# ================= 图1: 色带的形成 =================
domains = np.zeros((h, w), np.uint8)
for box, _, _ in observations:
    cv2.fillConvexPoly(domains, np.rint(box).astype(np.int32), 1)
for box, text, score in ocr.read_lines(np.repeat(view_best[..., None], 3, axis=2)) or []:
    bh, bl, bt = ocr._geometry(box)
    if score < .60 or len(text.strip()) < 3 or bt > 8 \
            or not .55*height < bh < 1.65*height or bl < 2*bh:
        continue
    cv2.fillConvexPoly(domains, np.rint(box).astype(np.int32), 1)
domains_d = cv2.dilate(domains, np.ones((5, 5), np.uint8)) > 0
weak_resp = domains_d & (feat_best > 2)

fig, axes = plt.subplots(2, 3, figsize=(19, 8.6))
fig.subplots_adjust(hspace=0.18, wspace=0.05)

panel(axes[0][0], crop(img), '① 原图（真实照片，局部）')
panel(axes[0][1], crop(op_best), f'② 对立通道{CH}：突出与水印同色的部分', 'gray')
panel(axes[0][2], crop(view_best), '③ tophat 增强：水印笔画变成清楚的暗字', 'gray')

ax = axes[1][0]
vis = img.copy()
for box, _, _ in observations:
    cv2.polylines(vis, [np.rint(box).astype(np.int32)], True, (0, 60, 255), 3)
ax.imshow(vis)
ax.set_title(f'④ 增强图上 OCR：{len(observations)} 行被确认是水印（蓝框）', fontsize=13)
ax.axis('off')

# ⑤ 弱响应 vs 真实笔画: 盘上 zoom, 黑底
zy = int(np.clip(plate_win[0].start, 0, h-320))
zx = int(np.clip(plate_win[1].start - 60, 0, w-320))
ZW = 320
zoom = np.zeros((ZW, ZW, 3), np.uint8)
sub_wr = weak_resp[zy:zy+ZW, zx:zx+ZW]
sub_gt = gt[zy:zy+ZW, zx:zx+ZW]
zoom[sub_wr] = [255, 255, 255]
zoom[sub_gt] = [0, 200, 0]
axes[1][1].imshow(zoom)
axes[1][1].set_title('⑤ 行框内的弱响应（白点）vs 真实笔画（绿）\n白点稀疏且对不上字形 = 桥接前的原料很少', fontsize=13)
axes[1][1].axis('off')

vis = crop(img).copy()
ov = np.zeros_like(vis)
ov[..., 0] = (crop(band)*180).astype(np.uint8)
ov[..., 1] = (crop(gt)*180).astype(np.uint8)
axes[1][2].imshow(cv2.addWeighted(vis, 0.55, ov, 0.45, 0))
axes[1][2].set_title('⑥ 膨胀连片成色带（红=最终选区 绿=真实笔画）', fontsize=13)
axes[1][2].axis('off')

fig.savefig(OUT+'原理1_色带的形成.png', dpi=110, bbox_inches='tight')
plt.close(fig)

# ================= 图2: 补全的工序 =================
vectors = [a[0][1]-a[0][0] for a in candidates]
angle = float(np.median([np.degrees(np.arctan2(v[1], v[0])) for v in vectors]))
matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
rw = int(abs(matrix[0, 0])*w+abs(matrix[0, 1])*h)
rh = int(abs(matrix[0, 0])*h+abs(matrix[0, 1])*w)
matrix[:, 2] += [(rw-w)/2, (rh-h)/2]
field_d = cv2.warpAffine(feat_best, matrix, (rw, rh))
signal = np.clip(field_d/8, 0, 1).astype(np.float32)
op_best_d = cv2.warpAffine(op_best, matrix, (rw, rh))
size = min(31, max(9, int(.45*height) | 1))
both = cv2.GaussianBlur(np.abs(op_best.astype(np.float32)-cv2.medianBlur(op_best, size)), (0, 0), .5)
matching_d = np.clip(cv2.warpAffine(both, matrix, (rw, rh))/8, 0, 1).astype(np.float32)

box0 = observations[0][0]
b0 = cv2.transform(box0[None].astype(np.float32), matrix)[0]
x0, y0 = np.maximum(np.floor(b0.min(0)).astype(int)-3, 0)
x1, y1 = np.minimum(np.ceil(b0.max(0)).astype(int)+4, [rw, rh])
pw = min(x1-x0, max(24, int(1.6*height)))
ph = y1-y0
patch = signal[y0:y0+ph, x0:x0+pw]
scores = cv2.matchTemplate(signal, patch, cv2.TM_CCOEFF_NORMED)
work = scores.copy()
found = []
for _ in range(60):
    _, value, _, (x, y) = cv2.minMaxLoc(work)
    if not np.isfinite(value) or value < .62:
        break
    found.append((x, y, float(value)))
    work[max(0, y-ph//2):y+ph//2+1, max(0, x-pw//2):x+pw//2+1] = -1
stack = np.stack([field_d[y:y+ph, x:x+pw] for x, y, _ in found[:5]])
typical = np.percentile(stack, 50, axis=0)
stencil = _stroke_template(typical)

# 实际渲染(与 wm/observed_text.py 相同): 转正坐标系里画 stencil, 再转回
result_d = np.zeros((rh, rw), np.uint8)
for x, y, v in found:
    st = stencil & (matching_d[y:y+ph, x:x+pw] > .12)
    result_d[y:y+ph, x:x+pw] |= st.astype(np.uint8)
inv = cv2.invertAffineTransform(matrix)
strokes_back = cv2.warpAffine(result_d, inv, (w, h), flags=cv2.INTER_NEAREST) > 0

fig, axes = plt.subplots(1, 4, figsize=(21, 5.8))
fig.subplots_adjust(wspace=0.06)

vis = img.copy()
ovb = np.zeros_like(vis); ovb[..., 2] = (band*130).astype(np.uint8)
axes[0].imshow(cv2.addWeighted(vis, 0.75, ovb, 0.25, 0))
axes[0].set_title('① 输入：色带作为"去哪找"的范围提示', fontsize=13)
axes[0].axis('off')

axes[1].imshow(patch, cmap='gray', vmin=0, vmax=1)
axes[1].set_title('② 从确认行切一块样板\n（几处实例逐像素取中位数=去噪）', fontsize=13)
axes[1].axis('off')

hm = np.clip((scores-0.2)/0.6, 0, 1)
heat = cv2.applyColorMap(np.uint8(hm[:rh-ph, :rw-pw]*255), cv2.COLORMAP_JET)
heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
heat = cv2.resize(heat, (640, int(640*(rh-ph)/(rw-pw))), interpolation=cv2.INTER_NEAREST)
for x, y, v in found:
    cv2.circle(heat, (int(x*640/(rw-pw)), int(y*640/(rw-pw))), 4, (255, 255, 255), -1)
axes[2].imshow(heat, vmin=0, vmax=1)
axes[2].set_title('③ 全图匹配热力图（蓝=不像 红=最像，白点=过0.62线）\n整图只有这几十个白点', fontsize=13)
axes[2].axis('off')

vis2 = img.copy()
ovo = np.zeros_like(vis2)
ovo[..., 1] = (strokes_back*255).astype(np.uint8)
ovo[..., 0] = (strokes_back*80).astype(np.uint8)
axes[3].imshow(cv2.addWeighted(vis2, 0.5, ovo, 0.5, 0))
axes[3].set_title(f'④ 每个及格位置渲染验证过的笔画（{len(found)}处，亮绿）\n比色带精准，但纹理区大片空白=找不到', fontsize=13)
axes[3].axis('off')

fig.savefig(OUT+'原理2_补全的工序.png', dpi=110, bbox_inches='tight')
plt.close(fig)

# ================= 图3: 两种背景 + 实测分数 =================
diff = np.linalg.norm(img.astype(np.float32)-clean.astype(np.float32), axis=2)

fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
fig.subplots_adjust(wspace=0.1)
axes[0].imshow(img[plate_win])
axes[0].set_title(f'亮底实例（干净底亮度{best_b[0]:.0f}，原样）', fontsize=13)
axes[0].axis('off')
axes[1].imshow(img[food_win])
axes[1].set_title(f'暗底实例（干净底亮度{worst_b[0]:.0f}，原样）', fontsize=13)
axes[1].axis('off')
amp = np.clip(diff*8, 0, 255).astype(np.uint8)
panel(axes[2], cv2.resize(crop(amp), (1040, 320)), '水印造成的实际像素变化放大8倍\n每一笔都真实存在、强度一致——包括菜上', 'hot')
fig.savefig(OUT+'原理3_两种背景对比.png', dpi=110, bbox_inches='tight')
plt.close(fig)

# 实测: 每个真值字形实例, 在其邻域内模板能拿到的最高分
# (包含被漏掉的实例——漏掉的实例才是要证明的对象)
n_lab, labels = cv2.connectedComponents(gt.astype(np.uint8), 8)
pts_b, pts_s = [], []
for lab in range(1, n_lab):
    ys, xs = np.where(labels == lab)
    if len(xs) < 30:          # 太小的是笔画碎片, 跳过
        continue
    cx, cy = int(xs.mean()), int(ys.mean())
    c = cv2.transform(np.array([[[cx, cy]]], np.float32), matrix)[0, 0]
    xi, yi = int(c[0]-pw/2), int(c[1]-ph/2)
    R = 70                    # 在实例附近 ±70px 找模板能拿到的最高分
    xlo, xhi = max(0, xi-R), min(scores.shape[1], xi+pw+R)
    ylo, yhi = max(0, yi-R), min(scores.shape[0], yi+ph+R)
    if xhi <= xlo or yhi <= ylo:
        continue
    local = scores[ylo:yhi, xlo:xhi]
    pts_s.append(float(np.nanmax(local)))
    pts_b.append(float(clean_gray[cy, cx]))
pts_b, pts_s = np.array(pts_b), np.array(pts_s)
n_pass = (pts_s >= .62)
print(f'真值实例 {len(pts_s)} 个: 及格 {n_pass.sum()} (亮底 {(n_pass & (pts_b > 140)).sum()}/'
      f'{(pts_b > 140).sum()}), 不及格 {(~n_pass).sum()} (暗底 {(~n_pass & (pts_b <= 140)).sum()}/{(~(pts_b > 140)).sum()})')

fig, ax = plt.subplots(figsize=(9.5, 5.6))
bright = pts_b > 140
jitter_b = pts_b + np.random.default_rng(7).uniform(-6, 6, len(pts_b))
ax.scatter(jitter_b[bright & n_pass], pts_s[bright & n_pass], c='#2a7', s=42,
           label=f'亮底及格 n={(bright & n_pass).sum()}')
ax.scatter(jitter_b[bright & ~n_pass], pts_s[bright & ~n_pass], c='#8b8', s=42,
           label=f'亮底不及格 n={(bright & ~n_pass).sum()}')
ax.scatter(jitter_b[~bright & n_pass], pts_s[~bright & n_pass], c='#c44', s=42,
           label=f'纹理及格 n={(~bright & n_pass).sum()}')
ax.scatter(jitter_b[~bright & ~n_pass], pts_s[~bright & ~n_pass], facecolors='none',
           edgecolors='#c44', s=60, linewidths=1.6,
           label=f'纹理不及格(丢失) n={(~bright & ~n_pass).sum()}')
ax.axhline(.62, color='k', linestyle='--', linewidth=1.5)
ax.text(10, .635, '及格线 0.62', fontsize=11)
for cond, color in ((bright, '#175c3a'), (~bright, '#7a2020')):
    if cond.any():
        ax.hlines(np.median(pts_s[cond]), 0, 255, colors=color, linestyles=':', linewidth=2)
ax.set_xlabel('该实例在干净底图上的亮度（越亮=背景越平滑）', fontsize=12)
ax.set_ylabel('该实例能拿到的最高匹配分数', fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_title('实测：每个真实水印字形的最高匹配分数\n空心红点=纹理上被丢掉的实例，集中在及格线下方', fontsize=13)
ax.legend(loc='upper left', fontsize=10)
fig.savefig(OUT+'原理3b_匹配分数实测.png', dpi=110, bbox_inches='tight')
plt.close(fig)

print(f'匹配位置数 {len(found)}；亮底中位分 {np.median(pts_s[bright]):.2f}，暗底中位分 {np.median(pts_s[~bright]):.2f}')
print('已生成四张原理图')
