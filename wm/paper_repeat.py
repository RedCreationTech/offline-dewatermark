"""Remove repeated neutral oblique overlays on bright paper from one image.

Whole-word visual matches confirm OCR candidates; all matching copies are used.
Only confirmed strokes on paper are filled. Dark print, coloured ink and table
rules are preserved rather than handed to a generative repair model.
"""
from __future__ import annotations
import warnings
import cv2
import numpy as np
from . import ocr, repeated
from .contract import MaskLayer, MaskSource


def table_rules(img):
    gray = img.astype(np.float32).mean(2)
    length = max(25, min(100, min(gray.shape)//12))
    ink = (gray < .65*float(np.percentile(gray, 97))).astype(np.uint8)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    # A dark watermark block is not a table rule. Keep only thin runs;
    # otherwise an invertible dark overlay would be copied back into the result.
    horizontal &= 1-cv2.morphologyEx(horizontal,cv2.MORPH_OPEN,np.ones((5,1),np.uint8))
    vertical &= 1-cv2.morphologyEx(vertical,cv2.MORPH_OPEN,np.ones((1,5),np.uint8))
    return cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8)) > 0


def estimate(img):
    gray = img.astype(np.float32).mean(2)
    if np.mean(gray > 235) < .70:
        return None
    observations = ocr.read_lines(img)
    img_lines = observations or []  # 原图 OCR 行留底，后面直立正文判别复用，不重复调用
    angles = []
    upright_body = False
    for box, text, confidence in observations or []:
        height, length, tilt = ocr._geometry(box)
        upright_body |= confidence>=.65 and len(text)>=4 and tilt<10
        v = box[1]-box[0]
        angle = float(np.degrees(np.arctan2(v[1], v[0])))
        if confidence >= .65 and len(text) >= 5 and length > 3*height and 15 < abs(angle) < 60:
            angles.append(angle)
    if not angles:
        return None
    angle = float(np.median(angles))
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rw = int(abs(matrix[0, 0])*w+abs(matrix[0, 1])*h)
    rh = int(abs(matrix[0, 0])*h+abs(matrix[0, 1])*w)
    matrix[:, 2] += [(rw-w)/2, (rh-h)/2]
    rotated = cv2.warpAffine(img, matrix, (rw, rh), borderValue=(255, 255, 255))
    valid = cv2.warpAffine(np.ones((h, w), np.uint8), matrix, (rw, rh), flags=cv2.INTER_NEAREST)>0
    # Bring faint oblique lettering forward while suppressing dark body ink.
    g = rotated.astype(np.float32).mean(2)
    enhanced = 255-np.clip((255-g)*4,0,255)
    enhanced[g < 140] = 255
    observations = list(ocr.read_lines(rotated) or [])
    intact = sum(conf>=.65 and len(text)>=5 and ocr._geometry(box)[2]<6
                 and ocr._geometry(box)[1]>4*ocr._geometry(box)[0]
                 for box,text,conf in observations)
    sparse = intact < 4
    observations.extend(ocr.read_lines(np.repeat(enhanced.astype(np.uint8)[:,:,None],3,axis=2)) or [])
    seeds = []
    seed_texts = []
    for box, text, confidence in observations:
        height, length, tilt = ocr._geometry(box)
        if confidence >= (.45 if sparse else .65) and len(text) >= (3 if sparse else 5) and tilt < 6 and length > (2.0 if sparse else 4)*height:
            seeds.append((box, height, length))
            seed_texts.append(text)
    if not seeds:
        return None
    g = rotated.astype(np.float32).mean(2)
    size = max(7, int(np.median([s[1] for s in seeds])*.6)) | 1
    # Dark body type used to dominate the match score on dense paragraphs.
    # Estimate the repeated letter tone from OCR candidates and suppress darker
    # foreground for localization only; source pixels are never preprocessed.
    keepout = ocr.text_zones(img)
    rotated_keepout = np.zeros(g.shape,bool) if keepout is None else cv2.warpAffine(
        keepout.astype(np.uint8),matrix,(rw,rh),flags=cv2.INTER_NEAREST)>0
    tones=[]
    neutral=rotated.max(2).astype(float)-rotated.min(2)<12
    for box,_,_ in seeds:
        lo=np.maximum(np.floor(box.min(0)).astype(int),0)
        hi=np.minimum(np.ceil(box.max(0)).astype(int),[rw,rh])
        y,x=slice(lo[1],hi[1]),slice(lo[0],hi[0])
        values=g[y,x]
        selected=(values>50)&(values<235)&~rotated_keepout[y,x]&neutral[y,x]
        tones.append(values[selected].ravel())
    if sum(len(v) for v in tones)<50:
        return None
    tone=float(np.percentile(np.concatenate(tones),10))
    feature = np.minimum(255-g, 1.5*(255-tone)).astype(np.float32)
    feature[g < (.94 if sparse else .7)*tone] = 0
    if tone < 140:
        feature = np.minimum(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((size,size),np.uint8)),150)
    options = []
    typical = float(np.median([s[2] for s in seeds]))
    for (box, height, length), token in zip(seeds,seed_texts):
        if not sparse and not .75*typical < length < 1.75*typical:
            continue
        x0,y0 = np.maximum(np.floor(box.min(0)).astype(int)-4, 0)
        x1,y1 = np.minimum(np.ceil(box.max(0)).astype(int)+4, [rw, rh])
        if y1-y0 < 8 or x1-x0 < 40:
            continue
        visible=valid[y0:y1,x0:x1]
        if visible.mean()<(.70 if sparse else .95) and not (sparse and np.count_nonzero(visible.mean(0)>.98)>=40):
            continue
        patch = feature[y0:y1, x0:x1]
        if patch.std() < 4:
            continue
        # Prefer a complete word away from underlying black text. OCR position
        # alone must never authorize erasing all pixels in its rectangle.
        black = float(np.mean(g[y0:y1, x0:x1] < 45))
        if not upright_body and black > .04:
            continue
        options.append((black, -length, x0, y0, x1, y1, token))
        # A boundary copy may expose a complete suffix while clipping its
        # prefix. Match the fully visible columns separately, without inventing
        # pixels outside the source page.
        if sparse and valid[y0:y1,x0:x1].mean()<.95:
            columns=np.flatnonzero(valid[y0:y1,x0:x1].mean(0)>.98)
            if columns.size>=40:
                lo,hi=x0+int(columns.min()),x0+int(columns.max())+1
                options.append((black-.001,-float(hi-lo),lo,y0,hi,y1,token+" visible"))
    chosen = None
    alternatives = []
    choices=[]
    seen=set()
    for option in sorted(options):
        token=''.join(c.lower() for c in option[-1] if c.isalnum())
        if sparse and token in seen:
            continue
        choices.append(option)
        seen.add(token)
    for _, _, x0,y0,x1,y1,_ in choices[:8]:
        patch = feature[y0:y1, x0:x1]
        th, tw = patch.shape
        scores = cv2.matchTemplate(feature, patch, cv2.TM_CCOEFF_NORMED)
        peaks = repeated._peaks(scores, patch.shape, .55, limit=256)
        if len(peaks) < 2:
            continue
        rank = len(peaks)*tw
        alternatives.append((rank,th,tw,peaks))
        if chosen is None or rank > chosen[0]:
            chosen = (rank, th, tw, peaks)
    if chosen is None:
        return None
    _, th, tw, peaks = chosen
    patches, luminance = [], []
    for x,y,_ in peaks:
        seen = valid[y:y+th,x:x+tw]
        patches.append(np.where(seen, feature[y:y+th,x:x+tw], np.nan))
        luminance.append(np.where(seen, g[y:y+th,x:x+tw], np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        recurring = np.nan_to_num(np.nanpercentile(patches, 25 if tone < 140 and len(peaks)>=3 else 60, axis=0))
        light = np.nan_to_num(np.nanpercentile(luminance, 90, axis=0), nan=255)
    threshold = max(7., .18*float(np.percentile(recurring, 98)))
    template = (recurring > threshold).astype(np.uint8)
    if template.sum() < 50 or not .02 < template.mean() < .5:
        return None
    field = np.zeros((rh+2*th, rw+2*tw), np.float32)
    valid = np.pad(valid, ((th,th),(tw,tw)))
    feature = np.pad(feature, ((th,th),(tw,tw)))
    peaks = [(x+tw,y+th,s) for x,y,s in peaks]
    matrix[:,2] += [tw,th]
    partial = repeated._partial_matches(feature, valid, template, peaks)
    for x,y,_ in [*peaks,*partial]:
        np.maximum(field[y:y+th,x:x+tw], template, out=field[y:y+th,x:x+tw])
    # OCR can split a sparse watermark into a Chinese prefix and an ASCII
    # suffix. Keep separately confirmed word parts instead of erasing only the
    # most frequently matched part. Dense grids already supply enough samples.
    if sparse:
        extra=np.zeros((rh,rw),np.float32)
        for _,eh,ew,ep in sorted(alternatives,key=lambda a:a[0],reverse=True)[:6]:
            if sum(v>.65 for _,_,v in ep)<2:
                continue
            stack=[np.where(valid[y+th:y+th+eh,x+tw:x+tw+ew],
                     feature[y+th:y+th+eh,x+tw:x+tw+ew],np.nan) for x,y,_ in ep]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                recurring_part=np.nan_to_num(np.nanpercentile(stack,60,axis=0))
            part=(recurring_part>max(7.,.18*float(np.percentile(recurring_part,98)))).astype(np.float32)
            if not .02<part.mean()<.5:
                continue
            for x,y,_ in ep:
                np.maximum(extra[y:y+eh,x:x+ew],part,out=extra[y:y+eh,x:x+ew])
        np.maximum(field[th:th+rh,tw:tw+rw],extra,out=field[th:th+rh,tw:tw+rw])
    keepout = ocr.text_zones(img)
    # 直立正文行 ≥2 才把 keepout 接进反解（有正文页字上水印靠它保护）；
    # 白底页的水印行会被 OCR 误当正文（tilt≤15° 的口径太宽），接进去会把
    # 晕圈 alpha 归零、召回实测掉 0.6pp——无直立正文时传 None，由反解内部
    # 按原逻辑自取（与基线行为一致）。
    _upright = 0
    for _box, _t, _c in img_lines:
        _h, _l, _tilt = ocr._geometry(_box)
        if _tilt <= 2.5 and _l >= 3 * _h:
            _upright += 1
    protected = None if keepout is None else cv2.warpAffine(keepout.astype(np.uint8), matrix,
                    (valid.shape[1],valid.shape[0]),flags=cv2.INTER_NEAREST)>0
    for x,y,tx,ty,patch in repeated._edge_fragments(feature, valid, template, peaks, protected,
                                                     band_px=max(120, th * 2)):
        ph,pw=patch.shape
        np.maximum(field[y:y+ph,x:x+pw], patch, out=field[y:y+ph,x:x+pw])
    # 行格外推补章（2026-11-25）：整行模板在页边实例上 NCC 最高仅 ~0.42（页外
    # 白填充稀释相关），实测边缘漏选 99.9% 是"该处无章"；字区实例同样被 NCC
    # 漏。峰已验证行存在与实例 x 位置：对每条有真实峰的行，沿 x 按该行峰间距
    # 双向外推到页边（中间有空档也补齐），盖同款已验证模板。只外推有真实峰的
    # 行，不造新样式；field*=valid 兜底页界裁剪。
    # 补章盖进独立场 field_fill：region 占比闸门装不下时整场撤掉补章回退到
    # 无补章行为（小字密页笔画周长大，补章+膨胀会顶破 .15——白底 s010 实测
    # 老路因此整场退到更差路径）；装得下就保留（文档底 s001 边缘修复靠它）。
    row_groups = []
    for px, py, _s in sorted(peaks, key=lambda p: p[1]):
        for grp in row_groups:
            if abs(float(np.median([gy for _, gy, _ in grp])) - py) < th * 1.5:
                grp.append((px, py, _s))
                break
        else:
            row_groups.append([(px, py, _s)])
    field_fill = np.zeros(valid.shape, np.float32)
    fill_stamps = []

    def _row_stamps(y_row, xs):
        xs = sorted(xs)
        diffs = np.diff(xs)
        step = int(np.median(diffs)) if len(diffs) and np.median(diffs) >= tw * .8 \
            else int(tw * 1.05)
        x = xs[0]
        while x + tw > 0:
            x -= step
        while x < valid.shape[1]:
            if x + tw > 0:
                fill_stamps.append((x, y_row))
            x += step

    centers = sorted(float(np.median([gy for _, gy, _ in grp])) for grp in row_groups)
    for grp in row_groups:
        _row_stamps(int(round(float(np.median([gy for _, gy, _ in grp])))),
                    [int(x) for x, _, _ in grp])
    # 行外推：整行实例全被页边裁掉时该行没有峰、上面不外推就整行漏。
    # 行距取已验证行中心差值中位（要求 ≥3 行且间距一致），只外推一行、只在
    # 已有章跨度两端各延一档（不铺整行——region 占比闸门 .15 装不下），
    # 专补页边行带。空白带上 measured≈0 不会误修（验证把关）。
    row_diffs = np.diff(centers)
    solid = row_diffs[row_diffs >= th * 2]
    if len(centers) >= 3 and len(solid) >= 2:
        y_period = float(np.median(solid))
        if np.percentile(solid, 75) - np.percentile(solid, 25) < th:
            xs_all = sorted(int(x) for x, _, _ in peaks)
            diffs_all = np.diff(xs_all)
            step_g = int(np.median(diffs_all)) if len(diffs_all) \
                and np.median(diffs_all) >= tw * .8 else int(tw * 1.05)
            span_lo, span_hi = xs_all[0] - step_g, xs_all[-1] + tw + step_g
            for y_extra in (centers[0] - y_period, centers[-1] + y_period):
                y_row = int(round(y_extra))
                if 0 <= y_row and y_row + th <= field.shape[0]:
                    x = span_lo
                    while x < span_hi:
                        if x >= 0 and x + tw <= field.shape[1]:
                            fill_stamps.append((x, y_row))
                        x += step_g
    for x, y in fill_stamps:
        if x >= 0 and 0 <= y and y + th <= field_fill.shape[0] and x + tw <= field_fill.shape[1]:
            np.maximum(field_fill[y:y+th, x:x+tw], template, out=field_fill[y:y+th, x:x+tw])
    field *= valid
    field_fill *= valid
    coverage_real = cv2.GaussianBlur(cv2.warpAffine(field, cv2.invertAffineTransform(matrix), (w,h)), (0, 0), 1.2)
    coverage = cv2.GaussianBlur(cv2.warpAffine(np.maximum(field, field_fill),
                                               cv2.invertAffineTransform(matrix), (w,h)), (0, 0), 1.2)
    # Soft coverage keeps antialiased stroke edges inside the invertible fit:
    # a hard binary template zeroes alpha exactly where the veil feathers out,
    # which shows up as residual watermark at the glyph rims. 印章模板只抓住笔画
    # 深色核心，真实抗锯齿晕圈在模板外约 1–2px；把选区再外扩 1px（总计约
    # 3px），让纸面直填覆盖整条晕圈，避免清完内部后留下空心轮廓。
    # 页边裁切实例的硬边晕圈可达 4–6px（实测边缘漏选 46% 在章旁 6px 内），
    # 但全页加宽膨胀会顶破 region 占比闸门 .15（晕圈环带面积随笔画周长放大）；
    # 只对页边 80px 带用 5px 膨胀，内部维持 3px。
    _band = np.zeros(coverage.shape, bool)
    _band[:80] = True; _band[-80:] = True; _band[:, :80] = True; _band[:, -80:] = True

    def _region_of(cov, wide_band):
        c = (cov > .03).astype(np.uint8)
        r = cv2.dilate(c, np.ones((7,7), np.uint8)) > 0
        if wide_band:
            r |= (cv2.dilate(c, np.ones((11,11), np.uint8)) > 0) & _band
        return r
    # 预算感知：小字密页笔画周长大，补章+页边加宽膨胀会把占比顶过 .15 闸门
    # （白底 s010 实测老路整场退场、落到更差路径）。先撤页边加宽，再整场撤
    # 补章，都装不下才拒——闸门是硬约束，补章是增益，增益让位于约束。
    region = _region_of(coverage, True)
    if not .0001 < region.mean() < .15:
        region = _region_of(coverage, False)
    if not .0001 < region.mean() < .15:
        coverage = coverage_real
        region = _region_of(coverage, False)
    if not .0001 < region.mean() < .15:
        return None
    samples = img[region & (gray>50)]
    if len(samples) < 50:
        return None
    # Prefer an invertible alpha-compositing fit over paper-only cleanup:
    # cleaning just the paper leaves watermark strokes where they cross body
    # text, which is exactly what the recognition condition charges against.
    # The repeated instances supply the covered-paper profile the fit needs.
    # 被盖纸面剖面：只盖真实峰（补章是外推，不拿它估纸面亮度），供反解用。
    light_field = np.full(valid.shape, 255, np.float32)
    for x, y, _ in peaks:
        light_field[y:y+th, x:x+tw] = np.minimum(
            light_field[y:y+th, x:x+tw], np.where(template > 0, light, 255))
    profile = cv2.warpAffine(light_field, cv2.invertAffineTransform(matrix), (w, h),
                             borderValue=255)
    fitted = repeated._document_overlay(img, region, profile,
                                        float(np.median([p[2] for p in peaks])),
                                        allow_broad_ink=True, coverage_hint=coverage,
                                        keepout_hint=keepout if _upright >= 2 else None)
    if fitted is not None:
        fitted.method = 'paper_repeat_document_overlay'
        return fitted
    # This path handles neutral lettering only. Coloured stamps have their own
    # measured matting path and must retain it.
    eligible = region & (gray>50) & (gray<245)
    if keepout is not None and np.count_nonzero(eligible & ~keepout)>50:
        eligible &= ~keepout
    bins = np.arange(0,258,2)
    histogram,_ = np.histogram(gray[eligible],bins=bins)
    if histogram.max()<20:
        return None
    level = float(bins[histogram.argmax()]+1)
    core = eligible & (np.abs(gray-level)<3)
    color = np.median(img[core],axis=0).astype(np.float32)
    if color.max()-color.min() > 12:
        return None
    paper = np.percentile(img.reshape(-1,3),97,axis=0).astype(np.float32)
    return MaskLayer(alpha=region.astype(np.float32),source=MaskSource.PRIOR,
        confidence=float(np.median([p[2] for p in peaks])),
        veil_strength=np.zeros(3,np.float32),veil_offset=color,
        paper=np.broadcast_to(paper,img.shape).copy(),footprint=region,
        method='single_repeated_paper',repair_mode='paper_repeat_cleanup',text_keepout=keepout)


def restore(img, mask):
    gray = img.astype(np.float32).mean(2)
    covered = float(mask.veil_offset.mean())
    # Real dark strokes are darker than the independently repeated paper glyph.
    # Include their antialiasing rim and coloured links/seals in the keep-out.
    ink = gray < max(35., .72*covered)
    if mask.text_keepout is not None:
        ink |= mask.text_keepout & (gray < .94*covered)
    ink |= img.max(2).astype(float)-img.min(2) > 18
    protected = cv2.dilate(ink.astype(np.uint8), np.ones((3,3),np.uint8))>0
    protected |= table_rules(img)
    region = mask.footprint & ~protected
    output = img.copy()
    output[region] = np.clip(mask.paper[region],0,255).astype(np.uint8)
    return output
