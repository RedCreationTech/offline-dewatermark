"""Locate repeated oblique lettering from observations in ONE input image.

OCR supplies direction and a seed line, not a hard-coded watermark phrase. A
second visual match is required. Registered repeats separate recurring strokes
from changing page content. No clean image, stored mask, or filename is read.
This path targets repeating diagonal text; other styles retain the tonal path.
"""
from __future__ import annotations
import warnings
import cv2
import numpy as np
from . import ocr
from .contract import MaskLayer, MaskSource


def _peaks(scores, shape, minimum, limit=12):
    scores = scores.copy()
    hh, ww = shape
    found = []
    for _ in range(limit):
        _, value, _, (x, y) = cv2.minMaxLoc(scores)
        if value < minimum or not np.isfinite(value):
            break
        found.append((x, y, value))
        scores[max(0,y-hh):y+hh, max(0,x-ww):x+ww] = -1
    return found


def _partial_matches(feature, valid, template, peaks):
    """Only accept clipped copies on rows supported by full visual repeats."""
    th, tw = template.shape
    ordered = np.array(sorted([p[:2] for p in peaks], key=lambda p:p[1]), np.float32)
    step = np.median(np.diff(ordered, axis=0), axis=0)
    period = float(np.linalg.norm(step))
    if period < th*2:
        return []
    normal = step / period
    phase = float(ordered[0] @ normal)
    _, cols = np.nonzero(template)
    if not cols.size:
        return []
    width = max(40, int(th*1.8))
    found = []
    for start in range(int(cols.min()), int(cols.max())-width+2, max(20,width//2)):
        part = template[:,start:start+width].astype(np.float32)
        if part.sum() < 30:
            continue
        score = cv2.matchTemplate(feature,part,cv2.TM_CCOEFF_NORMED)
        for x, y, confidence in _peaks(score,part.shape,.60,20):
            x -= start
            if x<0 or y<0 or x+tw>valid.shape[1] or y+th>valid.shape[0]:
                continue
            ratio = valid[y:y+th,x:x+tw][template>0].mean()
            row = (float(np.array([x,y])@normal)-phase)/period
            if not .02<ratio<.35 or abs(row-round(row))*period>th*.20:
                continue
            if any(np.linalg.norm(np.array([x,y])-p[:2])<th for p in [*peaks,*found]):
                continue
            found.append((x,y,confidence))
    return found


def _fragments(feature, template, peaks):
    """Locate short surviving pieces on the verified repeat rows.

    A watermark can be clipped before a document is rotated, so the fragment
    need not touch the final image border. Return only matched glyph pieces,
    never extrapolate an entire word across unrelated page content.
    """
    if len(peaks)<3:
        return []
    points=np.array(sorted([p[:2] for p in peaks],key=lambda p:p[1]),np.float32)
    step=np.median(np.diff(points,axis=0),axis=0)
    period=float(np.linalg.norm(step))
    th,tw=template.shape
    if period<th*2:
        return []
    normal=step/period
    phase=float(points[0]@normal)
    binary=(feature>30).astype(np.float32)
    width=max(35,int(th*1.2))
    found=[]
    for start in range(0,tw-width,max(16,width//2)):
        piece=template[:,start:start+width]
        yy,xx=np.nonzero(piece)
        if xx.size<40:
            continue
        x0,x1,y0,y1=int(xx.min()),int(xx.max())+1,int(yy.min()),int(yy.max())+1
        patch=piece[y0:y1,x0:x1].astype(np.float32)
        score=cv2.matchTemplate(binary,patch,cv2.TM_CCOEFF_NORMED)
        for x,y,confidence in _peaks(score,patch.shape,.67,20):
            full=np.array([x-start-x0,y-y0])
            row=(float(full@normal)-phase)/period
            if abs(row-round(row))*period>th*.25:
                continue
            if any(np.linalg.norm(full-p[:2])<th for p in peaks):
                continue
            found.append((x,y,start+x0,y0,patch))
    return found


def _edge_fragments(feature, valid, template, peaks, protected=None):
    """Match surviving boundary strokes, without requiring a full glyph/word.

    The missing part is ignored in normalized correlation. Both visual shape
    and the already verified repeat rows must agree. Body-text zones veto the
    match; only matched visible strokes are returned, never a whole border.
    """
    if len(peaks) < 2:
        return []
    th, tw = template.shape
    points = np.array(sorted([p[:2] for p in peaks], key=lambda p: p[1]), np.float32)
    step = np.median(np.diff(points, axis=0), axis=0)
    period = float(np.linalg.norm(step))
    if period < th * 2:
        return []
    normal = step / period
    phase = float(points[0] @ normal)
    interior = cv2.erode(valid.astype(np.uint8), np.ones((th + 1, th + 1), np.uint8)) > 0
    band = valid & ~interior
    binary = (feature > 30).astype(np.uint8)
    candidates = cv2.morphologyEx(binary & band, cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
    pad = th
    padded = np.pad(template, ((pad, pad), (pad, pad))).astype(np.float32)
    found = []
    for x, y, w, h, area in stats[1:]:
        if area < 8 or h > th * 2 or w > th * 3:
            continue
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(valid.shape[1], x + w + 2), min(valid.shape[0], y + h + 2)
        query = binary[y0:y1, x0:x1].astype(np.float32)
        observed = valid[y0:y1, x0:x1].astype(np.uint8)
        count = float((query * observed).sum())
        if count < 8 or query.shape[0] >= padded.shape[0] or query.shape[1] >= padded.shape[1]:
            continue
        if protected is not None:
            overlap = (query * observed * protected[y0:y1, x0:x1]).sum() / count
            if overlap > .1:
                continue
        scores = cv2.matchTemplate(padded, query, cv2.TM_CCORR_NORMED, mask=observed)
        scores[~np.isfinite(scores)] = -1
        for _ in range(30):
            _, confidence, _, (tx, ty) = cv2.minMaxLoc(scores)
            if confidence < .67:
                break
            scores[max(0, ty - 2):ty + 3, max(0, tx - 2):tx + 3] = -1
            origin = np.array([x0 - tx + pad, y0 - ty + pad])
            row = (float(origin @ normal) - phase) / period
            if abs(row - round(row)) * period > th * .20:
                continue
            matched = padded[ty:ty + query.shape[0], tx:tx + query.shape[1]]
            agreement = float((matched * query * observed).sum()) / count
            if agreement < .65:
                continue
            found.append((x0, y0, tx - pad, ty - pad, matched * observed))
            break
    return found


def _document_overlay(img, region, profile, confidence, allow_broad_ink: bool = False,
                      coverage_hint: np.ndarray | None = None,
                      allow_single_pole: bool = False):
    """Fit covered-paper and covered-ink poles from THIS document only.

    The repeated shape supplies alpha at text intersections. Elsewhere white
    paper directly measures attenuation. Refuse the fit without both poles.

    With ``allow_single_pole`` (watermark-only pages: no body ink anywhere),
    the covered-ink pole is unattainable by construction and unnecessary — the
    inverse B=(I-β·t)/(1-β·(1-s)) restores paper identically for any (s,t)
    on the line s·paper+t=covered. Fall back to a per-channel single-pole
    (s=.5, t=covered-.5·paper) so chromatic stamps unmix per channel.
    """
    keepout=ocr.text_zones(img)
    if keepout is None:
        return None
    if not keepout.any():
        # 整页无直立正文墨迹（如整页只有斜向水印）：keepout 为空。仅当调用方
        # 显式允许无交叠单极拟合时才继续——此时 keepout 不再参与极值估计，
        # alpha 由 measured 全区域给出，正文保护由修复侧的深色核心保护兜底。
        if not allow_single_pole:
            return None
        keepout = np.zeros(img.shape[:2], bool)
    gray=img.mean(2)
    paper=float(np.percentile(gray,97))
    estimated = np.clip(coverage_hint, 0, 1) if coverage_hint is not None else None
    edges=np.arange(0,258,2)
    histogram,_=np.histogram(gray[region & ~keepout],bins=edges)
    clean_peak = float(edges[int(np.argmax(histogram))])
    # Estimate the covered-paper pole on template-covered pixels only: the
    # whole region is mostly clean paper, so its near-white antialiasing can
    # otherwise outvote the actual veil pole (measured on layered samples).
    pool_mask = region & ~keepout & (estimated > .5) if estimated is not None \
        else region & ~keepout
    pool = gray[pool_mask]
    if pool.size < 200:
        pool_mask = region & ~keepout
        pool = gray[pool_mask]
    histogram,_=np.histogram(pool,bins=edges)
    histogram[(edges[:-1]<.16*paper)|(edges[:-1]>clean_peak-12)]=0
    if histogram.max()<20:
        return None
    covered=float(edges[np.argmax(histogram)]+1)
    if estimated is None:
        estimated = np.clip((paper-profile)/max(paper-covered,1),0,1)
    # Joint two-pole fit with a histogram peak for the covered-ink level.
    # The covered-ink pole sits between the uncovered body-ink level and the
    # covered-paper level; taking the strongest histogram peak in that open
    # interval avoids both pure-black contamination and antialiasing bias.
    body_pool = gray[keepout & ~region & (gray < .5 * paper)]
    pool = gray[region & keepout & (estimated > .5)]
    transmittance = ink = None
    covered_ink_level = None
    if body_pool.size >= 50 and pool.size >= 50:
        body_ink = float(np.percentile(body_pool, 20))
        hist, _ = np.histogram(pool, bins=np.arange(0, 258, 2))
        lo = int((body_ink + 4) // 2)
        hi = int(covered // 2)
        if hi > lo and hist[lo:hi].max() >= 10:
            covered_ink = float((int(np.argmax(hist[lo:hi])) + lo) * 2 + 1)
            if paper - body_ink > 30:
                s_fit = (covered - covered_ink) / (paper - body_ink)
                t_fit = covered - s_fit * paper
                if .08 < s_fit < .85 and 0 <= t_fit < covered:
                    transmittance, ink = float(s_fit), float(t_fit)
                    covered_ink_level = s_fit * body_ink + t_fit
    single_pole_rgb = None
    if transmittance is None and allow_single_pole and body_pool.size < 50:
        # 无正文墨迹页面：覆盖墨极在构造上不可得也不需要（反解对纸面像素
        # 恒等还原）。逐通道拟合"纸面极+覆盖纸极"，s 取 0.5，t 落在约束线。
        pool_rgb = img[pool_mask].reshape(-1, 3)
        if pool_rgb.shape[0] >= 200:
            paper_rgb = np.percentile(img.reshape(-1, 3), 97, axis=0)
            clean_rgb = np.percentile(img.reshape(-1, 3), 97, axis=0)
            s_rgb = np.full(3, .5, np.float32)
            t_rgb = np.zeros(3, np.float32)
            ok = True
            for c in range(3):
                hist_c, _ = np.histogram(pool_rgb[:, c], bins=edges)
                hist_c[(edges[:-1] < .16 * paper_rgb[c]) | (edges[:-1] > clean_rgb[c] - 12)] = 0
                if hist_c.max() < 20:
                    ok = False
                    break
                t_rgb[c] = float(edges[np.argmax(hist_c)] + 1) - .5 * paper_rgb[c]
            contrast = paper_rgb - (.5 * paper_rgb + t_rgb)
            if ok and float(np.min(contrast)) >= 12. and .08 < float(np.min(s_rgb)) \
                    and float(np.max(.5 * paper_rgb + t_rgb)) < 250.:
                single_pole_rgb = (s_rgb, t_rgb)
    if transmittance is None and single_pole_rgb is None:
        ink_pixels=gray[region & keepout & (estimated>.7) &
                        (gray>.05*paper) & (gray<.65*covered)]
        # The lower tail estimates black glyph cores, rather than the numerous
        # antialiased edges which would incorrectly raise the covered-ink pole.
        percentile = 10
        if ink_pixels.size < 20 and allow_broad_ink:
            broad = gray[region & keepout & (estimated > .5)]
            if broad.size >= 50:
                ink_pixels = broad
                percentile = 5
        if ink_pixels.size >= 20:
            ink = float(np.percentile(ink_pixels, percentile))
            transmittance = (covered - ink) / paper
            if not .08 < transmittance < .85:
                transmittance = ink = None
    if transmittance is None and single_pole_rgb is None:
        return None
    if transmittance is None:
        veil_strength, veil_offset = single_pole_rgb
    else:
        veil_strength = np.full(3, transmittance, np.float32)
        veil_offset = np.full(3, ink, np.float32)
    measured=np.clip((paper-gray)/max(paper-covered,1),0,1)
    # Undo the mild interpolation attenuation of the registered template; never
    # infer underlying glyph edges from their darkness alone. Inside the OCR
    # zones only the eroded high-confidence template core may drive the inverse:
    # antialiasing rims (clean or watermarked) are left untouched so a slightly
    # misregistered template cannot whiten or darken real body strokes.
    alpha=np.where(keepout,
                   np.where(estimated >= .15, measured, 0),
                   measured)
    if not keepout.any() and estimated is not None:
        # 无正文页面：没有需要防误伤的正文，抗锯齿晕圈本身就是待清水印；
        # 反解在 s=.5 单极下对纸面底层恒等饱和到纸面色，选区直接取整片
        # 印章覆盖区，保证掩码覆盖整条晕圈、与整框路径的口径可比。
        alpha = np.where(region, 1., 0.)
    if covered_ink_level is not None and transmittance is not None and transmittance < 0.4:
        # Heavy veils only: pixels darker than any veil-covered text can be
        # are clean body ink and must not enter the noisy inverse.
        alpha = np.where(keepout & (gray < covered_ink_level - 5), 0, alpha)
    alpha=np.where(region,alpha,0).astype(np.float32)
    return MaskLayer(alpha=alpha,source=MaskSource.PRIOR,confidence=confidence,
                     veil_strength=veil_strength,
                     veil_offset=veil_offset,
                     paper=np.full(img.shape,paper,np.float32),footprint=region,
                     repair_mode='document_overlay_full' if coverage_hint is not None else 'document_overlay',
                     method='repeated_document_overlay',
                     text_keepout=keepout)


def estimate(img: np.ndarray) -> MaskLayer | None:
    observations = ocr.read_lines(img)
    angles = []
    for box,text,confidence in observations or []:
        vector = box[1]-box[0]
        angle = float(np.degrees(np.arctan2(vector[1],vector[0])))
        height,length,_ = ocr._geometry(box)
        if confidence>=.65 and len(text)>=5 and length>3*height and 15<abs(angle)<60:
            angles.append(angle)
    if not angles:
        return None
    angle = float(np.median(angles))
    h,w = img.shape[:2]
    document=float((img.mean(2)>235).mean())>.65
    matrix = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    rw = int(abs(matrix[0,0])*w+abs(matrix[0,1])*h)
    rh = int(abs(matrix[0,0])*h+abs(matrix[0,1])*w)
    matrix[0,2] += (rw-w)/2
    matrix[1,2] += (rh-h)/2
    rotated = cv2.warpAffine(img,matrix,(rw,rh),borderValue=(255,255,255))
    valid = cv2.warpAffine(np.ones((h,w),np.uint8),matrix,(rw,rh),flags=cv2.INTER_NEAREST)>0
    seeds = []
    for box,text,confidence in ocr.read_lines(rotated) or []:
        height,length,tilt = ocr._geometry(box)
        if confidence>=.65 and len(text)>=5 and tilt<6 and length>4*height:
            seeds.append(box)
    if not seeds:
        return None
    median_line_h = float(np.median([ocr._geometry(b)[0] for b in seeds]))
    gray=rotated.mean(2).astype(np.uint8)
    reference=max(seeds,key=lambda b:float(b[:,1].mean()))
    size=max(5,int(ocr._geometry(reference)[0]*.5)+1)|1
    feature=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,np.ones((size,size),np.uint8)).astype(np.float32)
    feature=np.minimum(feature,150)

    def _window(b):
        bx0,by0 = np.floor(b.min(0)).astype(int)-3
        bx1,by1 = np.ceil(b.max(0)).astype(int)+3
        bx0=max(0,bx0-int((by1-by0)*(.3 if document else 1.5)));bx1=min(rw,bx1+int((by1-by0)*.3))
        by0=max(0,by0);by1=min(rh,by1)
        return bx0,by0,bx1,by1

    if document:
        windows=[_window(b) for b in seeds]
        th=int(round(float(np.median([w[3]-w[1] for w in windows]))))
        tw=int(round(float(np.median([w[2]-w[0] for w in windows]))))
    else:
        ref_win=_window(reference)
        th,tw=ref_win[3]-ref_win[1],ref_win[2]-ref_win[0]
    if th<8 or tw<40 or tw>rw or th>rh:
        return None
    # 平铺文字多列时，单列模板只覆盖一条对角线（白底红字实测：峰上限放宽到
    # 64 仍只有 13 峰、整列实例未处理）。文档页按旋转后 x 位置分列，各列分别
    # 匹配、分别盖章后取并集；照片纹理维持原单模板行为，不改动照片路径。
    if document and len(seeds)>1:
        clusters=[]
        for b in sorted(seeds,key=lambda b:float(b[:,0].mean())):
            cx=float(b[:,0].mean())
            for cl in clusters:
                if abs(cx-float(np.mean([o[:,0].mean() for o in cl])))<0.75*tw:
                    cl.append(b);break
            else:
                clusters.append([b])
    else:
        clusters=[[reference]]
    column_boxes=[max(cl,key=lambda b:float(b[:,1].mean())) for cl in clusters]
    max_th=max([_window(b)[3]-_window(b)[1] for b in column_boxes]+[th])
    max_tw=max([_window(b)[2]-_window(b)[0] for b in column_boxes]+[tw])
    # 照片路径保持原实现：在未填充的特征图上匹配（零填充黑边会改变页边
    # 相关分数，core_008 织物页因此由"命中"变"拒收"）；文档页在填充图上
    # 直接匹配，列模板尺寸不一，坐标即填充画布坐标。
    feature_unpadded=feature
    # Extra canvas permits partial instances whose full word extends off-image.
    feature=np.pad(feature,((max_th,max_th),(max_tw,max_tw)))
    valid=np.pad(valid,((max_th,max_th),(max_tw,max_tw)))
    gray=np.pad(gray,((max_th,max_th),(max_tw,max_tw)))
    matrix[:,2]+=[max_tw,max_th]
    field=np.zeros(valid.shape,np.float32)
    light_field=np.full(valid.shape,255,np.float32)
    peaks=[]
    multi_column=document and len(clusters)>1
    for column_box in column_boxes:
        bx0,by0,bx1,by1=_window(column_box)
        thk,twk=by1-by0,bx1-bx0
        suffix=bx0+int(twk*.5)
        px0,py0=(suffix+max_tw, by0+max_th) if document else (suffix, by0)
        half=bx1-suffix
        # 淡印章/彩色印章若黑帽响应弱会在此自然失败并回退其他路径；实测用
        # 4× 反相增强特征重试会引入配准漂移（浅黄残留 2,554→13,234），不采用。
        attempts=((feature if document else feature_unpadded,False),)
        col_peaks=None
        template=light=None
        for search,is_enh in attempts:
            patch=search[py0:py0+thk, px0:px0+half]
            if patch.std()<5:
                continue
            # Flat paper repeats are near-identical copies, so the correlation surface
            # supports many more reliable peaks than photo textures; a low cap silently
            # skips whole repeated rows (stamped field leaves unprocessed instances).
            matches=_peaks(cv2.matchTemplate(search,patch,cv2.TM_CCOEFF_NORMED),patch.shape,.50,
                           limit=64 if document else 12)
            if document:
                # 文档页各列只取列种子右侧的匹配：副本虽然近乎一致，左侧异相副本混入
                # 会拉低百分位模板的配准质量（白底红字实测召回 98.6%→94.0%）；
                # 左缘裁切实例由 _fragments 与 recall.expand 并集补全。
                cand=[(x-(suffix-bx0),y,s) for x,y,s in matches
                      if x>=suffix+max_tw and y+thk<=rh+max_th and x-(suffix-bx0)+twk<=rw+max_tw]
            else:
                cand=[(x+max_tw-(suffix-bx0),y+max_th,s) for x,y,s in matches
                      if x>=suffix-bx0 and y+thk<=rh and x-(suffix-bx0)+twk<=rw]
            if not cand:
                continue
            patches=[]
            luminance=[]
            for x,y,_ in cand:
                # 峰坐标统一为填充画布坐标（照片路径已加 max_tw/max_th 偏移），
                # 模板补丁一律从填充后的 feature 裁取；matchTemplate 用的 search
                # 在照片路径是未填充图，不能用于裁补丁。
                patches.append(np.where(valid[y:y+thk,x:x+twk],feature[y:y+thk,x:x+twk],np.nan))
                luminance.append(np.where(valid[y:y+thk,x:x+twk],gray[y:y+thk,x:x+twk],np.nan))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)  # no observations outside rotated source
                recurring=np.nan_to_num(np.nanpercentile(patches,0 if len(cand)<3 else 25,axis=0))
                light_k=np.nan_to_num(np.nanpercentile(luminance,90,axis=0),nan=255)
            if is_enh:
                # 增强图为反相（笔画暗、纸面亮），按低值取模板。
                template_k=(recurring<200).astype(np.uint8)
            else:
                template_k=(recurring>30).astype(np.uint8)
            if document:
                template_k[np.isfinite(patches).sum(axis=0)<2]=0
            if template_k.sum()<50:
                continue
            col_peaks,template,light=cand,template_k,light_k
            break
        if col_peaks is None:
            continue
        peaks.extend(col_peaks)
        partial=_partial_matches(feature,valid,template,col_peaks) if len(col_peaks)>=2 else []
        for x,y,_ in [*col_peaks,*partial]:
            target=field[y:y+thk,x:x+twk]
            np.maximum(target,template,out=target)
            light_field[y:y+thk,x:x+twk]=np.minimum(light_field[y:y+thk,x:x+twk],
                                                 np.where(template>0,light,255))
        if document:
            for x,y,tx,ty,frag in _fragments(feature,template,col_peaks):
                ph,pw=frag.shape
                np.maximum(field[y:y+ph,x:x+pw],frag,out=field[y:y+ph,x:x+pw])
                values=np.where(frag>0,light[ty:ty+ph,tx:tx+pw],255)
                np.minimum(light_field[y:y+ph,x:x+pw],values,out=light_field[y:y+ph,x:x+pw])
            # 单列文档页保留原边缘补片行为；多列时各列模板/光照不同，边缘残字
            # 交由 recall.expand 的并集补全（对齐戳印 _add_clipped）兜底。
            if not multi_column:
                keepout = ocr.text_zones(img)
                protected = (cv2.warpAffine(keepout.astype(np.uint8), matrix,
                                           (valid.shape[1], valid.shape[0]),
                                           flags=cv2.INTER_NEAREST) > 0) if keepout is not None else None
                padded_light = np.pad(light, ((thk, thk), (twk, twk)), constant_values=255)
                for x, y, tx, ty, frag in _edge_fragments(feature, valid, template, col_peaks, protected):
                    ph, pw = frag.shape
                    target = field[y:y+ph, x:x+pw]
                    np.maximum(target, frag, out=target)
                    values = np.where(frag > 0, padded_light[ty+thk:ty+thk+ph, tx+twk:tx+twk+pw], 255)
                    target_light = light_field[y:y+ph, x:x+pw]
                    np.minimum(target_light, values, out=target_light)
    if len(peaks)<2:
        return None
    field*=valid
    coverage=cv2.warpAffine(field,cv2.invertAffineTransform(matrix),(w,h))
    # 文档页选区同样覆盖笔画外 1–2px 的抗锯齿晕圈，避免修复后留空心轮廓。
    region=cv2.dilate((coverage>(.10 if document else .15)).astype(np.uint8),
                      np.ones((5,5) if document else (3,3),np.uint8))>0
    if not .0001<float(region.mean())<.20:
        return None
    confidence=float(min(p[2] for p in peaks))
    if document:
        profile=cv2.warpAffine(light_field,cv2.invertAffineTransform(matrix),(w,h),
                               borderValue=255)
        # 无交叠单极拟合只在大字（OCR 行高中位数≥48px）启用：小字（28/40px
        # 实测）覆盖区由边缘抗锯齿主导，直方图极不稳，反解欠校正（小字28
        # 残留 65→24,031、小字40 残留 1,053→37,877）；小字页保留原有路径。
        fitted=_document_overlay(img,region,profile,confidence,coverage_hint=coverage,
                                 allow_single_pole=median_line_h >= 48)
        if fitted is not None:
            return fitted
    # On white documents retain existing dark ink instead of asking a generic
    # inpainter to invent glyph strokes. Overlapping watermark may remain there.
    original_gray=img.mean(2)
    if document:
        inside=original_gray[(coverage>.5)&(original_gray>45)]
        if inside.size:
            paper_watermark=float(np.percentile(inside,25))
            region &= original_gray>max(35,min(65,paper_watermark*.65))
    color=np.median(img[region],axis=0).astype(np.float32)
    paper=np.broadcast_to(np.percentile(img.reshape(-1,3),97,axis=0).astype(np.float32),img.shape).copy()
    # Retain the registered glyph core separately from the padded repair
    # footprint. A matching template alone is insufficient at an individual
    # pixel: the original (unblurred) input must also show the dark stroke.
    native_feature = cv2.morphologyEx(original_gray.astype(np.float32),
                                     cv2.MORPH_BLACKHAT, np.ones((size,size),np.uint8))
    repeat_support = (coverage >= .75) & (native_feature > 30)
    return MaskLayer(alpha=region.astype(np.float32),source=MaskSource.PRIOR,
                     confidence=confidence,
                     veil_strength=np.zeros(3,np.float32),veil_offset=color,paper=paper,
                     footprint=region,repair_mode='inpaint',method='repeated_oblique_text',
                     repeat_stroke_support=repeat_support)
