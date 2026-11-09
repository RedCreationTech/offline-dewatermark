"""Select strokes from whole watermark copies aligned in source coordinates."""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import ocr, repeated
from .observed_text import _stroke_template


def _seed_boxes(lines, base):
    candidates = []
    height, width = base.shape
    for box, text, score in lines:
        size, length, tilt = ocr._geometry(box)
        if score < .60 or len(text.strip()) < 4 or length < 3 * size or not 15 < tilt < 60:
            continue
        pad = max(2, round(size * .18))
        x0, y0 = np.maximum(np.floor(box.min(0)).astype(int) - pad, 0)
        x1, y1 = np.minimum(np.ceil(box.max(0)).astype(int) + pad, (width, height))
        if x1 <= x0 or y1 <= y0:
            continue
        domain = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(domain, np.rint(box - (x0, y0)).astype(np.int32), 1)
        domain = cv2.dilate(domain, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
        seed = base[y0:y1, x0:x1] & (domain > 0)
        if seed.sum() >= 30:
            candidates.append((length, size, x0, y0, domain, seed))
    return sorted(candidates, key=lambda item: item[0], reverse=True)[:4]


def _signal(image, size, x, y, seed):
    gray = image.mean(2, dtype=np.float32)
    channels = [image[..., channel].astype(np.float32) - gray for channel in range(3)]
    channels.append(gray)
    window = min(31, max(9, round(.28 * size) | 1))
    region = np.s_[y:y + seed.shape[0], x:x + seed.shape[1]]
    choices = []
    for channel in channels:
        encoded = np.rint(np.clip(channel + (128 if channel is not gray else 0), 0, 255)).astype(np.uint8)
        local = cv2.medianBlur(encoded, window)
        residual = encoded.astype(np.float32) - local
        for sign in (1, -1):
            values = sign * residual[region][seed]
            strength = float(np.percentile(values, 80))
            if strength >= 3:
                choices.append((strength, encoded, residual, sign))
    return sorted(choices, key=lambda item: item[0], reverse=True)


def _template(field, patches, domain, size):
    recurring = np.median(patches, axis=0)
    initial = _stroke_template(recurring) & (domain > 0)
    # Exclude the observed ink when measuring its local background. The
    # neighbourhood defines where to measure, never where to select pixels.
    near = cv2.dilate(initial.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    weights = (~near).astype(np.float32)
    sigma = max(2., size * .12)
    blurred = cv2.GaussianBlur(weights, (0, 0), sigma)
    background = np.stack([cv2.GaussianBlur(copy * weights, (0, 0), sigma)
                           / np.maximum(blurred, .01) for copy in field])
    contrast = np.median(field - background, axis=0)
    detail = _stroke_template(contrast)
    return ((initial | (near & detail & (blurred >= .10))) & (domain > 0))


def _flat_background_edges(image, peaks, glyphs, domain):
    """Confirm faint ink against a dominant RGB background on multiple copies."""
    sample = image[::8, ::8].reshape(-1, 3)
    colours, counts = np.unique(sample, axis=0, return_counts=True)
    dominant = counts.argmax()
    if counts[dominant] < .5 * len(sample):
        return glyphs
    paper = colours[dominant].astype(np.float32)
    th, tw = glyphs.shape
    copies = np.stack([image[y:y+th, x:x+tw] for x, y, _ in peaks[:8]])
    same = np.stack([np.sum(np.all(copies == patch, axis=3), axis=0) >= 2 for patch in copies])
    delta = copies.astype(np.float32) - paper
    lengths = np.linalg.norm(delta, axis=3)
    core = same & glyphs & (lengths > 12)
    if core.sum() < 50:
        return glyphs
    direction = np.median(delta[core] / lengths[core, None], axis=0)
    magnitude = np.linalg.norm(direction)
    if magnitude < .90:
        return glyphs
    direction /= magnitude
    strength = delta @ direction
    error = np.linalg.norm(delta - strength[..., None] * direction, axis=3)
    confirmed = (same & (strength > .5) & (error <= 1.7)).any(axis=0) & (domain > 0)
    count, labels = cv2.connectedComponents((glyphs | confirmed).astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[labels[glyphs]] = True
    return keep[labels]


def _add_clipped(selection, feature, template, match_ink, peaks, domain):
    height, width = selection.shape
    th, tw = template.shape
    # A displacement observed at least twice can predict where to look for
    # a clipped copy. Its visible ink still has to match the current image.
    shifts = Counter((x2 - x1, y2 - y1) for x1, y1, _ in peaks
                     for x2, y2, _ in peaks if (x1, y1) != (x2, y2))
    proposals = {(x + dx, y + dy) for (dx, dy), count in shifts.items() if count >= 2
                 for x, y, _ in peaks}
    for x, y in proposals:
        if 0 <= x <= width - tw and 0 <= y <= height - th:
            continue
        l, t, r, b = max(0, x), max(0, y), min(width, x + tw), min(height, y + th)
        if l >= r or t >= b:
            continue
        part = np.s_[t-y:b-y, l-x:r-x]
        ink = template[part]
        support = match_ink[part] & ink
        if support.sum() < 30:
            continue
        observed = feature[t:b, l:r]
        blank = (domain[part] > 0) & ~ink
        if not blank.any():
            continue
        # Ink must be stronger than neighbouring gaps in the visible piece.
        # This rejects a scene edge filling the same rectangle.
        if np.mean(observed[support] > 2) >= .60 and np.mean(observed[support]) > 2 * max(1., np.mean(observed[blank])):
            selection[t:b, l:r] |= ink


def _align_core(image, base, lines):
    """Internal: return (region, peaks) or (None, [])."""
    height, width = base.shape
    for _, size, x, y, domain, seed in _seed_boxes(lines, base):
        th, tw = domain.shape
        if th > height or tw > width:
            continue
        for strength, encoded, residual, sign in _signal(image, size, x, y, seed)[:3]:
            feature = np.clip(sign * residual, 0, max(8., strength * 1.5))
            patch = feature[y:y+th, x:x+tw] * domain
            if not .03 < np.mean(patch[domain > 0] > 3) < .50:
                continue
            scores = cv2.matchTemplate(feature, patch, cv2.TM_CCORR_NORMED, mask=domain)
            scores[~np.isfinite(scores)] = 0
            peaks = repeated._peaks(scores, (max(1, th // 2), max(1, tw // 2)), .55, limit=48)
            if len(peaks) < 3:
                continue
            patches = np.stack([feature[py:py+th, px:px+tw] for px, py, _ in peaks])
            copies = [sign * encoded[py:py+th, px:px+tw].astype(np.float32) for px, py, _ in peaks]
            glyphs = _template(copies, patches, domain, size)
            if glyphs.sum() < 50:
                continue
            glyphs = _flat_background_edges(image, peaks, glyphs, domain)
            region = np.zeros_like(base)
            for px, py, _ in peaks:
                region[py:py+th, px:px+tw] |= glyphs
            match_ink = (patches > 2).sum(axis=0) * 2 >= len(peaks)
            _add_clipped(region, feature, glyphs, match_ink, peaks, domain)
            return region, peaks
    return None, []


def align(image, base, lines):
    """Return a source-supported binary glyph field, or no verified result."""
    region, _ = _align_core(image, base, lines)
    return region


def align_with_peaks(image, base, lines):
    """Return (region, peaks) for downstream completion."""
    return _align_core(image, base, lines)


def _measure_grid(positions):
    xs, ys = (sorted({point[axis] for point in positions}) for axis in (0, 1))
    if len(xs) < 2 or len(ys) < 2:
        return None
    dx, dy = int(np.min(np.diff(xs))), int(np.min(np.diff(ys)))
    if not (np.all(np.diff(xs) == dx) and np.all(np.diff(ys) == dy)):
        return None
    return xs[0], ys[0], dx, dy


def _estimate_background(copy, excluded):
    keep = (~excluded).astype(np.float32)
    support = np.zeros(keep.shape, np.float32)
    first = np.zeros_like(copy)
    second = np.zeros_like(copy)
    for sigma in (2.5, 4., 6., 9., 13.):
        inadequate = support < .15
        support[inadequate] = cv2.GaussianBlur(keep, (0, 0), sigma)[inadequate]
        first[inadequate] = cv2.GaussianBlur(copy * keep[..., None], (0, 0), sigma)[inadequate]
        second[inadequate] = cv2.GaussianBlur(copy ** 2 * keep[..., None], (0, 0), sigma)[inadequate]
    support = np.maximum(support, .001)[..., None]
    mean = first / support
    roughness = np.sqrt(np.maximum(second / support - mean ** 2, 0).mean(2))
    return mean, roughness


def _fit_colour_and_opacity(copies, backgrounds, roughness, core, start_colour):
    eligible = roughness < 6
    training = core & (eligible.sum(0) >= 2)
    if training.sum() < 500:
        return None
    weights = 1 / (.6 + roughness ** 2)
    base = backgrounds[:, training]
    observed = copies[:, training]
    fit_weights = weights[:, training] * eligible[:, training]

    def estimate(colour, bg, obs):
        direction = colour - bg
        strength = ((obs - bg) * direction).sum(-1) / np.maximum((direction ** 2).sum(-1), 1)
        return direction, strength

    def residual(colour):
        direction = colour - base
        delta = observed - base
        opacity = (fit_weights * (delta * direction).sum(-1)).sum(0) / np.maximum(
            (fit_weights * (direction ** 2).sum(-1)).sum(0), 1)
        return ((delta - opacity[None, :, None] * direction) * np.sqrt(fit_weights)[..., None]).ravel()

    colour = least_squares(residual, start_colour, loss='soft_l1', f_scale=1., bounds=(0, 255)).x
    direction, values = estimate(colour, backgrounds, copies)
    error = np.sqrt(np.mean((copies - backgrounds - values[..., None] * direction) ** 2, axis=3))
    weights *= np.minimum(1., 2 / (error + .1))
    opacity = np.median(values, axis=0)
    for _ in range(4):
        weight = weights / np.maximum(1., np.abs(values - opacity) / .032)
        opacity = (values * weight).sum(0) / weight.sum(0)
    agreement = (values > .025) & (np.abs(values - opacity) < np.maximum(.035, .10 * opacity)) & (error < 4)
    votes = agreement.sum(0)
    return colour, opacity, votes, int(training.sum())


def _verify_additions(image, proposed, unavailable, alpha_map, colour_map):
    yy, xx = np.nonzero(proposed)
    if len(xx) == 0:
        return np.zeros_like(proposed)
    observed = image[yy, xx]
    alpha = alpha_map[yy, xx, None]
    ink = colour_map[yy, xx]
    overlay_best = np.full(len(xx), np.inf)
    clean_best = np.full(len(xx), np.inf)
    height, width = proposed.shape
    for dy in range(-18, 19):
        for dx in range(-18, 19):
            distance2 = dx * dx + dy * dy
            if distance2 > 18 ** 2:
                continue
            y = np.clip(yy + dy, 0, height - 1)
            x = np.clip(xx + dx, 0, width - 1)
            valid = ~unavailable[y, x]
            if not valid.any():
                continue
            reference = image[y, x]
            mix = reference * (1 - alpha) + alpha * ink
            cost = .09 * np.sqrt(distance2)
            with_overlay = np.abs(mix - observed).mean(1) + cost
            without_overlay = np.abs(reference - observed).mean(1) + cost
            np.minimum(overlay_best, np.where(valid, with_overlay, np.inf), out=overlay_best)
            np.minimum(clean_best, np.where(valid, without_overlay, np.inf), out=clean_best)
    accepted = (overlay_best <= 9.) & (clean_best >= overlay_best + 1.)
    confirmed = np.zeros_like(proposed)
    confirmed[yy[accepted], xx[accepted]] = True
    return confirmed


def complete_and_invert(image, initial_mask, peaks):
    """Refine aligned mask with source-only completion and transparency inversion.

    Returns (combined_mask, inverse_mask, alpha_map, colour_map) or four Nones on failure.
    """
    if not peaks or len(peaks) < 3:
        return None, None, None, None

    height, width = image.shape[:2]
    positions = sorted((int(x), int(y)) for x, y, _ in peaks)
    grid = _measure_grid(positions)
    if grid is None:
        return None, None, None, None

    x0, y0, tw, th = grid
    rgb = image.astype(np.float32)

    observations = np.zeros((len(positions), th, tw, 3), dtype=np.float32)
    observed_mask = np.zeros((len(positions), th, tw), dtype=bool)
    for i, (x, y) in enumerate(positions):
        yh = min(y + th, height)
        xw = min(x + tw, width)
        vh, vw = yh - y, xw - x
        observations[i, :vh, :vw] = rgb[y:yh, x:xw]
        observed_mask[i, :vh, :vw] = initial_mask[y:yh, x:xw]

    base_glyph = observed_mask.sum(0) >= 3
    colour_deltas = np.stack([copy - cv2.medianBlur(copy.astype(np.uint8), 13) for copy in observations])
    colour_deltas -= colour_deltas.mean(3, keepdims=True)
    strong = observed_mask & (np.linalg.norm(colour_deltas, axis=3) > 10)
    if strong.sum() < 100:
        return None, None, None, None

    hue = np.median(colour_deltas[strong], axis=0)
    hue /= np.linalg.norm(hue)
    contrast = colour_deltas @ hue
    near = cv2.dilate(base_glyph.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    extra_support = ((contrast > 3).sum(0) >= 3) & near
    exclude = cv2.dilate((base_glyph | extra_support).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    core = base_glyph & (np.median(contrast, axis=0) > 12)
    bg, rough = map(np.stack, zip(*(_estimate_background(copy, exclude) for copy in observations)))
    start_colour = np.median(observations[strong], axis=0)

    alpha_map = np.zeros((height, width), np.float32)
    colour_map = np.zeros((height, width, 3), np.float32)
    proof = np.zeros((height, width), bool)
    unavailable = initial_mask.copy()

    common_model = _fit_colour_and_opacity(observations, bg, rough, core, start_colour)
    if common_model is None:
        return None, None, None, None

    for y in range(y0, height, th):
        for x in range(x0, width, tw):
            vh = min(th, height - y)
            vw = min(tw, width - x)
            if (x, y) in positions:
                peers = np.arange(len(positions)) != positions.index((x, y))
                model = _fit_colour_and_opacity(observations[peers], bg[peers], rough[peers], core, start_colour)
            else:
                if (initial_mask[y:y + vh, x:x + vw] & base_glyph[:vh, :vw]).sum() < 30:
                    continue
                model = common_model
            if model is None:
                continue
            colour, opacity, votes, _ = model
            alpha_map[y:y + vh, x:x + vw] = opacity[:vh, :vw]
            colour_map[y:y + vh, x:x + vw] = colour
            proof[y:y + vh, x:x + vw] = (near[:vh, :vw] & (votes[:vh, :vw] >= 3) &
                                           (opacity[:vh, :vw] >= .032) & (opacity[:vh, :vw] <= .84))
            unavailable[y:y + vh, x:x + vw] |= exclude[:vh, :vw]

    proposals = proof & ~initial_mask
    added = _verify_additions(rgb, proposals, unavailable, alpha_map, colour_map)
    combined = initial_mask | added
    inverse = combined & proof

    yy, xx = np.nonzero(inverse)
    if len(yy) == 0:
        return combined, inverse, alpha_map, colour_map

    a = alpha_map[yy, xx, None]
    raw = (rgb[yy, xx] - a * colour_map[yy, xx]) / (1 - a)
    plausible = np.all((raw >= -7) & (raw <= 262), axis=1)
    inverse[yy[~plausible], xx[~plausible]] = False

    return combined, inverse, alpha_map, colour_map
