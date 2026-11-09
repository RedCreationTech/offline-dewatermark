"""Download NEW CC0 photographs for the locked independent validation set.

Differs from tools/fetch_cc0_training_bases.py on purpose:
- different search queries (never reuses v1 query/result order);
- excludes every v1 accession by title, raw sha256 AND perceptual hash
  (near-duplicates of v1 photos are rejected even from different pages);
- stops at 12 license-verified photos (3 per background category).

Raw bytes, API evidence and hashes are kept under data/watermark_independent_v1/.
Never imports upstream executable code.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, io, json, time, urllib.parse, urllib.request, urllib.error

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/watermark_independent_v1'
V1 = ROOT / 'data/watermark_training_v1'
UA = 'WatermarkIndependentSet/1.0 (local dataset curation; CC0 provenance)'
TARGET = 12
PER_CATEGORY = 3
# New queries: disjoint wording from v1's QUERIES so result order cannot repeat v1.
QUERIES = [
    ('landscape', 'coast shoreline -painting -drawing -art -map -portrait'),
    ('vegetation', 'grassland meadow -painting -drawing -portrait'),
    ('architecture', 'castle exterior -painting -drawing -portrait'),
    ('objects', 'food cuisine plate -painting -drawing -portrait'),
    ('landscape', 'desert dunes -painting -drawing -portrait'),
    ('vegetation', 'autumn leaves park -painting -drawing -portrait'),
    ('architecture', 'city street -painting -drawing -portrait'),
    ('objects', 'workshop tools -painting -drawing -portrait'),
]
CROSS_SET_DHASH_MAX = 5  # stricter than v1's within-set threshold of 3


def dhash(im: Image.Image) -> int:
    a = list(im.convert('L').resize((9, 8)).getdata())
    return sum((a[y * 9 + x] > a[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def fetch(url: str, limit: int = 24_000_000) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=40) as r:
                b = r.read(limit + 1)
                if len(b) > limit:
                    raise ValueError('download exceeds configured limit')
                return b
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                raise
            if attempt == 2:
                raise
            time.sleep(2 + attempt)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def v1_exclusions() -> tuple[set, set, list]:
    rows = json.loads((V1 / 'sources/commons_cc0.json').read_text())
    titles = {r['commons_title'] for r in rows}
    raw_hashes = {r['raw_sha256'] for r in rows}
    hashes = set()
    for q in V1.glob('sources/quarantined_*.json'):
        for item in json.loads(q.read_text()):
            s = item.get('sample') or item
            if isinstance(s, dict) and s.get('commons_title'):
                titles.add(s['commons_title'])
    v1_hashes = []
    for r in rows:
        with Image.open(V1 / r['normalized_path']) as im:
            v1_hashes.append(dhash(im))
    return titles, raw_hashes, v1_hashes


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'sources/raw').mkdir(parents=True, exist_ok=True)
    (DEST / 'sources/evidence').mkdir(parents=True, exist_ok=True)
    (DEST / 'bases/photos').mkdir(parents=True, exist_ok=True)
    target = DEST / 'sources/commons_cc0_independent.json'
    rows = json.loads(target.read_text()) if target.exists() else []
    seen_titles, seen_raw, v1_hashes = v1_exclusions()
    seen = seen_titles | {r['commons_title'] for r in rows}
    seen_hash = seen_raw | {r['raw_sha256'] for r in rows}
    counts = {k: sum(r['background'] == k for r in rows) for k in ['landscape', 'vegetation', 'architecture', 'objects']}
    for qi, (category, query) in enumerate(QUERIES):
        if len(rows) >= TARGET:
            break
        if counts[category] >= PER_CATEGORY:
            continue
        params = dict(action='query', format='json', generator='search',
                      gsrsearch='incategory:"CC-Zero" filetype:bitmap ' + query,
                      gsrnamespace=6, gsrlimit=30, prop='imageinfo',
                      iiprop='url|extmetadata|size|sha1', iiurlwidth=1280)
        url = 'https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode(params)
        raw = fetch(url)
        (DEST / f'sources/evidence/search_{qi:02d}.json').write_bytes(raw)
        pages = json.loads(raw).get('query', {}).get('pages', {})
        for page in sorted(pages.values(), key=lambda p: p.get('index', 0)):
            if len(rows) >= TARGET or counts[category] >= PER_CATEGORY:
                break
            if page['title'] in seen:
                print('SKIP seen in v1 or set', page['title'], flush=True)
                continue
            info = page.get('imageinfo', [{}])[0]
            meta = info.get('extmetadata', {})
            lic = meta.get('LicenseShortName', {}).get('value', '')
            licurl = meta.get('LicenseUrl', {}).get('value', '')
            if lic != 'CC0' or 'creativecommons.org/publicdomain/zero/1.0' not in licurl:
                continue
            if not urllib.parse.urlparse(info.get('url', '')).path.lower().endswith(('.jpg', '.jpeg')):
                continue
            if info.get('size', 0) > 24_000_000 or min(info.get('width', 0), info.get('height', 0)) < 768:
                continue
            try:
                download_url = info.get('thumburl', info['url'])
                imagebytes = fetch(download_url)
                digest = hashlib.sha256(imagebytes).hexdigest()
                if digest in seen_hash:
                    print('SKIP duplicate bytes', page['title'], flush=True)
                    continue
                image = ImageOps.exif_transpose(Image.open(io.BytesIO(imagebytes))).convert('RGB')
                image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
                candidate_hash = dhash(image)
                if any(hamming(candidate_hash, h) <= CROSS_SET_DHASH_MAX for h in v1_hashes):
                    print('SKIP near duplicate of v1', page['title'], flush=True)
                    continue
                pid = f'xphoto_{len(rows) + 1:03d}'
                rawpath = DEST / f'sources/raw/{pid}.jpg'
                rawpath.write_bytes(imagebytes)
                normalized = DEST / f'bases/photos/{pid}.png'
                image.save(normalized)
                evidence = DEST / f'sources/evidence/{pid}.json'
                evidence.write_text(json.dumps(page, ensure_ascii=False, indent=2))
                rows.append(dict(
                    id=pid, commons_title=page['title'], page_url=info['descriptionurl'],
                    download_url=download_url, license='CC0-1.0', license_url=licurl,
                    author_html=meta.get('Artist', {}).get('value', ''),
                    description_html=meta.get('ImageDescription', {}).get('value', ''),
                    source_group='commons:' + str(page['pageid']), background=category,
                    raw_path=str(rawpath.relative_to(DEST)), raw_sha256=digest,
                    normalized_path=str(normalized.relative_to(DEST)),
                    normalized_sha256=hashlib.sha256(normalized.read_bytes()).hexdigest(),
                    evidence_path=str(evidence.relative_to(DEST)),
                    size=list(image.size), downloaded_on='2026-11-02',
                    transformations=['EXIF orientation', 'RGB', 'long edge at most 1536; no upscaling']))
                seen.add(page['title'])
                seen_hash.add(digest)
                counts[category] += 1
                target.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
                print(pid, category, page['title'], image.size, flush=True)
            except urllib.error.HTTPError as e:
                if e.code in (403, 429):
                    print('PAUSED_HTTP_LIMIT', e.code, 'resume later, completed', len(rows), flush=True)
                    return
                print('SKIP', page['title'], str(e)[:180], flush=True)
            except Exception as e:
                print('SKIP', page['title'], str(e)[:180], flush=True)
            time.sleep(2)
    print('DOWNLOADED', len(rows), counts, flush=True)
    if len(rows) < TARGET:
        raise RuntimeError(f'Need {TARGET} license-verified photos; incomplete accession retained for resume')


if __name__ == '__main__':
    main()
