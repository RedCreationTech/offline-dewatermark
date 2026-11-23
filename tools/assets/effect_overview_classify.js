// 效果定档工具（随明细报告一起加载）：
// 每张样本卡片提供 效果好／提示词可处理／待解决 三个定档按钮（键盘 1/2/3），
// 预筛线 召回≥98% 且 精确率≥90% 只作“候选”标注；定档以人工点击为准，存在浏览器本地。
// 顶部工具条显示进度、只看未定档、导出 JSON（交给 tools/archive_classified.py 归档）。
(() => {
  const data = JSON.parse(document.getElementById('reviewed-data').textContent);
  const cfg = data.classify;
  if (!cfg || !cfg.enabled) return;
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const CLASSES = cfg.classes;                     // ['效果好','提示词可处理','待解决']
  const HOTKEYS = {'1': 0, '2': 1, '3': 2};
  const STORE_KEY = 'wm-classify:' + data.manifest.source_record_sha256;
  const load = () => { try { return JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); } catch { return {}; } };
  const save = m => localStorage.setItem(STORE_KEY, JSON.stringify(m));
  let assigned = load();                            // {rowKey: classIndex}
  let lastHover = null;                             // 最近鼠标所在的卡片 data-key

  const rowByKey = {};
  for (const r of data.rows) rowByKey[r.key] = r;
  const eligible = r => !r.negative && r.status !== 'error' && r.recall != null && r.precision != null;
  const isCandidate = r => eligible(r) && r.recall >= cfg.good_recall && r.precision >= cfg.good_precision;

  // ---- 顶部工具条（插在筛选行之后）----
  const bar = document.createElement('div');
  bar.className = 'classify-bar';
  bar.innerHTML = `
    <span class="cls-title">效果定档（人工定档后导出，供归档进数据集）</span>
    <span class="cls-progress" id="cls-progress"></span>
    <label class="cls-filter"><input type="checkbox" id="cls-only-unassigned"> 只看未定档</label>
    <button type="button" id="cls-export">导出任档结果（JSON）</button>
    <span class="cls-hint">卡片按钮或悬停后按 1／2／3 定档；预筛“候选效果好”≠已定档</span>`;
  const filters = document.querySelector('.filters');
  filters.parentNode.insertBefore(bar, filters.nextSibling);

  function refreshProgress() {
    const rows = data.rows.filter(eligible);
    const done = rows.filter(r => assigned[r.key] != null).length;
    $('cls-progress').textContent = `已人工定档 ${done} / ${rows.length}` + (rows.length && done === rows.length ? '（可导出）' : '');
    $('cls-progress').classList.toggle('cls-done', rows.length > 0 && done === rows.length);
  }

  // ---- 卡片注入：预筛徽标 + 定档按钮 ----
  function decorate(card) {
    if (card.dataset.clsReady) return;
    card.dataset.clsReady = '1';
    const key = card.dataset.key;
    const row = rowByKey[key];
    if (!row) return;
    const body = card.querySelector('.card-body') || card;
    const holder = document.createElement('div');
    holder.className = 'cls-holder';
    holder.innerHTML = (isCandidate(row) ? '<span class="cls-badge">机器预筛：候选效果好（≥' +
        Math.round(cfg.good_recall * 100) + '%／≥' + Math.round(cfg.good_precision * 100) + '%）</span>' : '') +
      '<span class="cls-state" data-state></span><span class="cls-btns">' +
      CLASSES.map((c, i) => `<button type="button" class="cls-btn" data-cls="${i}">${c}（${i + 1}）</button>`).join('') +
      '</span>';
    holder.addEventListener('click', e => {
      const b = e.target.closest('[data-cls]');
      if (!b) return;
      const idx = Number(b.dataset.cls);
      if (assigned[key] === idx) delete assigned[key]; else assigned[key] = idx;   // 再点一次取消
      save(assigned);
      paint(card, row);
      refreshProgress();
      applyUnassignedFilter();
    });
    body.appendChild(holder);
    paint(card, row);
  }
  function paint(card, row) {
    const key = card.dataset.key;
    const idx = assigned[key];
    card.dataset.cls = idx == null ? '' : String(idx);
    const state = card.querySelector('[data-state]');
    if (state) state.textContent = idx == null ? '未定档' : '已定档：' + CLASSES[idx];
    card.querySelectorAll('.cls-btn').forEach(b => b.classList.toggle('on', Number(b.dataset.cls) === idx));
  }
  function decorateAll() { document.querySelectorAll('#cards article.card').forEach(decorate); refreshProgress(); applyUnassignedFilter(); }

  // ---- 只看未定档：渲染后隐藏已定档卡片 ----
  function applyUnassignedFilter() {
    const on = $('cls-only-unassigned').checked;
    document.querySelectorAll('#cards article.card').forEach(card => {
      const row = rowByKey[card.dataset.key];
      // 勾选后隐藏“已定档”的卡片，只留未定档的给人处理。
      card.style.display = (on && row && eligible(row) && assigned[card.dataset.key] != null) ? 'none' : '';
    });
  }
  $('cls-only-unassigned').addEventListener('change', applyUnassignedFilter);

  // ---- 键盘 1/2/3：给定到鼠标悬停的卡片 ----
  document.addEventListener('mouseover', e => {
    const card = e.target.closest && e.target.closest('#cards article.card');
    if (card) lastHover = card.dataset.key;
  });
  document.addEventListener('keydown', e => {
    if (e.target.matches('input,select,textarea')) return;
    const idx = HOTKEYS[e.key];
    if (idx == null || !lastHover) return;
    const card = document.querySelector(`#cards article.card[data-key="${CSS.escape(lastHover)}"]`);
    if (!card) return;
    const btn = card.querySelector(`.cls-btn[data-cls="${idx}"]`);
    if (btn) btn.click();
  });

  // ---- 导出 JSON ----
  $('cls-export').addEventListener('click', () => {
    const samples = data.rows.map(r => ({
      key: r.key, id: r.id, page: r.page, group: r.group, style: r.style, label: r.label,
      negative: r.negative, status: r.status,
      recall: r.recall, precision: r.precision, strong_recall: r.strong_recall,
      missed_pixels: r.missed_pixels, false_positive_pixels: r.false_positive_pixels,
      fp_frac: r.fp_frac, watermark_mae: r.watermark_mae, background_mae: r.background_mae,
      elapsed_ms: r.elapsed_ms, method: r.method, repair_mode: r.repair_mode,
      machine_bucket: r.bucket,
      machine_candidate: isCandidate(r),
      user_class: assigned[r.key] == null ? null : CLASSES[assigned[r.key]],
    }));
    const payload = {
      schema_version: 1,
      created_at: new Date().toISOString(),
      run_dir: cfg.run_dir,
      record_sha256: data.manifest.source_record_sha256,
      report_title: document.title,
      preclass_thresholds: {recall: cfg.good_recall, precision: cfg.good_precision},
      classes: CLASSES,
      samples,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = '效果定档_' + data.manifest.source_record_sha256.slice(0, 8) + '.json';
    a.click();
    URL.revokeObjectURL(a.href);
    const missing = samples.filter(s => !s.negative && s.status !== 'error' && s.user_class == null).length;
    if (missing) alert(`还有 ${missing} 张带水印样本未定档；可继续定档后再导出，或由归档脚本拒绝接收。`);
  });

  // 分页/筛选重渲染后重新注入
  const mo = new MutationObserver(() => decorateAll());
  mo.observe($('cards'), {childList: true});
  decorateAll();
})();
