
(() => {
 const data=JSON.parse(document.getElementById('reviewed-data').textContent);
 const $=id=>document.getElementById(id), modeNames={desktop:'桌面默认',desktop_protected:'关闭笔画补全',cli:'CLI默认'};
 const presentModes=Object.keys(data.counts);
 const currentMode=()=>$('mode')?$('mode').value:(presentModes.includes('desktop')?'desktop':presentModes[0]);
 const ranks={good:0,review:1,poor:2};const tabs=[...document.querySelectorAll('[data-bucket]')];
 const state={bucket:'good',page:1,size:16};
 const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const pct=n=>n==null?'未评估':`${(n*100).toFixed(1)}%`,num=n=>n==null?'未评估':Number(n).toFixed(2);
 const count=n=>Number(n).toLocaleString('zh-CN');
 const unique=rows=>{
  const groups=new Map();for(const row of rows){const key=`${row.input_sha256}:${row.page}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row);}
  return [...groups.values()].map(rows=>{const chosen=[...rows].sort((a,b)=>ranks[b.bucket]-ranks[a.bucket]||(b.watermark_mae??0)-(a.watermark_mae??0))[0];return {...chosen,occurrences:rows};});
 };
 const entries=()=>{const rows=data.rows.filter(r=>r.mode===currentMode());return $('duplicates').checked?rows.map(r=>({...r,occurrences:[r]})):unique(rows);};
 const labelFor=r=>{const name=r.id.split('/').pop();const hints=[['cat_lowlight','低光猫咪'],['food_dish','菜品照片'],['food_soup','汤品照片'],['indoor_signage','室内标识'],['outdoor_building','建筑照片'],['charts_','图表'],['document_','文档'],['photo_','照片'],['textures_','纹理']];const hint=hints.find(([key])=>name.includes(key));return `${r.label||r.style}${hint?' · '+hint[1]:''}${r.page>1?' · 第'+r.page+'页':''}`;};
 function image(src,full,caption){return src?`<figure><button type="button" data-image="${esc(full||src)}" data-caption="${esc(caption)}"><img loading="lazy" src="${esc(src)}" alt="${esc(caption)}"></button><figcaption>${esc(caption)}</figcaption></figure>`:'';}
 function card(r){
  const contexts=r.occurrences.length>1?`<details><summary>同图有 ${r.occurrences.length} 条批次记录，默认展示较差档</summary>${r.occurrences.map(o=>`<p class="record-id">${esc(o.key)} · ${esc({good:'较好初筛',review:'待复核',poor:'问题较多'}[o.bucket])} · <a href="${esc(o.output)}" target="_blank" rel="noopener">实际结果</a></p>`).join('')}</details>`:'';
  const showOld=(state.bucket==='worse'||state.bucket==='better')&&r.old_output;
  return `<article class="card"><div class="card-head"><span class="eyebrow">${esc(r.group)} · ${r.negative?'<span class="badge negative">无水印对照</span>':'<span class="badge">带水印样本</span>'}</span><h2>${esc(labelFor(r))}</h2></div><div class="compare">${image(r.source_preview,r.source_preview,'原图预览')}${showOld?image(r.old_output,r.old_output,'上次基线处理结果'):''}${image(r.output_preview,r.output,'本次处理结果')}</div><div class="card-body"><ul class="reasons">${r.reasons.map(s=>`<li>${esc(s)}</li>`).join('')}</ul><div class="metrics"><span>全部笔画召回 <strong>${pct(r.recall)}</strong></span><span>选区精确率 <strong>${pct(r.precision)}</strong></span><span>水印区对参考误差 <strong>${num(r.source_watermark_mae)} → ${num(r.watermark_mae)}</strong></span><span>非水印区新增误差 <strong>${num(r.background_damage_mae)}</strong></span><span>漏选／多选像素 <strong>${r.missed_pixels==null?'未评估':count(r.missed_pixels)} ／ ${r.false_positive_pixels==null?'未评估':count(r.false_positive_pixels)}</strong></span><span>耗时 <strong>${r.elapsed_ms==null?'未知':(r.elapsed_ms/1000).toFixed(1)+'秒'}</strong></span></div><details><summary>查看选区、核对图及处理记录</summary><div class="selection">${r.selection_preview?`<img loading="lazy" src="${esc(r.selection_preview)}" alt="实际选区红色叠加预览">`:''}${r.diagnostic?`<img loading="lazy" src="${esc(r.diagnostic)}" alt="蓝色漏选、橙色多选核对图">`:''}</div><p>定位：${esc(r.method)}；修复：${esc(r.repair_mode)}。水印残留与识别正确率：未评估。</p><p class="record-id">${esc(r.key)}</p></details>${contexts}<div class="filelinks"><a href="${esc(r.original)}" target="_blank" rel="noopener">原始输入文件</a>${r.output?`<a href="${esc(r.output)}" target="_blank" rel="noopener">原尺寸处理结果</a>`:''}</div></div></article>`;
 }
 function render(reset=false){
  if(reset)state.page=1;const rows=entries();
  const query=$('search').value.trim().toLocaleLowerCase();
  const filtered=rows.filter(r=>(!$('style').value||r.style===$('style').value)&&(!$('group').value||r.group===$('group').value)&&($('negatives').checked||!r.negative)&&(!query||[r.id,r.label,r.style,...r.reasons].join(' ').toLocaleLowerCase().includes(query)));
  const fpOf=r=>Math.max(0,...r.occurrences.map(o=>o.fp_frac??0));
  const FP_MIN=0.002;
  for(const bucket of Object.keys(ranks))$('count-'+bucket).textContent=count(filtered.filter(r=>r.bucket===bucket).length);
  $('count-fp').textContent=count(filtered.filter(r=>!r.negative&&fpOf(r)>FP_MIN).length);
  const cmpRows=b=>filtered.filter(r=>r.cmp===b);
  if($('count-worse'))$('count-worse').textContent=count(cmpRows('worse').length);
  if($('count-better'))$('count-better').textContent=count(cmpRows('better').length);
  const isFp=state.bucket==='fp';
  const isWorse=state.bucket==='worse',isBetter=state.bucket==='better';
  const selected=isWorse?cmpRows('worse').sort((a,b)=>(a.d_recall??0)-(b.d_recall??0)||(a.d_mae??0)-(b.d_mae??0)):isBetter?cmpRows('better').sort((a,b)=>(b.d_recall??0)-(a.d_recall??0)||(b.d_mae??0)-(a.d_mae??0)):isFp?filtered.filter(r=>!r.negative&&fpOf(r)>FP_MIN).sort((a,b)=>fpOf(b)-fpOf(a)||(b.false_positive_pixels??0)-(a.false_positive_pixels??0)):filtered.filter(r=>r.bucket===state.bucket).sort((a,b)=>Number(a.negative)-Number(b.negative)||(state.bucket==='poor'?(b.background_damage_mae??0)-(a.background_damage_mae??0)||(a.watermark_error_reduction??0)-(b.watermark_error_reduction??0):(a.watermark_mae??0)-(b.watermark_mae??0)));
  const pages=Math.max(1,Math.ceil(selected.length/state.size));state.page=Math.min(state.page,pages);
  $('cards').innerHTML=selected.slice((state.page-1)*state.size,state.page*state.size).map(card).join('');
  $('empty').hidden=selected.length!==0;$('previous').disabled=state.page<=1;$('next').disabled=state.page>=pages;
  $('pagination').textContent=`第 ${state.page}／${pages} 页`;
  const negatives=selected.filter(r=>r.negative).length;
  $('view-note').textContent=isWorse?`较上次基线退化 ${selected.length} 张，按召回率降幅从大到小排列；每页 ${state.size} 条。`:isBetter?`较上次基线优化 ${selected.length} 张，按召回率升幅从大到小排列；每页 ${state.size} 条。`:isFp?`多选背景占比 >0.2% 共 ${selected.length} 张（带水印样本），按占比从大到小排列；每页 ${state.size} 条。`:`本档 ${selected.length} 条：带水印样本 ${selected.length-negatives} 条，无水印对照 ${negatives} 条；每页 ${state.size} 条。${$('duplicates').checked?'当前保留全部批次记录。':'同图重复批次已合并，默认展示较差档。'}`;
  if($('scope'))$('scope').textContent=`${modeNames[currentMode()]} · 原始记录 ${data.rows.filter(r=>r.mode===currentMode()).length} 条 · 当前口径 ${rows.length} 条${$('negatives').checked?'':' · 已隐藏 '+rows.filter(r=>r.negative).length+' 条无水印对照'}`;
  $('deduplication-note').textContent=`当前模式在移除指定样本后有 ${data.rows.filter(r=>r.mode===currentMode()).length} 条记录，按“模式＋输入文件哈希＋页码”合并为 ${unique(data.rows.filter(r=>r.mode===currentMode())).length} 个页面。快速20张与增量集重叠；同图批次结果不一致时显示较差档。勾选“显示重复批次记录”可查看当前范围全部批次。PDF与JPG属于不同输入。`;
  tabs.forEach(t=>{const active=t.dataset.bucket===state.bucket;t.setAttribute('aria-selected',String(active));t.tabIndex=active?0:-1;});
  $('effect-panel').setAttribute('aria-labelledby','tab-'+state.bucket);
 }
 for(const id of ['mode','style','group','duplicates','negatives']){const el=$(id);if(el)el.addEventListener('change',()=>render(true));}
 $('search').addEventListener('input',()=>render(true));
 tabs.forEach((tab,index)=>{tab.addEventListener('click',()=>{state.bucket=tab.dataset.bucket;render(true);});tab.addEventListener('keydown',e=>{let i;if(e.key==='ArrowRight')i=(index+1)%tabs.length;else if(e.key==='ArrowLeft')i=(index+tabs.length-1)%tabs.length;else if(e.key==='Home')i=0;else if(e.key==='End')i=tabs.length-1;else return;e.preventDefault();tabs[i].focus();tabs[i].click();});});
 $('previous').addEventListener('click',()=>{state.page--;render();$('effect-panel').scrollIntoView({block:'start'});});$('next').addEventListener('click',()=>{state.page++;render();$('effect-panel').scrollIntoView({block:'start'});});
 $('cards').addEventListener('click',e=>{const b=e.target.closest('[data-image]');if(!b)return;$('viewer-image').src=b.dataset.image;$('viewer-title').textContent=b.dataset.caption;$('viewer-link').href=b.dataset.image;$('viewer').showModal();});$('viewer-close').addEventListener('click',()=>$('viewer').close());
 const styleLabels=new Map(data.rows.map(r=>[r.style,r.label||r.style]));for(const [value,label] of [...styleLabels].sort((a,b)=>a[1].localeCompare(b[1],'zh-CN'))){const o=document.createElement('option');o.value=value;o.textContent=label;$('style').appendChild(o);}for(const value of [...new Set(data.rows.map(r=>r.group))]){const o=document.createElement('option');o.value=value;o.textContent=value;$('group').appendChild(o);}
 $('sources').innerHTML=`<a href="${data.sources.original_records}">封存逐图记录</a><a href="${data.sources.inventory}">原始样本清单</a><a href="${data.sources.rules}">本页初筛规则</a><a href="逐图效果初筛_20261029.json">本页逐图分档数据</a>`;
 if(data.manifest.sample_scope_revision){$('scope-revision').hidden=false;$('scope-revision').textContent=`按用户要求移除 ${data.manifest.sample_scope_revision.removed_ids.length} 张手写样本及其三模式记录。原始封存 ${data.manifest.source_record_count} 条，当前效果页保留 ${data.rows.length} 条；这是样本范围变更，不代表去水印效果提升。历史报告按当时范围保留。`;$('sources').innerHTML+='<a href="样本移除说明_20261029.json">本次样本移除说明</a>';}
 $('mode-summary').innerHTML=`<h3>全量模式统计（含无水印对照，不随页面筛选变化）</h3><table class="mode-table"><thead><tr><th>模式</th><th>较好初筛</th><th>一般／待复核</th><th>问题较多</th><th>较好档内无水印对照</th></tr></thead><tbody>`+presentModes.map(m=>`<tr><td>${modeNames[m]||m}</td><td>${data.counts[m].good||0}</td><td>${data.counts[m].review||0}</td><td>${data.counts[m].poor||0}</td><td>${data.negative_counts[m]||0}</td></tr>`).join('')+'</tbody></table>';
 render();
})();
