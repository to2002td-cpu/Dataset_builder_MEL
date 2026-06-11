#!/usr/bin/env python3
"""
Browse instances.jsonl (output of make_dataset.py) as a self-contained HTML file.

Each instance shows the disambiguation task:
  - Mention (ambiguous surface form) + Image → which entity?
  - Text candidates   : entities sharing the same surface form
  - Visual candidates : entities sharing the body image
  - The answer entity (text ∩ visual) is highlighted in red.

Usage:
    python scripts/view_split.py output/final/instances.jsonl
    python scripts/view_split.py output/final/instances.jsonl --kb output/final/kb.jsonl
    python scripts/view_split.py output/final/instances.jsonl --sample 300 --open
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>wikiambig — Dataset Viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
  display:flex;height:100vh;overflow:hidden;background:#f8fafc;color:#1e293b}

/* ── Sidebar ── */
#sidebar{width:260px;min-width:260px;background:#fff;
  border-right:1px solid #e2e8f0;display:flex;flex-direction:column}
#search-bar{padding:.6rem .7rem;border-bottom:1px solid #e2e8f0;
  display:flex;gap:.5rem;align-items:center}
#search{flex:1;padding:.35rem .6rem;border:1px solid #cbd5e1;
  border-radius:6px;font-size:.82rem;outline:none}
#search:focus{border-color:#3b82f6;box-shadow:0 0 0 2px #bfdbfe}
#count{font-size:.73rem;color:#94a3b8;white-space:nowrap}
#list{flex:1;overflow-y:auto}
.item{display:flex;gap:8px;align-items:center;padding:.45rem .7rem;
  cursor:pointer;border-bottom:1px solid #f1f5f9;transition:background .1s}
.item:hover{background:#f8fafc}
.item.active{background:#eff6ff;border-left:3px solid #3b82f6;
  padding-left:calc(.7rem - 3px)}
.item-thumb{width:40px;height:40px;object-fit:cover;border-radius:4px;
  flex-shrink:0;background:#e2e8f0}
.item-text{flex:1;min-width:0}
.item-mention{font-size:.83rem;font-weight:600;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.item-meta{font-size:.7rem;color:#94a3b8;margin-top:1px}

/* ── Main panel ── */
#main{flex:1;overflow-y:auto}
#detail{padding:1.4rem 1.6rem;max-width:1100px;margin:0 auto}

/* ── Task header ── */
.task-header{display:flex;gap:1.4rem;align-items:flex-start;
  background:#fff;border:1px solid #e2e8f0;border-radius:12px;
  padding:1rem 1.2rem;margin-bottom:1.2rem;box-shadow:0 1px 4px rgba(0,0,0,.05)}
.task-image-wrap{flex-shrink:0;text-align:center}
.task-img{max-height:200px;max-width:240px;border-radius:8px;
  box-shadow:0 2px 8px rgba(0,0,0,.12);display:block}
.task-img-meta{font-size:.68rem;color:#94a3b8;margin-top:.35rem;
  max-width:240px;word-break:break-word}
.task-mention-wrap{flex:1;display:flex;flex-direction:column;
  justify-content:center;gap:.5rem}
.task-label{font-size:.7rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.08em;color:#94a3b8}
.task-mention{font-size:2rem;font-weight:800;color:#0f172a;line-height:1.2}
.task-arrow{font-size:.85rem;color:#64748b;margin-top:.2rem}

/* ── Legend ── */
.legend{display:flex;gap:1rem;margin-bottom:.9rem;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:.3rem;font-size:.73rem;color:#475569}
.dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.dot-text{background:#93c5fd}.dot-visual{background:#86efac}.dot-both{background:#FF2500}

/* ── Candidate columns ── */
.cand-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem}
.cand-section h3{font-size:.76rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.08em;margin-bottom:.65rem;padding-bottom:.3rem;border-bottom:2px solid}
.cand-section.text  h3{color:#1d4ed8;border-color:#bfdbfe}
.cand-section.visual h3{color:#15803d;border-color:#bbf7d0}
.cards{display:flex;flex-direction:column;gap:.55rem}

/* ── Entity card ── */
.ecard{display:flex;gap:.65rem;padding:.65rem;border-radius:8px;border:2px solid}
.ecard.role-text  {border-color:#93c5fd;background:#f0f7ff}
.ecard.role-visual{border-color:#86efac;background:#f0fdf4}
.ecard.role-both  {border-color:#FF2500;background:#faf5ff}
.e-thumb{width:60px;height:60px;object-fit:cover;border-radius:5px;
  flex-shrink:0;background:#f1f5f9}
.e-body{flex:1;min-width:0}
.e-name{font-weight:700;font-size:.88rem;color:#0f172a}
.badge{display:inline-block;padding:.1em .38em;border-radius:4px;
  font-size:.65rem;font-weight:700;margin-left:.3em;vertical-align:middle}
.badge-pers {background:#dbeafe;color:#1e40af}
.badge-org  {background:#fef3c7;color:#92400e}
.badge-loc  {background:#d1fae5;color:#065f46}
.badge-other{background:#f1f5f9;color:#475569}
.badge-ans  {background:#FF2500;color:#fff}
.e-intro{font-size:.77rem;color:#475569;margin:.3rem 0;line-height:1.5;
  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.e-link{font-size:.73rem;color:#3b82f6;text-decoration:none}
.e-link:hover{text-decoration:underline}

.empty{padding:3rem;text-align:center;color:#94a3b8}
</style>
</head>
<body>
<div id="sidebar">
  <div id="search-bar">
    <input id="search" type="search" placeholder="Filter mention…">
    <span id="count"></span>
  </div>
  <div id="list"></div>
</div>
<div id="main"><div id="detail"></div></div>
<script>
const KB=__KB__;
const D=__DATA__;
let vis=[...D],cur=0;

function esc(s){
  return String(s||'')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function thumbUrl(raw,w){
  if(!raw)return'';
  const s=raw.replace('http://','https://');
  const m=s.match(/^https:\/\/upload\.wikimedia\.org\/wikipedia\/([a-z0-9_-]+)\//);
  if(m){
    const wiki=m[1];
    const host=wiki==='commons'?'commons.wikimedia.org':`${wiki}.wikipedia.org`;
    const fname=decodeURIComponent(s.split('/').pop());
    return`https://${host}/wiki/Special:FilePath/${encodeURIComponent(fname)}?width=${w}`;
  }
  if(s.includes('Special:FilePath/')){
    return s.includes('width=')?s:`${s}${s.includes('?')?'&':'?'}width=${w}`;
  }
  const fname=s.startsWith('File:')?s.slice(5):s;
  return`https://commons.wikimedia.org/wiki/Special:FilePath/${encodeURIComponent(fname)}?width=${w}`;
}

function role(qid,tqs,vqs){
  const t=tqs.has(qid),v=vqs.has(qid);
  return(t&&v)?'both':t?'text':'visual';
}

function entityCard(e,r,isAnswer){
  if(!e)return'';
  const src=thumbUrl(e.infobox_img,80);
  const imgHtml=src?`<img class="e-thumb" src="${esc(src)}" loading="lazy" alt="">`:'';
  const t=(e.type||'other').toLowerCase();
  const intro=esc((e.intro||e.desc||'').slice(0,300));
  const link=(e.url_wikipedia&&e.url_wikipedia.startsWith('http'))
    ?`<a class="e-link" href="${esc(e.url_wikipedia)}" target="_blank" rel="noopener">Wikipedia →</a>`
    :'';
  const ansBadge=isAnswer?`<span class="badge badge-ans">Answer</span>`:'';
  return`<div class="ecard role-${esc(r)}">
    ${imgHtml}
    <div class="e-body">
      <div><span class="e-name">${esc(e.name)}</span>
        <span class="badge badge-${esc(t)}">${esc(e.type||'OTHER')}</span>${ansBadge}</div>
      <p class="e-intro">${intro}</p>
      ${link}
    </div>
  </div>`;
}

function renderDetail(idx){
  const el=document.getElementById('detail');
  if(idx<0||idx>=vis.length){el.innerHTML='<div class="empty">No instance selected.</div>';return;}
  const inst=vis[idx];
  const img=inst.image;
  const imgSrc=thumbUrl(img.url,320);
  const meta=[
    img.n_used_by!=null?`Used by ${img.n_used_by} article(s)`:'',
    img.license||'',
    (img.width&&img.height)?`${img.width}×${img.height}px`:'',
  ].filter(Boolean).join(' · ');

  const tqs=new Set(inst.text_candidates);
  const vqs=new Set(inst.visual_candidates);
  const answer=inst.answer||'';

  const tCards=inst.text_candidates.map(qid=>entityCard(KB[qid],role(qid,tqs,vqs),qid===answer)).join('');
  const vCards=inst.visual_candidates.map(qid=>entityCard(KB[qid],role(qid,tqs,vqs),qid===answer)).join('');

  el.innerHTML=`
    <div class="task-header">
      <div class="task-image-wrap">
        <img class="task-img" src="${esc(imgSrc)}" loading="lazy" alt="">
        <div class="task-img-meta">${esc(meta)}</div>
      </div>
      <div class="task-mention-wrap">
        <span class="task-label">Mention</span>
        <div class="task-mention">${esc(inst.mention)}</div>
        <div class="task-arrow">Which entity does this (mention, image) pair refer to?</div>
      </div>
    </div>
    <div class="legend">
      <span class="legend-item"><span class="dot dot-text"></span>Text only</span>
      <span class="legend-item"><span class="dot dot-visual"></span>Visual only</span>
      <span class="legend-item"><span class="dot dot-both"></span>Answer (text ∩ visual)</span>
    </div>
    <div class="cand-grid">
      <div class="cand-section text">
        <h3>Text candidates (${inst.text_candidates.length})</h3>
        <div class="cards">${tCards}</div>
      </div>
      <div class="cand-section visual">
        <h3>Visual candidates (${inst.visual_candidates.length})</h3>
        <div class="cards">${vCards}</div>
      </div>
    </div>`;
}

function renderList(){
  document.getElementById('count').textContent=`${vis.length} / ${D.length}`;
  const el=document.getElementById('list');
  el.innerHTML=vis.map((inst,i)=>{
    const src=thumbUrl(inst.image.url,50);
    const thumb=src
      ?`<img class="item-thumb" src="${esc(src)}" loading="lazy" alt="">`
      :`<div style="width:40px;height:40px;border-radius:4px;background:#e2e8f0;flex-shrink:0"></div>`;
    const nc=inst.text_candidates.length,nv=inst.visual_candidates.length;
    return`<div class="item${i===cur?' active':''}" data-i="${i}">
      ${thumb}
      <div class="item-text">
        <div class="item-mention">${esc(inst.mention)}</div>
        <div class="item-meta">${nc}T · ${nv}V · ×${inst.image.n_used_by}</div>
      </div>
    </div>`;
  }).join('');
  el.querySelectorAll('.item').forEach(item=>{
    item.addEventListener('click',()=>select(+item.dataset.i));
  });
}

function select(idx){
  cur=idx;
  renderList();
  renderDetail(idx);
  const active=document.querySelector('#list .item.active');
  if(active)active.scrollIntoView({block:'nearest'});
}

document.getElementById('search').addEventListener('input',function(){
  const q=this.value.toLowerCase();
  vis=D.filter(x=>x.mention.toLowerCase().includes(q));
  cur=0;
  renderList();
  renderDetail(0);
});

document.addEventListener('keydown',function(e){
  if(document.activeElement===document.getElementById('search'))return;
  if(e.key==='ArrowDown'){e.preventDefault();select(Math.min(cur+1,vis.length-1));}
  if(e.key==='ArrowUp'){e.preventDefault();select(Math.max(cur-1,0));}
});

renderList();
renderDetail(0);
</script>
</body>
</html>
"""


def generate_html(instances: list[dict], kb: dict[str, dict]) -> str:
    kb_json   = json.dumps(kb, ensure_ascii=False, separators=(",", ":"))
    data_json = json.dumps(instances, ensure_ascii=False, separators=(",", ":"))
    return _HTML.replace("__KB__", kb_json).replace("__DATA__", data_json)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="instances.jsonl (make_dataset.py output)")
    ap.add_argument("--kb", type=Path, default=None,
                    help="kb.jsonl (default: auto-detect in same directory as input)")
    ap.add_argument("--sample", "-n", type=int, default=500, metavar="N",
                    help="max instances to embed (default: 500; 0 = all)")
    ap.add_argument("--out", "-o", type=Path, default=Path("split_viewer.html"))
    ap.add_argument("--open", action="store_true",
                    help="open the generated viewer in the default browser")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")

    kb_path = args.kb or (args.input.parent / "kb.jsonl")
    if not kb_path.exists():
        raise SystemExit(f"KB not found at {kb_path}. Pass --kb to specify its location.")

    # Load instances
    instances: list[dict] = []
    limit = args.sample if args.sample > 0 else None
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
                if limit and len(instances) >= limit:
                    break

    # Load only KB entries referenced by the sampled instances
    referenced: set[str] = set()
    for inst in instances:
        referenced.update(inst["text_candidates"])
        referenced.update(inst["visual_candidates"])

    kb: dict[str, dict] = {}
    with kb_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e["qid"] in referenced:
                kb[e["qid"]] = e

    html = generate_html(instances, kb)
    args.out.write_text(html, encoding="utf-8")
    print(f"Written {len(instances):,} instances, {len(kb):,} KB entries → {args.out}")

    if args.open:
        webbrowser.open(args.out.resolve().as_uri())


if __name__ == "__main__":
    main()
