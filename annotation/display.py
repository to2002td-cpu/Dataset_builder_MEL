"""
HTML rendering of annotated instances for Jupyter (cards and compact tables).

Label badge colors reuse the shared fig_gen families (green = YES, pink = NO,
orange = UNCERTAIN) so HTML output and matplotlib figures match.

Usage (from the analysis notebook, cwd = annotation/):
    from display import show_cards, show_table

    show_cards(master[master.disagree], annotators, kb, title="Disagreements")
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "fig_gen")):
    if _p not in sys.path:
        sys.path.append(_p)

from fig_gen.utils import COLORS  # noqa: E402

LABEL_BADGE = {
    "YES": COLORS["green_edge"],
    "NO": COLORS["pink_edge"],
    "UNCERTAIN": COLORS["orange_edge"],
}


def _vote_badges(row: pd.Series, annotators: list[str]) -> str:
    out = []
    for u in annotators:
        v = row[u]
        if not isinstance(v, str):
            continue
        color = LABEL_BADGE.get(v, COLORS["gray_edge"])
        out.append(f'<span style="background:{color};color:#fff;border-radius:5px;'
                   f'padding:1px 7px;margin-right:5px;font-size:12px">{u}: {v}</span>')
    return " ".join(out)


def _candidate_strip(row: pd.Series, kb: dict[str, dict]) -> str:
    qids = row.get("candidate_qids") or []
    if len(qids) == 0:
        return ""
    chips = ""
    for q in qids:
        entity = kb.get(q, {})
        name = entity.get("name") or q
        img = entity.get("infobox_img") or ""
        gold = q == row["qid"]
        border = f'2px solid {COLORS["green_edge"]}' if gold else "1px solid #ddd"
        star = "★ " if gold else ""
        chips += (f'<div style="text-align:center;width:82px;margin:4px">'
                  f'<img src="{img}" style="width:76px;height:76px;object-fit:cover;'
                  f'border-radius:6px;border:{border}" onerror="this.style.opacity=.2">'
                  f'<div style="font-size:11px;color:#333;margin-top:2px;'
                  f'line-height:1.15">{star}{name}</div></div>')
    return ('<div style="margin-top:8px">'
            '<div style="font-size:11px;color:#888;margin-bottom:2px">'
            'Possible candidates (★ = gold):</div>'
            f'<div style="display:flex;flex-wrap:wrap">{chips}</div></div>')


def show_cards(df: pd.DataFrame, annotators: list[str], kb: dict[str, dict],
               img_w: int = 300, desc_len: int = 260, title: str | None = None) -> None:
    """Rich cards: image, mention → entity, vote badges, candidate strip."""
    if title:
        display(HTML(f"<h4 style='margin:8px 0'>{title} — {len(df)} cases</h4>"))
    if len(df) == 0:
        display(HTML("<i>No cases.</i>"))
        return
    html = ""
    for _, row in df.iterrows():
        desc = (row["desc"] or "")[:desc_len]
        html += f'''
        <div style="display:flex;gap:14px;border:1px solid #e5e5e5;border-radius:10px;
                    padding:12px;margin:10px 0;align-items:flex-start">
          <img src="{row['image_url']}" style="width:{img_w}px;border-radius:8px;flex:0 0 auto"
               onerror="this.style.opacity=.3;this.alt='image unavailable'">
          <div style="min-width:0">
            <div style="font-weight:700;font-size:15px">{row['mention']}
                 <span style="color:#666;font-weight:400">→ {row['entity_name']}
                 <span style="background:#eef;border-radius:4px;padding:0 6px">{
                    row['category']}</span></span></div>
            <div style="margin:6px 0">{_vote_badges(row, annotators)}</div>
            <div style="color:#444;font-size:13px">{desc}…</div>
            {_candidate_strip(row, kb)}
          </div>
        </div>'''
    display(HTML(html))


def show_table(df: pd.DataFrame, annotators: list[str],
               img_w: int = 90, desc_len: int = 110, title: str | None = None) -> None:
    """Compact table: thumbnail, mention, entity, category, votes, description."""
    if title:
        display(HTML(f"<h4 style='margin:8px 0'>{title} — {len(df)} cases</h4>"))
    if len(df) == 0:
        display(HTML("<i>No cases.</i>"))
        return
    head = "".join(f"<th style='text-align:left;padding:4px 8px'>{c}</th>"
                   for c in ["image", "mention", "entity", "cat.", "votes", "desc"])
    body = ""
    for _, row in df.iterrows():
        desc = (row["desc"] or "")[:desc_len]
        body += (f"<tr style='border-top:1px solid #eee'>"
                 f"<td style='padding:4px 8px'><img src='{row['image_url']}'"
                 f" style='width:{img_w}px;border-radius:4px'></td>"
                 f"<td style='padding:4px 8px;font-weight:600'>{row['mention']}</td>"
                 f"<td style='padding:4px 8px'>{row['entity_name']}</td>"
                 f"<td style='padding:4px 8px'>{row['category']}</td>"
                 f"<td style='padding:4px 8px'>{_vote_badges(row, annotators)}</td>"
                 f"<td style='padding:4px 8px;color:#555;font-size:12px'>{desc}…</td></tr>")
    display(HTML(f"<table style='border-collapse:collapse;font-size:13px'>"
                 f"<tr>{head}</tr>{body}</table>"))
