"""HTML report renderer for annotation analysis results.

Single public function: render(annotators, s) -> str
It knows nothing about file paths or argument parsing.
"""

import json
from collections import Counter

from loader import AnnotatorData
from stats import AnnotationStats, interpret_kappa


def render(annotators: dict[str, AnnotatorData], s: AnnotationStats) -> str:
    """Return the complete HTML report as a string."""
    short_names = [annotators[aid].short_name for aid in s.annotator_ids]

    majority_counts    = Counter(r.majority for r in s.instance_results)
    unanimous_count    = sum(1 for r in s.instance_results if r.unanimous)
    disagreement_count = len(s.instance_results) - unanimous_count

    unanimous_yes = [r.instance_id for r in s.instance_results if r.unanimous and r.majority == "YES"]
    unanimous_no  = [r.instance_id for r in s.instance_results if r.unanimous and r.majority == "NO"]
    all_sorted    = sorted(s.instance_results, key=lambda r: r.agreement_pct)

    yes_counts = [annotators[aid].label_counts.get("YES", 0)       for aid in s.annotator_ids]
    no_counts  = [annotators[aid].label_counts.get("NO", 0)        for aid in s.annotator_ids]
    unc_counts = [annotators[aid].label_counts.get("UNCERTAIN", 0) for aid in s.annotator_ids]

    n_yes         = majority_counts.get("YES", 0)
    n_no          = majority_counts.get("NO", 0)
    n_no_majority = majority_counts.get("NO MAJORITY", 0)

    kpi_html = "".join([
        _kpi("Annotators",    str(len(s.annotator_ids))),
        _kpi("Instances",     str(len(s.all_instances))),
        _kpi("Mean κ",        str(s.mean_kappa), interpret_kappa(s.mean_kappa)),
        _kpi("Unanimous",     f"{unanimous_count} / {len(s.all_instances)}"),
        _kpi("Disagreements", f"{disagreement_count} / {len(s.all_instances)}"),
        _kpi("Majority YES",  str(n_yes)),
        _kpi("Majority NO",   str(n_no)),
        _kpi("No majority",   str(n_no_majority)),
    ])

    # ── Majority alignment table ─────────────────────────────────────────────
    majority_rows = "".join(
        f"""<tr>
          <td style="text-align:left;padding:8px 10px;font-weight:500;">{ms.short_name}</td>
          <td>{ms.voted_instances}</td>
          <td>{ms.agreed_with_majority}</td>
          <td>{_bar(ms.agreement_rate, '#2a78d6')}<span class="bar-label">{ms.agreement_rate}%</span></td>
          <td>{ms.outlier_count}</td>
          <td>{_bar(ms.outlier_rate, '#e34948')}<span class="bar-label">{ms.outlier_rate}%</span></td>
        </tr>"""
        for ms in s.majority_stats
    )

    # ── Bias table ───────────────────────────────────────────────────────────
    bias_rows = "".join(
        f"""<tr>
          <td style="text-align:left;padding:8px 10px;font-weight:500;">{b.short_name}</td>
          <td>{b.yes_rate}%</td>
          <td>{b.group_yes_rate}%</td>
          <td>{_bias_bar(b.bias)}</td>
        </tr>"""
        for b in s.annotator_biases
    )

    # ── Fragile instances ────────────────────────────────────────────────────
    if s.fragile_instances:
        fragile_rows = "".join(
            f"""<tr>
              <td style="text-align:left;font-family:monospace;padding:8px 10px;color:var(--muted);">{fi.instance_id}</td>
              <td>{_pill(fi.current_majority)}</td>
              <td style="text-align:left;font-size:12px;color:var(--text2);">{", ".join(fi.flipping_annotators)}</td>
            </tr>"""
            for fi in s.fragile_instances
        )
        fragile_html = f"""
  <table>
    <thead>
      <tr>
        <th style="text-align:left;">Instance</th>
        <th>Current majority</th>
        <th style="text-align:left;">Flips if removed</th>
      </tr>
    </thead>
    <tbody>{fragile_rows}</tbody>
  </table>"""
    else:
        fragile_html = '<p style="font-size:13px;color:var(--muted);">No fragile instances — the majority label is stable for all instances.</p>'

    # ── Comment correlation ──────────────────────────────────────────────────
    cc = s.comment_correlation
    comment_points_json = json.dumps(cc.points)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Annotation analysis report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>{_CSS}</style>
</head>
<body>

<h1>Annotation analysis report</h1>
<p class="subtitle">{len(s.all_instances)} instances · {len(s.annotator_ids)} annotators</p>

<div class="kpi-row">{kpi_html}</div>

<!-- MAJORITY VOTE -->
<div class="section">
  <h2>Majority vote result</h2>
  <p class="section-desc">
    A label wins when <strong>more than half</strong> of the annotators chose it
    (strict majority: <code>votes for label &gt; total voters ÷ 2</code>).
    A 2–2 tie among 4 annotators produces <em>no majority</em>.
  </p>
  <div class="chart-wrap" style="height:200px;"><canvas id="majorityChart"></canvas></div>
  <div class="legend">
    <span><span class="dot" style="background:#1baf7a;"></span>YES ({n_yes})</span>
    <span><span class="dot" style="background:#e34948;"></span>NO ({n_no})</span>
    <span><span class="dot" style="background:#888;"></span>No majority ({n_no_majority})</span>
  </div>
</div>

<!-- UNANIMOUS -->
<div class="section">
  <h2>Unanimous instances ({unanimous_count})</h2>
  <p class="section-desc">All {len(s.annotator_ids)} annotators assigned the same label.</p>
  <h3 class="sub-heading">Unanimous YES ({len(unanimous_yes)})</h3>
  <div class="instance-list" id="unanimousYes"></div>
  <h3 class="sub-heading">Unanimous NO ({len(unanimous_no)})</h3>
  <div class="instance-list" id="unanimousNo"></div>
</div>

<!-- LABEL DISTRIBUTION -->
<div class="section">
  <h2>Label distribution per annotator</h2>
  <p class="section-desc">
    Share of YES / NO / UNCERTAIN labels per annotator as a percentage of their own annotations.
    Large gaps may indicate divergent interpretations of the guidelines.
  </p>
  <div class="chart-wrap"><canvas id="distChart"></canvas></div>
  <div class="legend">
    <span><span class="dot" style="background:#1baf7a;"></span>YES</span>
    <span><span class="dot" style="background:#e34948;"></span>NO</span>
    <span><span class="dot" style="background:#eda100;"></span>UNCERTAIN</span>
  </div>
</div>

<!-- KAPPA HEATMAP -->
<div class="section">
  <h2>Cohen's κ — inter-annotator agreement</h2>
  <p class="section-desc">
    Cohen's kappa measures agreement between two annotators,
    <strong>corrected for the agreement expected by chance</strong>.
    Raw percentage agreement is misleading: two annotators who always pick the most common label
    would agree often without reading anything. Kappa removes that bias:<br><br>
    <code>κ = (P_observed − P_chance) / (1 − P_chance)</code><br><br>
    κ = 1 → perfect agreement · κ = 0 → chance level · κ &lt; 0 → systematic disagreement.
    Interpretation scale: Landis &amp; Koch (1977).
  </p>
  <table class="heatmap" id="heatmapTable"></table>
  <div class="legend" style="margin-top:10px;">
    <span><span class="dot" style="background:#E1F5EE;"></span>Almost perfect (&gt;0.8)</span>
    <span><span class="dot" style="background:#9FE1CB;"></span>Substantial (0.6–0.8)</span>
    <span><span class="dot" style="background:#B5D4F4;"></span>Moderate (0.4–0.6)</span>
    <span><span class="dot" style="background:#FAEEDA;"></span>Fair (0.2–0.4)</span>
    <span><span class="dot" style="background:#FCEBEB;"></span>Slight / Poor (&lt;0.2)</span>
  </div>
</div>

<!-- ANNOTATOR ALIGNMENT -->
<div class="section">
  <h2>Annotator alignment with majority vote</h2>
  <p class="section-desc">
    For each annotator: how often they voted the same as the crowd majority,
    and how often they were the <strong>only dissenter</strong> (outlier —
    every other annotator agreed on a different label). Sorted by agreement rate.
  </p>
  <table>
    <thead>
      <tr>
        <th style="text-align:left;">Annotator</th>
        <th>Instances voted</th>
        <th>Agreed with majority</th>
        <th style="min-width:160px;">Agreement rate</th>
        <th>Outlier count</th>
        <th style="min-width:130px;">Outlier rate</th>
      </tr>
    </thead>
    <tbody>{majority_rows}</tbody>
  </table>
</div>

<!-- ANNOTATOR BIAS -->
<div class="section">
  <h2>Annotator bias — YES/NO tendency</h2>
  <p class="section-desc">
    Each annotator's personal YES rate compared to the group average.
    A <strong>positive bias</strong> (bar to the right) means this annotator says YES
    more often than the group — they tend to <em>over-annotate</em>.
    A <strong>negative bias</strong> (bar to the left) means they say YES less often
    than the group — they lean towards NO or UNCERTAIN.
    A bias close to 0 means they match the group average.
  </p>
  <table>
    <thead>
      <tr>
        <th style="text-align:left;">Annotator</th>
        <th>Their YES rate</th>
        <th>Group average</th>
        <th style="min-width:200px;">Bias (their rate − group avg)</th>
      </tr>
    </thead>
    <tbody>{bias_rows}</tbody>
  </table>
</div>

<!-- FRAGILE INSTANCES -->
<div class="section">
  <h2>Fragile instances ({len(s.fragile_instances)})</h2>
  <p class="section-desc">
    An instance is <strong>fragile</strong> if removing just one annotator's vote
    would change the majority label (or eliminate any majority).
    These are the cases where the result depends on a single person's opinion —
    they should be reviewed carefully before being used as ground truth.
  </p>
  {fragile_html}
</div>

<!-- COMMENT CORRELATION -->
<div class="section">
  <h2>Comments vs. disagreement</h2>
  <p class="section-desc">
    Do annotators comment more on ambiguous instances?
    Each dot is one instance: its horizontal position is the agreement percentage,
    its vertical position is the number of annotators who left a comment.
    If comments cluster on the left (low agreement), annotators tend to comment
    precisely when they're unsure — which is a good sign.
    Instances with no comments appear on the bottom row.<br><br>
    Mean agreement on <strong>commented</strong> instances: <strong>{cc.mean_agreement_with_comment}%</strong>
    ({cc.commented_count} instances) ·
    on <strong>uncommented</strong> instances: <strong>{cc.mean_agreement_without_comment}%</strong>
    ({cc.uncommented_count} instances).
  </p>
  <div class="chart-wrap" style="height:280px;"><canvas id="commentChart"></canvas></div>
</div>

<!-- ALL INSTANCES -->
<div class="section">
  <h2>All instances</h2>
  <p class="section-desc">
    Every annotated instance sorted from most to least disagreement.
    Comments from annotators appear below rows with disagreement.
    <code>agreement % = (votes for majority label ÷ total voters) × 100</code>
  </p>
  <div style="overflow-x:auto;"><table id="allInstancesTable"></table></div>
</div>

<script>
const isDark     = matchMedia('(prefers-color-scheme: dark)').matches;
const muted      = '#898781';
const gridColor  = isDark ? '#2c2c2a' : '#e1e0d9';
const text2Color = isDark ? '#c3c2b7' : '#52514e';

const shortNames    = {json.dumps(short_names)};
const numAnnotators = shortNames.length;

// ── Majority vote donut ────────────────────────────────────────────────────
new Chart(document.getElementById('majorityChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['YES', 'NO', 'No majority'],
    datasets: [{{
      data: [{n_yes}, {n_no}, {n_no_majority}],
      backgroundColor: ['#1baf7a', '#e34948', '#888'],
      borderWidth: 0,
    }}],
  }},
  options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }} }},
}});

// ── Unanimous instance tags ────────────────────────────────────────────────
const unanimousYes = {json.dumps(unanimous_yes)};
const unanimousNo  = {json.dumps(unanimous_no)};

function instanceTag(id, variant) {{
  return '<span class="instance-tag instance-tag-' + variant + '">' + id + '</span>';
}}
document.getElementById('unanimousYes').innerHTML = unanimousYes.map(id => instanceTag(id, 'y')).join('');
document.getElementById('unanimousNo').innerHTML  = unanimousNo.map(id  => instanceTag(id, 'n')).join('');

// ── Label distribution stacked bar ────────────────────────────────────────
const yesCounts = {json.dumps(yes_counts)};
const noCounts  = {json.dumps(no_counts)};
const uncCounts = {json.dumps(unc_counts)};
const totals    = shortNames.map((_, i) => yesCounts[i] + noCounts[i] + uncCounts[i]);
const toPct     = counts => counts.map((v, i) => Math.round(v / totals[i] * 100));

new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{
    labels: shortNames,
    datasets: [
      {{ label: 'YES',       data: toPct(yesCounts), backgroundColor: '#1baf7a' }},
      {{ label: 'NO',        data: toPct(noCounts),  backgroundColor: '#e34948' }},
      {{ label: 'UNCERTAIN', data: toPct(uncCounts), backgroundColor: '#eda100' }},
    ],
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.raw + '%' }} }},
    }},
    scales: {{
      x: {{ stacked: true, ticks: {{ color: muted }},    grid: {{ display: false }} }},
      y: {{ stacked: true, max: 100, ticks: {{ color: muted, callback: v => v + '%' }}, grid: {{ color: gridColor }} }},
    }},
  }},
}});

// ── Kappa heatmap ─────────────────────────────────────────────────────────
const kappaMatrix = {json.dumps(s.kappa_matrix)};

function kappaBg(v) {{
  if (v === null) return 'transparent';
  if (v >= 0.8)   return isDark ? '#085041' : '#E1F5EE';
  if (v >= 0.6)   return isDark ? '#0F6E56' : '#9FE1CB';
  if (v >= 0.4)   return isDark ? '#185FA5' : '#B5D4F4';
  if (v >= 0.2)   return isDark ? '#854F0B' : '#FAEEDA';
  return isDark ? '#791F1F' : '#FCEBEB';
}}
function kappaFg(v) {{
  if (v >= 0.8) return isDark ? '#9FE1CB' : '#04342C';
  if (v >= 0.6) return isDark ? '#E1F5EE' : '#085041';
  if (v >= 0.4) return isDark ? '#E6F1FB' : '#042C53';
  if (v >= 0.2) return isDark ? '#FAEEDA' : '#412402';
  return isDark ? '#FCEBEB' : '#501313';
}}

let heatHtml = '<tr><td></td>' + shortNames.map(s => '<th>' + s + '</th>').join('') + '</tr>';
for (let i = 0; i < numAnnotators; i++) {{
  heatHtml += '<tr><td>' + shortNames[i] + '</td>';
  for (let j = 0; j < numAnnotators; j++) {{
    const v       = kappaMatrix[i][j];
    const display = v === null ? '' : v === 1 ? '1.00' : v.toFixed(2);
    const color   = v === 1 ? muted : kappaFg(v);
    heatHtml += '<td style="background:' + kappaBg(v) + ';color:' + color + ';">' + display + '</td>';
  }}
  heatHtml += '</tr>';
}}
document.getElementById('heatmapTable').innerHTML = heatHtml;

// ── Comment vs disagreement scatter ───────────────────────────────────────
const commentPoints = {comment_points_json};

new Chart(document.getElementById('commentChart'), {{
  type: 'scatter',
  data: {{
    datasets: [
      {{
        label: 'With comment',
        data: commentPoints.filter(p => p.has_comment).map(p => ({{ x: p.agreement_pct, y: p.comment_count, id: p.instance_id }})),
        backgroundColor: '#2a78d6',
        pointRadius: 6,
      }},
      {{
        label: 'No comment',
        data: commentPoints.filter(p => !p.has_comment).map(p => ({{ x: p.agreement_pct, y: 0, id: p.instance_id }})),
        backgroundColor: isDark ? '#444' : '#ccc',
        pointRadius: 5,
      }},
    ],
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => ctx.raw.id + ' — ' + ctx.raw.x + '% agreement, ' + ctx.raw.y + ' comment(s)',
        }},
      }},
    }},
    scales: {{
      x: {{ min: 0, max: 100, title: {{ display: true, text: 'Agreement %', color: muted }}, ticks: {{ color: muted, callback: v => v + '%' }}, grid: {{ color: gridColor }} }},
      y: {{ min: -0.3, title: {{ display: true, text: 'Number of comments', color: muted }}, ticks: {{ color: muted, stepSize: 1 }}, grid: {{ color: gridColor }} }},
    }},
  }},
}});

// ── All instances table ────────────────────────────────────────────────────
function pill(label) {{
  if (!label || label === 'NO MAJORITY') return '<span class="pill-empty">—</span>';
  const cls    = label === 'YES' ? 'pill-y' : label === 'NO' ? 'pill-n' : 'pill-u';
  const letter = label === 'UNCERTAIN' ? '?' : label[0];
  return '<span class="pill ' + cls + '">' + letter + '</span>';
}}
function agreementBar(pct) {{
  const color = pct >= 100 ? '#1baf7a' : pct >= 80 ? '#2a78d6' : pct >= 60 ? '#eda100' : '#e34948';
  return '<div class="bar-bg"><div class="bar-fill" style="width:' + pct + '%;background:' + color + ';"></div></div>';
}}

const allInstances = {json.dumps([_row_to_js(r, short_names) for r in all_sorted])};
let html = '<tr>'
  + '<th style="text-align:left;">Instance</th>'
  + shortNames.map(s => '<th>' + s + '</th>').join('')
  + '<th>Majority</th><th style="min-width:80px;">Agreement</th></tr>';

allInstances.forEach((row, idx) => {{
  const bg = idx % 2 ? (isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)') : 'transparent';
  html += '<tr style="background:' + bg + ';">';
  html += '<td style="text-align:left;font-family:monospace;color:' + muted + ';padding:6px 8px;">' + row.instance_id + '</td>';
  row.votes.forEach(l => html += '<td>' + pill(l) + '</td>');
  html += '<td>' + pill(row.majority) + '</td>';
  html += '<td>' + agreementBar(row.agreement_pct) + '<span style="font-size:11px;color:' + muted + ';">' + row.agreement_pct + '%</span></td>';
  html += '</tr>';
  if (!row.unanimous && row.comments.length > 0) {{
    html += '<tr style="background:' + bg + ';"><td colspan="' + (shortNames.length + 3) + '" style="padding:2px 8px 10px;">';
    row.comments.forEach(c => {{
      html += '<div style="font-size:12px;color:' + text2Color + ';padding:2px 0;"><span style="color:' + muted + ';font-size:11px;">' + c.annotator + ':</span> ' + c.text.replace(/</g, '&lt;') + '</div>';
    }});
    html += '</td></tr>';
  }}
}});
document.getElementById('allInstancesTable').innerHTML = html;
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _row_to_js(result, short_names: list[str]) -> dict:
    return {
        "instance_id":   result.instance_id,
        "votes":         [result.votes.get(s, "") for s in short_names],
        "majority":      result.majority,
        "agreement_pct": result.agreement_pct,
        "unanimous":     result.unanimous,
        "comments":      result.comments,
    }


def _kpi(label: str, value: str, subtitle: str = "") -> str:
    sub = f'<div class="kpi-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{sub}</div>'


def _pill(label: str) -> str:
    if not label or label == "NO MAJORITY":
        return '<span class="pill-empty">—</span>'
    cls    = "pill-y" if label == "YES" else "pill-n" if label == "NO" else "pill-u"
    letter = "?" if label == "UNCERTAIN" else label[0]
    return f'<span class="pill {cls}">{letter}</span>'


def _bar(pct: float, color: str, width: int = 100) -> str:
    return (
        f'<div class="bar-bg" style="width:{width}px;display:inline-block;vertical-align:middle;margin-right:6px;">'
        f'<div class="bar-fill" style="width:{pct}%;background:{color};"></div></div>'
    )


def _bias_bar(bias: float) -> str:
    """Diverging bar centred at 0: positive = right (blue), negative = left (red)."""
    clamped = max(-50.0, min(50.0, bias))   # cap at ±50 pp for display
    half    = 50                             # half-width of the bar in px
    if bias >= 0:
        fill = f'width:{abs(clamped)}%;background:#2a78d6;margin-left:50%;'
    else:
        fill = f'width:{abs(clamped)}%;background:#e34948;margin-left:{50 + clamped}%;'
    sign = "+" if bias > 0 else ""
    return (
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<div class="bar-bg" style="width:200px;display:inline-block;position:relative;">'
        f'<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--border);"></div>'
        f'<div class="bar-fill" style="{fill}"></div></div>'
        f'<span class="bar-label">{sign}{bias} pp</span></div>'
    )


_CSS = """\

  :root { --bg: #fcfcfb; --card: #fff; --text: #0b0b0b; --text2: #52514e; --muted: #898781; --border: #e1e0d9; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #1a1a19; --card: #222; --text: #fff; --text2: #c3c2b7; --muted: #898781; --border: #333; }
  }
  * { margin: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); padding: 2rem; max-width: 1100px; margin: 0 auto; line-height: 1.6; }
  h1 { font-size: 22px; font-weight: 500; margin-bottom: 0.25rem; }
  h2 { font-size: 18px; font-weight: 500; margin: 2rem 0 0.4rem; }
  .subtitle     { font-size: 13px; color: var(--muted); margin-bottom: 1.5rem; }
  .section-desc { font-size: 13px; color: var(--text2); margin-bottom: 1rem; max-width: 720px; line-height: 1.7; }
  .section-desc code { background: var(--card); border: 0.5px solid var(--border); border-radius: 4px; padding: 1px 6px; font-size: 12px; }
  .sub-heading  { font-size: 14px; color: var(--text2); margin: 12px 0 4px; font-weight: 500; }
  .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin: 1.5rem 0 2rem; }
  .kpi     { background: var(--card); border: 0.5px solid var(--border); border-radius: 8px; padding: 1rem; text-align: center; }
  .kpi-label { font-size: 12px; color: var(--muted); }
  .kpi-value { font-size: 24px; font-weight: 500; }
  .kpi-sub   { font-size: 11px; color: var(--muted); }
  .chart-wrap { position: relative; width: 100%; height: 320px; margin-bottom: 1rem; }
  .legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: var(--text2); margin-bottom: 1.5rem; }
  .legend span { display: flex; align-items: center; gap: 4px; }
  .legend .dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
  table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; }
  th { text-align: center; padding: 8px 4px; color: var(--text2); font-weight: 500; font-size: 12px; border-bottom: 1px solid var(--border); }
  th:first-child { text-align: left; }
  td { padding: 6px 4px; text-align: center; vertical-align: middle; }
  .pill { padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 500; white-space: nowrap; display: inline-block; }
  .pill-y { background: #E1F5EE; color: #04342C; }
  .pill-n { background: #FCEBEB; color: #501313; }
  .pill-u { background: #FAEEDA; color: #412402; }
  .pill-empty { color: var(--muted); }
  @media (prefers-color-scheme: dark) {
    .pill-y { background: #085041; color: #5DCAA5; }
    .pill-n { background: #791F1F; color: #F09595; }
    .pill-u { background: #854F0B; color: #FAC775; }
  }
  .heatmap td { padding: 12px 6px; border-radius: 4px; font-weight: 500; }
  .heatmap td:first-child { text-align: right; padding-right: 10px; font-weight: 400; color: var(--text2); }
  .section { margin-bottom: 2.5rem; }
  .instance-list { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
  .instance-tag { font-family: monospace; font-size: 12px; padding: 2px 8px; border-radius: 4px; }
  .instance-tag-y { background: #E1F5EE; color: #04342C; }
  .instance-tag-n { background: #FCEBEB; color: #501313; }
  @media (prefers-color-scheme: dark) {
    .instance-tag-y { background: #085041; color: #5DCAA5; }
    .instance-tag-n { background: #791F1F; color: #F09595; }
  }
  .bar-bg   { background: var(--border); border-radius: 3px; height: 8px; width: 100%; }
  .bar-fill { height: 8px; border-radius: 3px; }
  .bar-label { font-size: 11px; color: var(--muted); }"""
