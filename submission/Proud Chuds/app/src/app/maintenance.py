"""Plain-language findings shared by the screen and downloadable report."""
from datetime import datetime, timezone
from html import escape

NAMES = {"door": "Doors", "acv": "Air conditioning", "rail": "Rail condition", "shm": "Structural health"}
PALETTE = {
    "attention": ("#B42318", "#FEF3F2", "!"),
    "normal": ("#067647", "#ECFDF3", "✓"),
    "review": ("#92400E", "#FFFBEB", "?"),
    "neutral": ("#475467", "#F2F4F7", "—"),
}


def findings(result):
    """One traceable finding per input, without inventing fault thresholds."""
    rows = []
    for name in result.source_files:
        detail = result.diagnostics["by_file"].get(name, {})
        if result.subsystem == "door":
            table = result.table
            count = int(table.prediction.eq("Abnormal resistance").sum())
            ambiguous = detail.get("ambiguous_operations", 0)
            rows.append(dict(source=name, status="Inspect" if count else ("Review" if ambiguous else "No issue detected"),
                tone="attention" if count else ("review" if ambiguous else "normal"),
                finding=f"{count} of {len(table)} door cycles show abnormal resistance.",
                action="Inspect the door mechanism at the flagged cycle times." if count else "Continue routine door checks.",
                note=(f"{ambiguous} cycles have unclear opening/closing flags; review the recording. " if ambiguous else "") +
                     "This recording does not identify a car or door number."))
        elif result.subsystem == "acv":
            ranked = result.table.loc[result.table.file_id.eq(name), "ranked_cars"].iloc[0].split("|")
            rows.append(dict(source=name, status="Inspect first", tone="attention",
                finding=f"Car {ranked[0]} is the first priority for a refrigerant-leak check.",
                action=f"Check Car {ranked[0]} first, then follow the car inspection order if needed.",
                note="Inspection order: " + " → ".join(ranked) + ". This is a relative ranking; lower-ranked cars are not confirmed healthy."))
        elif result.subsystem == "rail":
            label = result.table.loc[result.table.file_id.eq(name), "prediction"].iloc[0]
            low = bool(detail.get("low_speed_rule"))
            rows.append(dict(source=name,
                status="Check again" if low else ("No issue detected" if label == "Normal" else "Inspect"),
                tone="review" if low else ("normal" if label == "Normal" else "attention"),
                finding="Speed was too low to assess rail corrugation." if low else (
                    "No corrugation pattern detected." if label == "Normal" else f"Corrugation pattern detected on {label}."),
                action="Review a suitable recording at operating speed." if low else (
                    "Continue routine rail checks." if label == "Normal" else f"Prioritise inspection of rail {label}."),
                note="CSV result is Normal under the low-speed rule; the classifier was not run." if low else
                     "The recording does not identify a track location."))
        else:
            value = float(result.table.loc[result.table.file_id.eq(name), "prediction"].iloc[0])
            rows.append(dict(source=name, status="Engineering review", tone="review",
                finding=f"Estimated cumulative fatigue damage: {value:.6g}.",
                action="Refer the estimate to engineering for comparison with approved limits.",
                note="Estimate for this recording only. No pass/fail limit or remaining-life estimate is supplied."))
    return rows


def badge(text, tone):
    colour, background, symbol = PALETTE[tone]
    return f'<span class="status" style="color:{colour};background:{background}">{symbol} {escape(text)}</span>'


def train_view(cars):
    """Responsive coach symbols with connected, textual status callouts."""
    items = []
    for label, status, tone in cars:
        colour, background, symbol = PALETTE[tone]
        items.append(f'''<div class="coach-item" style="--coach:{colour};--tint:{background}">
          <div class="coach-callout">{symbol} {escape(status)}</div><div class="coach-stem"></div>
          <svg viewBox="0 0 180 90" role="img" aria-label="{escape(label)}: {escape(status)}">
            <rect x="5" y="5" width="170" height="66" rx="16" fill="{background}" stroke="{colour}" stroke-width="3"/>
            <path d="M0 52h5m170 0h5" stroke="{colour}" stroke-width="3"/>
            <rect x="18" y="18" width="28" height="21" rx="4" fill="{colour}" opacity=".22"/>
            <rect x="55" y="18" width="28" height="21" rx="4" fill="{colour}" opacity=".22"/>
            <rect x="92" y="18" width="28" height="21" rx="4" fill="{colour}" opacity=".22"/>
            <rect x="132" y="17" width="27" height="46" rx="4" fill="none" stroke="{colour}" stroke-width="2"/>
            <circle cx="38" cy="76" r="9" fill="{colour}"/><circle cx="141" cy="76" r="9" fill="{colour}"/>
            <text x="20" y="59" fill="{colour}" font-family="Arial,sans-serif" font-size="14" font-weight="700">{escape(label)}</text>
          </svg></div>''')
    return '<div class="train-view">' + ''.join(items) + '</div>'


VISUAL_CSS = '''
.train-view{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:20px 14px;margin:20px 0}
.coach-item{max-width:240px;width:100%;margin:auto}
.coach-callout{border:1px solid var(--coach);border-radius:7px;padding:7px 5px;text-align:center;color:var(--coach);background:var(--tint);font:600 14px Arial,sans-serif}
.coach-stem{height:17px;width:1px;background:var(--coach);margin:auto}
.coach-item svg{display:block;width:100%}
.status{display:inline-block;border-radius:6px;padding:5px 9px;font-size:14px;font-weight:700}
@media(max-width:640px){.train-view{grid-template-columns:repeat(2,minmax(0,1fr))}}
'''


def summary_html(results):
    """Self-contained, printable report; generated only from validated results."""
    for result in results.values():
        errors = result.validation_errors()
        if errors:
            raise ValueError("Cannot create a report: " + " ".join(errors))
    created = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    sections = []
    for key, title in NAMES.items():
        result = results.get(key)
        if result is None:
            sections.append(f'<section><h2>{title}</h2>{badge("Not checked", "neutral")}<p>No current result supplied.</p></section>')
            continue
        entries = []
        for row in findings(result):
            entries.append(f'''<article><h3>{escape(row['source'])}</h3>{badge(row['status'], row['tone'])}
            <p><strong>{escape(row['finding'])}</strong></p><p>{escape(row['action'])}</p>
            <p class="muted">{escape(row['note'])}</p></article>''')
        table = result.table.to_html(index=False, escape=True, border=0)
        sections.append(f'''<section><h2>{title}</h2><p class="muted">{len(result.source_files)} file(s) · Checked {escape(result.created_at)}</p>
        {''.join(entries)}<details><summary>Full results</summary><div class="table-wrap">{table}</div></details></section>''')
    return (f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Maintenance summary</title><style>{VISUAL_CSS}
    *{{box-sizing:border-box}}body{{font:16px/1.55 Arial,sans-serif;color:#182B3A;max-width:1000px;margin:36px auto;padding:0 24px;background:#fff}}
    h1{{font-size:30px;margin-bottom:8px}}h2{{font-size:22px}}h3{{font-size:16px;overflow-wrap:anywhere}}p{{margin:9px 0}}
    .muted{{color:#475467;font-size:14px}}section{{border-top:1px solid #D0D5DD;padding:18px 0}}article{{border-left:3px solid #D0D5DD;padding:4px 18px;margin:18px 0;break-inside:avoid}}
    .table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{padding:8px;border-bottom:1px solid #D0D5DD;text-align:left;overflow-wrap:anywhere}}
    summary{{cursor:pointer;padding:8px 0}}@media print{{body{{margin:0;max-width:none}}.print-help{{display:none}}}}
    </style></head><body><h1>Maintenance summary</h1><p class="muted">Created {created} · {len(results)} of 4 subsystems checked</p>
    <p>Findings are grouped by source recording; uploads are not assumed to describe the same train or inspection.</p>
    <p class="muted print-help">Use your browser’s Print → Save as PDF to save or print this report. Expand Full results first if required.</p>
    {''.join(sections)}<footer class="muted">Model findings support inspection planning. “No issue detected” is not a release-to-service approval.</footer></body></html>''').encode("utf-8")

# --- Evidence strength -----------------------------------------------------
#
# The door model is a threshold on integrated motor current. How far a cycle
# sits from that threshold is the evidence for its call, and it varies enormously:
# on the official test stream, flagged cycles range from +0.04% to +36.5% over.
# Presenting a +0.04% exceedance with the same certainty as a +36.5% one sends a
# technician out on a coin flip, so the margin is surfaced rather than hidden.
#
# MARGINAL_BAND is a presentation choice, not a fitted quantity. It does not
# change any prediction; the CSV is unaffected. The margin is a distance from a
# fitted threshold and is deliberately NOT called a probability or a confidence
# percentage, because the model cannot support that claim.

MARGINAL_BAND = 5.0  # percent either side of the threshold


def margin_pct(current_sum, threshold):
    """How far a cycle sits above (+) or below (-) its operation's threshold."""
    if not threshold:
        return 0.0
    return (float(current_sum) / float(threshold) - 1.0) * 100.0


def evidence_label(margin):
    """Plain words for how firm a call is. Never a fabricated probability."""
    if abs(margin) < MARGINAL_BAND:
        return "Borderline"
    return "Clear"


def evidence_note(margin, prediction):
    """One sentence a technician can act on."""
    if abs(margin) >= MARGINAL_BAND:
        if prediction == "Abnormal resistance":
            return f"Well above the threshold ({margin:+.1f}%)."
        return f"Well below the threshold ({margin:+.1f}%)."
    if prediction == "Abnormal resistance":
        return f"Only just over the threshold ({margin:+.1f}%). Verify before acting."
    return f"Only just under the threshold ({margin:+.1f}%). Worth a look while you are there."
