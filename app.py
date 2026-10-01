"""Four trained models in one maintenance wrapper."""
import logging
import streamlit as st
from src.app.components import INFO, RENDERERS
from src.app.maintenance import VISUAL_CSS, NAMES, findings, summary_html
from src.app.service import SUBSYSTEMS, analyse, model_status, source_digest
from src.app.submission import predictions_zip

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Train checks", page_icon=":material/train:", layout="wide")

# Colour, radius and font come from .streamlit/config.toml via Streamlit's own
# theming API, so tabs, buttons and widgets pick them up without depending on
# internal DOM class names. The CSS below only does layout work the theme API
# cannot express: the header band, the status strip and the spacing rhythm.
st.markdown("""<style>
:root{
  --ink:#1B2A41; --ink-soft:#4A5A6E; --ink-faint:#7A8899;
  --line:#DCE1E8; --accent:#C8102E;
  --surface:#FFFFFF; --canvas:#F4F6F9;
  --shadow:0 1px 2px rgba(27,42,65,.06);
  --shadow-card:0 1px 3px rgba(27,42,65,.08),0 1px 2px rgba(27,42,65,.04);
}
.stApp{background:var(--canvas)}
.block-container{padding-top:1.2rem;padding-bottom:4rem;max-width:1180px}

/* Header band --------------------------------------------------------- */
.app-header{
  background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:20px 24px;margin-bottom:18px;box-shadow:var(--shadow-card);
  border-top:3px solid var(--accent);
}
.app-header h1{
  font-size:1.6rem;font-weight:700;letter-spacing:-.4px;margin:0 0 4px;color:var(--ink)
}
.app-header p{margin:0;color:var(--ink-soft);font-size:.92rem}

/* Status strip: four cards, one per subsystem ------------------------- */
.status-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px}
.status-card{
  background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:14px 16px;box-shadow:var(--shadow);border-left:3px solid var(--line);
}
.status-card.done{border-left-color:#067647}
.status-card.pending{border-left-color:#DCE1E8}
.status-card.down{border-left-color:#B42318}
.status-card .label{
  font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;
  color:var(--ink-faint);font-weight:700;margin-bottom:6px
}
.status-card .value{font-size:1.05rem;font-weight:650;color:var(--ink);line-height:1.3}
.status-card .sub{font-size:.78rem;color:var(--ink-faint);margin-top:3px}

/* Panels around the working area -------------------------------------- */
[data-testid="stFileUploader"]{background:var(--surface);border-radius:10px}
[data-testid="stExpander"]{background:var(--surface);border-radius:10px;box-shadow:var(--shadow)}

/* Section headings ---------------------------------------------------- */
.section-title{
  font-size:.78rem;text-transform:uppercase;letter-spacing:.6px;font-weight:700;
  color:var(--ink-faint);margin:4px 0 10px
}

/* Keep a visible focus ring for keyboard users ------------------------ */
button:focus-visible,summary:focus-visible,input:focus-visible{
  outline:3px solid var(--accent)!important;outline-offset:2px;border-radius:4px
}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}

@media(max-width:820px){
  .status-strip{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:640px){
  .block-container{padding-left:1rem;padding-right:1rem}
  .app-header{padding:16px}.app-header h1{font-size:1.3rem}
  .status-strip{grid-template-columns:1fr;gap:8px}
}
""" + VISUAL_CSS + "</style>", unsafe_allow_html=True)


@st.cache_resource
def backend_status():
    return {key: model_status(key) for key in SUBSYSTEMS}


def status_strip(status, results):
    """Four cards showing, at a glance, what has been checked and what is left.

    This is the difference between a form and a dashboard: the page says
    something before any file is uploaded.
    """
    cards = []
    for key in SUBSYSTEMS:
        name = NAMES[key]
        if not status[key][0]:
            cards.append(f'<div class="status-card down"><div class="label">{name}</div>'
                         f'<div class="value">Unavailable</div>'
                         f'<div class="sub">Model not loaded</div></div>')
        elif key in results:
            result = results[key]
            rows = len(result.table)
            noun = "segment" if key == "door" else "file"
            plural = "" if rows == 1 else "s"
            cards.append(f'<div class="status-card done"><div class="label">{name}</div>'
                         f'<div class="value">Checked</div>'
                         f'<div class="sub">{rows} {noun}{plural} · {len(result.source_files)} upload'
                         f'{"" if len(result.source_files) == 1 else "s"}</div></div>')
        else:
            cards.append(f'<div class="status-card pending"><div class="label">{name}</div>'
                         f'<div class="value">Not checked</div>'
                         f'<div class="sub">Awaiting upload</div></div>')
    st.markdown('<div class="status-strip">' + "".join(cards) + "</div>",
                unsafe_allow_html=True)


status = backend_status()
results = st.session_state.setdefault("results", {})

st.markdown(
    '<div class="app-header"><h1>Train condition checks</h1>'
    "<p>Upload recorded sensor data for a subsystem, run the check, "
    "and download the results.</p></div>",
    unsafe_allow_html=True,
)

# Reserve the strip's position now, but fill it in AFTER the tabs have run.
# The tabs are what populate `results`, so rendering here directly would show
# the previous run's state and a freshly checked subsystem would still read
# "Not checked" until the next interaction.
strip_slot = st.empty()

# All four upload widgets stay mounted so changing tabs preserves selections.
for tab, key in zip(st.tabs([NAMES[k] for k in SUBSYSTEMS]), SUBSYSTEMS):
    with tab:
        if not status[key][0]:
            st.error("This check is temporarily unavailable. Please contact the app administrator.")
            results.pop(key, None)
            continue

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown('<div class="section-title">1 · Upload</div>', unsafe_allow_html=True)
            st.write(INFO[key]["input"])
            uploads = st.file_uploader(f"{NAMES[key]} data", type=["xlsx"] if key == "acv" else ["csv"],
                accept_multiple_files=key != "door", key=f"uploads_{key}", label_visibility="collapsed", max_upload_size=64)
            uploads = ([uploads] if uploads is not None else []) if key == "door" else (uploads or [])
            names = [upload.name for upload in uploads]
            duplicates = len(names) != len(set(names))
            files = {upload.name: upload.getvalue() for upload in uploads}
            changed = key in results and (duplicates or not files or source_digest(files) != results[key].source_digest)
            if changed:
                results.pop(key, None)
                if files:
                    st.info("Files changed. Run the check again to update the result.")
            if duplicates:
                st.error("Two files have the same name. Remove or rename the duplicate.")
            run = st.button("Check data", type="primary", icon=":material/play_arrow:",
                            key=f"analyse_{key}", disabled=not files or duplicates,
                            use_container_width=True)
        with right:
            st.markdown('<div class="section-title">Selected</div>', unsafe_allow_html=True)
            if files:
                st.metric("Files ready", len(files))
                with st.expander(f"{len(files)} file{'' if len(files) == 1 else 's'}"):
                    for name in sorted(files):
                        st.caption(name)
            else:
                st.caption("No files selected yet.")

        if run:
            results.pop(key, None)
            bar = st.progress(0.0, text="Checking your data…")
            try:
                def update(current, total, name):
                    bar.progress(current/total, text=f"Checked {current} of {total}: {name}")
                results[key] = analyse(key, files, progress=update)
            except ValueError as exc:
                st.error(str(exc))
            finally:
                bar.empty()
        if key in results:
            result = results[key]
            st.divider()
            st.markdown('<div class="section-title">2 · Results</div>', unsafe_allow_html=True)
            st.caption("Checked: " + ", ".join(result.source_files))
            RENDERERS[key](result)
            st.download_button("Download results (CSV)", result.csv_bytes, result.filename, "text/csv",
                key=f"download_{key}", icon=":material/download:")
        del files, uploads

# Now that every tab has run, the strip reflects this run's results.
with strip_slot.container():
    status_strip(status, results)

if results:
    st.divider()
    st.markdown('<div class="section-title">3 · Report and submission</div>', unsafe_allow_html=True)
    report_col, zip_col = st.columns(2, gap="large")

    with report_col:
        st.subheader("Summary report")
        st.caption(f"{len(results)} of 4 subsystems checked. Unchecked systems are labelled in the report.")
        with st.expander("Preview summary"):
            for key in SUBSYSTEMS:
                st.write(f"**{NAMES[key]}**")
                if key in results:
                    for row in findings(results[key]):
                        st.text(f"{row['source']} · {row['status']}\n{row['finding']}\n{row['action']}")
                else:
                    st.write("Not checked")
        st.download_button("Download summary report", summary_html(results), "maintenance_summary.html", "text/html",
            type="primary", key="summary_report", icon=":material/download:", use_container_width=True)
        st.caption("Open the report to read or print it. Choose Print → Save as PDF for a PDF copy.")

    with zip_col:
        st.subheader("Prediction files")
        st.caption("Includes " + ", ".join(result.filename for result in results.values()))
        # predictions_zip validates every result and raises on a bad one. Build
        # it before rendering the control: calling it inside st.download_button
        # lets the exception escape and blanks the whole page, and this is the
        # one download that matters most.
        try:
            archive = predictions_zip(results)
        except ValueError as exc:
            st.error(f"Cannot build predictions.zip: {exc}")
        else:
            st.download_button("Download predictions.zip", archive, "predictions.zip",
                "application/zip", key="predictions_zip", icon=":material/folder_zip:",
                use_container_width=True)
            st.caption("One archive, one CSV per subsystem checked, at the top level.")
