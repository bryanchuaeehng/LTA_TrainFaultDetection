"""Technician-facing results, with detail available only when requested."""
import pandas as pd
import streamlit as st
from src.app.maintenance import (MARGINAL_BAND, NAMES, badge, evidence_label,
                                  evidence_note, findings, margin_pct, train_view)

INFO = {
    "door": {"title": NAMES["door"], "input": "Upload one door recording (.csv)."},
    "acv": {"title": NAMES["acv"], "input": "Upload air-conditioning recordings (.xlsx), each containing all eight cars."},
    "rail": {"title": NAMES["rail"], "input": "Upload one or more rail vibration recordings (.csv)."},
    "shm": {"title": NAMES["shm"], "input": "Upload one or more stress recordings (.csv, one column without a header)."},
}


def show_finding(row):
    st.markdown(badge(row["status"], row["tone"]), unsafe_allow_html=True)
    st.write(row["finding"])
    st.write(row["action"])
    st.caption(row["note"])


def show_door(result):
    row = findings(result)[0]
    show_finding(row)
    st.markdown(train_view([("Door recording", row["status"], row["tone"])]), unsafe_allow_html=True)

    detail = next(iter(result.diagnostics["by_file"].values()))
    cycles = pd.DataFrame(detail["cycles"])
    cycles["margin"] = [margin_pct(c, t) for c, t in zip(cycles["current_sum"], cycles["threshold"])]
    cycles["Evidence"] = cycles["margin"].map(evidence_label)
    cycles["Cycle"] = range(1, len(cycles) + 1)

    flagged = cycles[cycles.prediction.eq("Abnormal resistance")].sort_values("margin", ascending=False)
    watch = cycles[cycles.prediction.eq("Normal") & cycles.margin.gt(-MARGINAL_BAND)].sort_values("margin", ascending=False)

    if len(flagged):
        st.write("**Cycles to inspect** — strongest evidence first")
        st.dataframe(
            flagged.assign(**{"Over threshold": flagged["margin"]})[
                ["Cycle", "start_time", "end_time", "operation", "Over threshold", "Evidence"]
            ].rename(columns={"start_time": "Start time", "end_time": "End time", "operation": "Movement"}),
            column_config={"Over threshold": st.column_config.NumberColumn(format="%+.1f%%")},
            hide_index=True, width="stretch")
        borderline = int((flagged["margin"].abs() < MARGINAL_BAND).sum())
        if borderline:
            st.caption(f"{borderline} of these sit within {MARGINAL_BAND:.0f}% of the threshold. "
                       "Verify those before raising work.")

    if len(watch):
        st.write(f"**Worth a look** — {len(watch)} cycle(s) close to the threshold but not flagged")
        st.dataframe(
            watch.assign(**{"Below threshold": watch["margin"]})[
                ["Cycle", "start_time", "operation", "Below threshold"]
            ].rename(columns={"start_time": "Start time", "operation": "Movement"}),
            column_config={"Below threshold": st.column_config.NumberColumn(format="%+.1f%%")},
            hide_index=True, width="stretch")
        st.caption("Not faults. Listed because they are close enough that a check while you are "
                   "at the door costs little.")

    display = cycles[["Cycle", "start_time", "end_time", "operation", "prediction", "margin", "Evidence"]].rename(
        columns={"start_time": "Start time", "end_time": "End time", "operation": "Movement",
                 "prediction": "Finding", "margin": "Margin"})
    with st.expander("All door cycles"):
        st.dataframe(display, column_config={"Margin": st.column_config.NumberColumn(format="%+.1f%%")},
                     hide_index=True, width="stretch")
        st.caption("Margin is the distance from this movement's fitted current threshold. "
                   "It is not a probability, and it does not change the submitted result.")


def show_acv(result):
    selected = st.selectbox("Recording", result.source_files, key="acv_recording") if len(result.source_files) > 1 else result.source_files[0]
    row = next(row for row in findings(result) if row["source"] == selected)
    # Keep the ranking caveat next to the visual, without repeating the full order.
    st.markdown(badge(row["status"], row["tone"]), unsafe_allow_html=True)
    st.write(row["finding"])
    ranked = result.table.loc[result.table.file_id.eq(selected), "ranked_cars"].iloc[0].split("|")
    cars = sorted(ranked)
    st.markdown(train_view([(f"Car {car}", "Inspect first" if car == ranked[0] else f"Priority {ranked.index(car)+1}",
                            "attention" if car == ranked[0] else "neutral") for car in cars]), unsafe_allow_html=True)
    st.caption("Cars shown by ID, not physical position. Grey means lower priority, not confirmed healthy.")
    with st.expander("Car inspection order"):
        st.dataframe(pd.DataFrame({"Priority": range(1, len(ranked)+1), "Car": ranked}), hide_index=True, width="stretch")
        st.caption("This ranking suggests where to check first; it does not confirm a refrigerant leak.")


def show_rail(result):
    rows = findings(result)
    flagged = [r for r in rows if r["tone"] == "attention"]
    recheck = [r for r in rows if r["tone"] == "review"]
    normal = [r for r in rows if r["tone"] == "normal"]
    for col, label, count in zip(st.columns(3), ["Inspect", "No issue detected", "Check again"], [len(flagged), len(normal), len(recheck)]):
        col.metric(label, count)
    if len(rows) == 1:
        show_finding(rows[0])
    else:
        if flagged:
            st.write("**Recordings to inspect**")
            st.dataframe(pd.DataFrame([{"Recording": r["source"], "Finding": r["finding"], "Next step": r["action"]} for r in flagged]), hide_index=True, width="stretch")
        if recheck:
            st.warning(f"{len(recheck)} recording(s) were too slow to assess. Review a suitable recording at operating speed.")
        with st.expander("All rail recordings"):
            st.dataframe(pd.DataFrame([{"Recording": r["source"], "Status": r["status"], "Finding": r["finding"]} for r in rows]), hide_index=True, width="stretch")
        st.caption("Recording names identify the inputs; track locations are not supplied.")


def show_shm(result):
    st.markdown(badge("Engineering review", "review"), unsafe_allow_html=True)
    st.write("Refer these fatigue estimates to engineering for comparison with approved limits.")
    st.dataframe(result.table.rename(columns={"file_id": "Recording", "prediction": "Estimated fatigue damage"}),
                 column_config={"Estimated fatigue damage": st.column_config.NumberColumn(format="%.6g")},
                 hide_index=True, width="stretch")
    st.caption("Each value covers its recording only. No pass/fail limit or remaining-life estimate is supplied.")


RENDERERS = {"door": show_door, "acv": show_acv, "rail": show_rail, "shm": show_shm}
