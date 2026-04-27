import os
from datetime import datetime

import pandas as pd
import streamlit as st

from building_project_predictor import BuildingPredictor

st.set_page_config(
    page_title="Building Project Prediction Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(__file__)
EXCEL_PATH = os.path.join(BASE_DIR, "D_Building_2000_prediction_dataset.xlsx")

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1250px;}
.main-title {font-size: 2.35rem; font-weight: 850; letter-spacing: -0.04em; margin-bottom: .2rem; color:#101828;}
.sub-title {font-size: 1.04rem; color: #667085; margin-bottom: 1.1rem; max-width: 980px;}
.section-title {font-size:1.25rem; font-weight:800; margin-top: 1.4rem; margin-bottom:.65rem; color:#1D2939;}
.card {
    border: 1px solid #e4e7ec; border-radius: 20px; padding: 20px;
    background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
    box-shadow: 0 10px 28px rgba(16, 24, 40, 0.055);
    margin-bottom: 14px;
}
.hero-card {
    border: 1px solid #d0d5dd; border-radius: 24px; padding: 24px;
    background: linear-gradient(135deg, #f8fbff 0%, #ffffff 45%, #f5f8ff 100%);
    box-shadow: 0 14px 34px rgba(16, 24, 40, 0.06);
}
.small-muted {color: #667085; font-size: .9rem;}
.good {color: #027A48; font-weight: 800;}
.warn {color: #B54708; font-weight: 800;}
.bad {color: #B42318; font-weight: 800;}
.pill {display:inline-block; padding:6px 11px; border-radius:999px; background:#f2f4f7; color:#344054; font-weight:700; font-size:.86rem; margin: 3px 5px 3px 0;}
.rec-box {border-left: 5px solid #475467; padding: 12px 16px; background:#f9fafb; border-radius: 12px; margin-bottom: 10px;}
.stMetric {background: #ffffff; border: 1px solid #e4e7ec; padding: 14px; border-radius: 18px; box-shadow: 0 6px 18px rgba(16,24,40,.04);}
hr {margin-top: 1rem; margin-bottom: 1rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def pretty_name(col: str) -> str:
    return col.replace("_", " ").title()


def scenario_class(value: str) -> str:
    value = str(value).lower()
    if "optimistic" in value:
        return "good"
    if "pessimistic" in value:
        return "bad"
    return "warn"


def risk_class(value: str) -> str:
    value = str(value).lower()
    if value == "low":
        return "good"
    if value == "high":
        return "bad"
    return "warn"


def money(value: float) -> str:
    return f"${value:,.0f}"


def risk_interpretation(level: str) -> str:
    level = str(level).lower()
    if level == "low":
        return "The project profile indicates relatively controlled cost and schedule exposure. Normal monitoring is still required to avoid unexpected procurement, labor, or site-condition changes."
    if level == "high":
        return "The project profile indicates serious exposure to cost growth and schedule delay. The project should be reviewed before execution, with stronger contingency, procurement control, and progress monitoring."
    return "The project profile indicates moderate exposure. The project is manageable, but cost, time, labor, equipment, and procurement conditions should be controlled carefully from the early stages."


def generate_recommendations(out: dict, result: dict) -> list[str]:
    recs = []
    risk_level = str(out["risk_level"]).lower()
    cost_pct = float(out["cost_overrun_percentage"]) * 100
    time_pct = float(out["schedule_overrun_percentage"]) * 100

    if risk_level == "high":
        recs.append("Increase management attention before approval because the model indicates a high-risk project profile.")
        recs.append("Prepare a stronger contingency budget and review the main cost drivers before construction starts.")
    elif risk_level == "medium":
        recs.append("Use weekly progress monitoring because the project has a moderate risk profile and may become delayed without control.")
        recs.append("Keep a practical contingency allowance for cost and schedule changes.")
    else:
        recs.append("Maintain standard project controls, because even low-risk projects can face procurement and site-condition changes.")

    if cost_pct > 5:
        recs.append("Monitor material prices and supplier quotations during procurement because the predicted cost overrun is noticeable.")
    else:
        recs.append("Keep the cost baseline updated and compare planned versus predicted cost during each reporting period.")

    if time_pct > 5:
        recs.append("Strengthen schedule control through weekly planned-versus-actual comparison and early corrective actions.")
    else:
        recs.append("Maintain the current schedule logic, but continue tracking critical activities and productivity levels.")

    recs.extend([
        "Confirm labor and equipment availability before the main construction phase to reduce productivity risk.",
        "Use digital records for quantities, cost changes, and progress reports to improve transparency and decision-making.",
        f"Give special attention to cost factors related to {str(result['cost_primary_cause']).lower()} and {str(result['cost_secondary_cause']).lower()}.",
        f"Give special attention to schedule factors related to {str(result['schedule_primary_cause']).lower()} and {str(result['schedule_secondary_cause']).lower()}.",
    ])
    return recs


@st.cache_resource(show_spinner="Loading prediction model...")
def load_system():
    system = BuildingPredictor(EXCEL_PATH)
    system.load_data()
    system.train(system.data.copy())
    return system


system = load_system()

st.markdown('<div class="main-title">🏗️ Building Project Risk & Performance Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">An academic and modern decision-support system for predicting construction cost, duration, cost/time scenarios, and project risk level using machine-learning analysis.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dashboard Overview")
    st.write("This application supports early-stage building project decision-making by estimating cost performance, schedule performance, and risk level.")
    st.divider()
    st.caption("Model outputs")
    st.markdown("<span class='pill'>Actual Cost</span><span class='pill'>Duration</span><span class='pill'>Risk Level</span><span class='pill'>Scenarios</span>", unsafe_allow_html=True)
    st.divider()
    st.caption(f"Training records: {len(system.data):,}")
    st.caption(f"Input features: {len(system.input_columns)}")

st.markdown('<div class="section-title">1. Enter Project Information</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    project_id = st.text_input("Project ID", value=f"NEW_{datetime.now().strftime('%Y%m%d_%H%M')}")

    tabs = st.tabs(["Main Data", "Technical / Site", "Resources / Environment"])
    row = {"project_id": project_id}

    input_cols = system.input_columns
    groups = [input_cols[0::3], input_cols[1::3], input_cols[2::3]]

    for tab, cols in zip(tabs, groups):
        with tab:
            col_left, col_right = st.columns(2)
            for i, col in enumerate(cols):
                target_col = col_left if i % 2 == 0 else col_right
                series = system.inputs_df[col]
                label = pretty_name(col)
                with target_col:
                    if col in system.numeric_columns:
                        values = pd.to_numeric(series, errors="coerce").dropna()
                        min_v = float(values.min())
                        max_v = float(values.max())
                        med_v = float(values.median())
                        is_int = bool((values.round() == values).all()) if len(values) else False
                        if is_int:
                            row[col] = st.number_input(label, min_value=int(min_v), max_value=int(max_v), value=int(round(med_v)), step=1)
                        else:
                            step = (max_v - min_v) / 100 if max_v > min_v else 1.0
                            row[col] = st.number_input(label, min_value=min_v, max_value=max_v, value=med_v, step=step)
                    else:
                        options = [str(x) for x in series.dropna().astype(str).unique().tolist()]
                        default = str(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else (options[0] if options else "")
                        default_index = options.index(default) if default in options else 0
                        row[col] = st.selectbox(label, options=options, index=default_index)

    submitted = st.form_submit_button("Predict Project Results", use_container_width=True)

if submitted:
    raw_df = pd.DataFrame([row])
    result = system.predict(raw_df)
    out = result["predicted_outputs"]

    st.markdown('<div class="section-title">2. Executive Prediction Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Actual Cost", money(out["actual_cost"]), f"{money(out['cost_overrun'])} overrun")
    c2.metric("Predicted Duration", f"{out['actual_duration']:,.0f} days", f"{out['schedule_deviation']:,.0f} days deviation")
    c3.metric("Cost Overrun", f"{out['cost_overrun_percentage']*100:.2f}%", out["cost_scenario"])
    c4.metric("Schedule Overrun", f"{out['schedule_overrun_percentage']*100:.2f}%", out["time_scenario"])

    st.markdown('<div class="section-title">3. Risk Interpretation Dashboard</div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, .95])
    with left:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.subheader("Overall Project Risk")
        st.markdown(f"Risk level: <span class='{risk_class(out['risk_level'])}'>{out['risk_level']}</span>", unsafe_allow_html=True)
        st.write(risk_interpretation(out["risk_level"]))
        st.progress(min(max(out["risk_score"] / 100, 0), 1), text=f"Overall risk score: {out['risk_score']:.1f}/100")
        st.progress(min(max(out["cost_risk_score"] / 100, 0), 1), text=f"Cost risk score: {out['cost_risk_score']:.1f}/100")
        st.progress(min(max(out["schedule_risk_score"] / 100, 0), 1), text=f"Schedule risk score: {out['schedule_risk_score']:.1f}/100")
        st.markdown(
            f"<span class='pill'>Cost Scenario: <span class='{scenario_class(out['cost_scenario'])}'>{out['cost_scenario']}</span></span>"
            f"<span class='pill'>Time Scenario: <span class='{scenario_class(out['time_scenario'])}'>{out['time_scenario']}</span></span>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Risk Probability Distribution")
        prob_df = pd.DataFrame({"Risk Level": list(result["risk_probabilities"].keys()), "Probability": list(result["risk_probabilities"].values())})
        prob_df["Probability"] = prob_df["Probability"] * 100
        st.bar_chart(prob_df.set_index("Risk Level"))
        st.caption("The chart shows the model confidence distribution across Low, Medium, and High risk classes.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">4. Prediction Explanation</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Cost Drivers")
        st.write(f"Primary cause: **{result['cost_primary_cause']}**")
        st.write(f"Secondary cause: **{result['cost_secondary_cause']}**")
        st.markdown('</div>', unsafe_allow_html=True)
    with e2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Schedule Drivers")
        st.write(f"Primary cause: **{result['schedule_primary_cause']}**")
        st.write(f"Secondary cause: **{result['schedule_secondary_cause']}**")
        st.markdown('</div>', unsafe_allow_html=True)
    st.info(result["result_reason_summary"])

    st.markdown('<div class="section-title">5. Project Management Recommendations</div>', unsafe_allow_html=True)
    recommendations = generate_recommendations(out, result)
    for i, rec in enumerate(recommendations, start=1):
        st.markdown(f"<div class='rec-box'><b>Recommendation {i}:</b> {rec}</div>", unsafe_allow_html=True)

    if result["input_driven_flags"]:
        st.markdown('<div class="section-title">6. Input-Based Warning Flags</div>', unsafe_allow_html=True)
        for item in result["input_driven_flags"]:
            st.warning(item)

    if result["input_warnings"]:
        st.markdown('<div class="section-title">7. Data Range Warnings</div>', unsafe_allow_html=True)
        for item in result["input_warnings"]:
            st.warning(item)

    st.markdown('<div class="section-title">8. Methodology & Academic Note</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        This dashboard is designed as a decision-support tool for construction project management. The prediction model uses historical building project data to estimate actual cost, actual duration, cost overrun, schedule overrun, scenario classification, and risk level. The results should support early planning, budgeting, and risk monitoring, but they should not replace professional engineering judgment, site investigation, or expert review.
        </div>
        """,
        unsafe_allow_html=True,
    )

    report_payload = {"input": raw_df.iloc[0].to_dict(), "prediction": result, "recommendations": recommendations}
    st.download_button(
        "Download Prediction Report (JSON)",
        data=pd.Series(report_payload).to_json(indent=2),
        file_name=f"building_prediction_{project_id}.json",
        mime="application/json",
        use_container_width=True,
    )
else:
    st.info("Fill the project information and click **Predict Project Results**.")
