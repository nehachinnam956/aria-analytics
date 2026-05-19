"""
ARIA — Adaptive Reasoning Intelligence for Analytics
Streamlit UI

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.append(os.path.dirname(__file__))
from utils.aria_pipeline import (
    profile_dataset, classify_task, run_automl,
    get_feature_importance, generate_report
)

st.set_page_config(
    page_title="ARIA — Adaptive Reasoning Intelligence for Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #CC0000; margin-bottom: 0; }
    .sub-title  { font-size: 1rem; color: #888; margin-top: 0; }
    .stage-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        background: #1A1A1A; color: white !important; font-size: 0.75rem;
        font-weight: 600; margin: 2px;
    }
    .task-pill {
        display: inline-block; padding: 6px 18px; border-radius: 20px;
        background: #1A1A1A; color: white !important; font-weight: 700; font-size: 1rem;
    }
    .confidence-bar { height: 8px; background: #CC0000; border-radius: 4px; }
    .finding-box {
        background: #f9f9f9 !important; border: 1px solid #ddd;
        border-left: 4px solid #CC0000; padding: 10px 14px;
        border-radius: 4px; margin: 5px 0;
        color: #1A1A1A !important;
    }
    .finding-box * { color: #1A1A1A !important; }
    .advocate-box {
        background: #f4f4f4 !important; border: 1px solid #ccc;
        padding: 10px 14px; border-radius: 4px;
        font-size: 0.88rem; margin: 5px 0;
        color: #1A1A1A !important;
    }
    .advocate-box * { color: #1A1A1A !important; }
    hr { border: none; border-top: 1px solid #eee; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">ARIA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Adaptive Reasoning Intelligence for Analytics &nbsp;·&nbsp; '
    'ABB EngineeredX 2.0 &nbsp;·&nbsp; SRM University AP</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<br>'
    '<span class="stage-badge">1 Profile</span>'
    '<span class="stage-badge">2 Classify</span>'
    '<span class="stage-badge">3 AutoML</span>'
    '<span class="stage-badge">4 Explain</span>'
    '<span class="stage-badge">5 Report</span>',
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    uploaded = st.file_uploader("Upload Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
    problem_text = st.text_area(
        "Problem Description",
        placeholder="e.g. Predict motor failure from vibration and current data",
        height=80
    )
    target_col_input = st.text_input(
        "Target Column (leave blank if unknown)",
        placeholder="e.g. fault, churn, price, sentiment"
    )
    consequence = st.selectbox(
        "Consequence of Being Wrong",
        ["balanced", "false_negative_costly", "false_positive_costly"],
        format_func=lambda x: {
            "balanced": "Both errors equally bad",
            "false_negative_costly": "Missing a case is worse (e.g. missed fault)",
            "false_positive_costly": "False alarm is worse",
        }[x]
    )
    budget = st.selectbox(
        "Computation Budget",
        ["Fast", "Standard", "Exhaustive"],
        help="Fast = 3 models, Standard = 5, Exhaustive = 8"
    )
    expertise = st.selectbox("Your Expertise Level", ["Beginner", "Practitioner", "Expert"])
    run_btn = st.button("🚀 Run ARIA", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("**Demo datasets to try:**")
    st.markdown("- [Telecom Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")
    st.markdown("- [House Prices (Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)")
    st.markdown("- [Mall Customers (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)")
    st.markdown("---")
    st.caption("Built by Neha Chinnam | SRM University AP | 2026")

for key in ["profile", "task_result", "automl_result", "feature_imp", "report", "df"]:
    if key not in st.session_state:
        st.session_state[key] = None

if run_btn:
    if uploaded is None:
        st.error("Please upload a dataset first.")
    elif not problem_text.strip():
        st.error("Please enter a problem description.")
    else:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.session_state.df = df
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        target = target_col_input.strip() if target_col_input.strip() in df.columns else None

        prog = st.progress(0, text="Stage 1: Profiling dataset...")
        profile = profile_dataset(df)
        st.session_state.profile = profile

        prog.progress(20, text="Stage 2: Classifying task type...")
        task_result = classify_task(df, profile, problem_text, target)
        st.session_state.task_result = task_result

        prog.progress(40, text=f"Stage 3: Running AutoML for {task_result['task']}...")
        automl_result = run_automl(df, task_result, target, budget)
        st.session_state.automl_result = automl_result

        prog.progress(70, text="Stage 4: Computing feature importance...")
        feature_imp = pd.DataFrame()
        if target and task_result["task"] in ["Classification", "Regression"]:
            feature_imp = get_feature_importance(df, target, task_result["task"])
        st.session_state.feature_imp = feature_imp

        prog.progress(90, text="Stage 5: Generating tri-persona report...")
        report = generate_report(profile, task_result, automl_result, feature_imp, problem_text, consequence)
        st.session_state.report = report

        prog.progress(100, text="Done.")
        prog.empty()
        st.success("✅ ARIA pipeline complete.")

if st.session_state.profile is not None:
    profile       = st.session_state.profile
    task_result   = st.session_state.task_result
    automl_result = st.session_state.automl_result
    feature_imp   = st.session_state.feature_imp
    report        = st.session_state.report
    df            = st.session_state.df

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown("### Task Detected")
        st.markdown(f'<div class="task-pill">{task_result["task"]}</div>', unsafe_allow_html=True)
        conf_pct = int(task_result["confidence"] * 100)
        st.markdown(f"Confidence: **{conf_pct}%**")
        st.markdown(f'<div class="confidence-bar" style="width:{conf_pct}%"></div>', unsafe_allow_html=True)
        if task_result["ontology_match"]:
            st.caption(f"🏭 ABB Ontology: {task_result['ontology_match']}")
    with col2:
        st.metric("Rows", f"{profile['shape']['rows']:,}")
    with col3:
        st.metric("Columns", profile["shape"]["cols"])
    with col4:
        st.metric("Drift Signal", "⚠️ Detected" if profile["drift_flag"] else "✅ None")

    st.markdown("<hr>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Profile", "🧠 Task Classification",
        "🤖 AutoML Results", "📈 Feature Importance", "📄 Tri-Persona Report"
    ])

    with tab1:
        st.subheader("Dataset Overview")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Column types**")
            dtypes = df.dtypes.astype(str).value_counts().reset_index()
            dtypes.columns = ["Type", "Count"]
            fig = px.bar(dtypes, x="Type", y="Count", color="Type",
                         color_discrete_sequence=["#CC0000", "#1A1A1A", "#888"],
                         template="simple_white")
            fig.update_layout(showlegend=False, height=250, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            if profile["missing"]:
                st.markdown("**Missing values (%)**")
                miss_df = pd.DataFrame(list(profile["missing"].items()), columns=["Column", "Missing %"])
                fig2 = px.bar(miss_df, x="Column", y="Missing %",
                              color_discrete_sequence=["#CC0000"], template="simple_white")
                fig2.update_layout(height=250, margin=dict(t=20, b=20))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.success("No missing values found.")
        st.markdown("**Column statistics**")
        numeric_df = df.describe().T.reset_index()
        numeric_df.columns = ["Column"] + list(numeric_df.columns[1:])
        st.dataframe(numeric_df.style.format(precision=3), use_container_width=True)
        if profile["correlated_pairs"]:
            st.markdown(f"**High-correlation pairs ({len(profile['correlated_pairs'])} found)**")
            st.dataframe(pd.DataFrame(profile["correlated_pairs"]), use_container_width=True)
        if profile["skewed_features"]:
            st.warning(f"Skewed features: {', '.join(profile['skewed_features'])}")

    with tab2:
        st.subheader("How ARIA Classified the Task")
        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("**Reasoning chain**")
            for r in task_result["reasoning"]:
                st.markdown(f'<div class="finding-box">&#10003;&nbsp; {r}</div>', unsafe_allow_html=True)
            st.markdown("<br>**Preprocessing plan**", unsafe_allow_html=True)
            for p in task_result["preprocessing_plan"]:
                st.markdown(f"- {p}")
        with col_y:
            st.markdown("**Devil's Advocate check**")
            for d in task_result["devil_advocate"]:
                st.markdown(f'<div class="advocate-box">&#128269;&nbsp; {d}</div>', unsafe_allow_html=True)
            st.markdown("<br>**Selected metric**", unsafe_allow_html=True)
            st.info(f"📐 {task_result['metric']}")
            if task_result["ontology_match"]:
                st.markdown("**ABB Industrial Ontology match**")
                st.success(f"🏭 {task_result['ontology_match']}")

    with tab3:
        st.subheader("AutoML Model Results")
        if automl_result.get("error"):
            st.error(f"AutoML error: {automl_result['error']}")
        else:
            st.markdown(f"**Best model: `{automl_result['best_model_name']}`**")
            if automl_result["metrics"]:
                cols = st.columns(min(len(automl_result["metrics"]), 4))
                for i, (k, v) in enumerate(list(automl_result["metrics"].items())[:4]):
                    with cols[i % len(cols)]:
                        try:    st.metric(k, round(float(v), 4))
                        except: st.metric(k, str(v))
            if automl_result["leaderboard"] is not None:
                st.markdown("**Full model leaderboard**")
                lb = automl_result["leaderboard"]
                numeric_lb = lb.select_dtypes(include=[float, int])
                st.dataframe(
                    lb.style.format({c: "{:.4f}" for c in numeric_lb.columns}, na_rep="-"),
                    use_container_width=True
                )
            if automl_result["predictions"] is not None:
                st.markdown("**Sample predictions**")
                st.dataframe(automl_result["predictions"].head(20), use_container_width=True)

    with tab4:
        st.subheader("Feature Importance")
        if feature_imp is not None and not feature_imp.empty and "Error" not in feature_imp.columns:
            top_n = feature_imp.head(15)
            fig3 = px.bar(
                top_n, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale=["#f5f5f5", "#CC0000"],
                template="simple_white", title="Top 15 Features by Importance (Random Forest)"
            )
            fig3.update_layout(
                height=450, yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False, margin=dict(l=10, r=10, t=40, b=20)
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("**What this means (plain English)**")
            for i, row in feature_imp.head(5).iterrows():
                st.markdown(
                    f"- **{row['Feature']}** accounts for "
                    f"**{round(row['Importance']*100, 1)}%** of the model's decisions."
                )
        else:
            st.info("Feature importance requires a target column. Specify one in the sidebar and re-run.")

    with tab5:
        st.subheader("Three Outputs, Three Readers")
        p_tab1, p_tab2, p_tab3 = st.tabs(["🔬 Engineer Mode", "📋 Manager Mode", "📁 Regulator Mode"])
        eng = report["engineer"]
        mgr = report["manager"]
        reg = report["regulator"]

        with p_tab1:
            st.markdown(f"**Model:** `{eng['model']}` &nbsp;|&nbsp; **Task:** `{eng['task']}`")
            st.markdown(f"**Task confidence:** {int(eng['confidence']*100)}%")
            if eng["metrics"]:
                cols = st.columns(min(len(eng["metrics"]), 4))
                for i, (k, v) in enumerate(list(eng["metrics"].items())[:4]):
                    with cols[i % len(cols)]:
                        try:    st.metric(k, round(float(v), 4))
                        except: st.metric(k, str(v))
            st.markdown("**Top features**")
            for f in eng["top_features"]:
                st.markdown(f"- {f}")
            st.markdown("**Preprocessing applied**")
            for p in eng["preprocessing"]:
                st.markdown(f"- {p}")
            st.markdown("**Devil's Advocate audit**")
            for d in eng["devil_advocate"]:
                st.markdown(f'<div class="advocate-box">&#128269;&nbsp; {d}</div>', unsafe_allow_html=True)
            if eng["drift_flag"]:
                st.warning("⚠️ Drift signal detected in input data. Monitor closely after deployment.")

        with p_tab2:
            st.markdown(f"### {mgr['finding']}")
            st.markdown("---")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1: st.metric("Model Confidence", mgr["confidence"])
            with col_m2: st.metric("Risk Level", mgr["risk"])
            with col_m3: st.metric("Next Check", mgr["next_check"])
            st.markdown("**Primary metric**")
            st.info(mgr["metric"] if mgr["metric"] else "See technical report")
            st.markdown("**Key drivers**")
            st.markdown(f"> {mgr['top_drivers']}")
            st.markdown("**Recommended action**")
            st.success(mgr["action"])

        with p_tab3:
            st.markdown(f"#### {reg['title']}")
            st.markdown(
                f"**Model:** {reg['model']}  |  **Task:** {reg['task']}  |  "
                f"**Training date:** {reg['training_date']}"
            )
            st.markdown(f"**Training samples:** {reg['training_rows']:,}")
            st.markdown("**Performance metrics**")
            for k, v in reg["metrics"].items():
                try:    st.markdown(f"- {k}: `{round(float(v), 4)}`")
                except: st.markdown(f"- {k}: `{v}`")
            st.markdown("**Preprocessing applied**")
            for p in reg["preprocessing"]:
                st.markdown(f"- {p}")
            st.markdown("**Explainability**")
            st.markdown(reg["explainability"])
            st.markdown("**Drift status**")
            if "flagged" in reg["drift_status"].lower():
                st.warning(reg["drift_status"])
            else:
                st.success(reg["drift_status"])
            st.markdown("**Bias check**")
            st.info(reg["bias_check"])
            st.markdown("**Deployment recommendation**")
            st.info(reg["recommendation"])
            st.markdown("**Data lineage**")
            st.markdown(reg["data_lineage"])

else:
    st.markdown("""
    ### How ARIA works

    1. **Upload any CSV or Excel dataset** — tabular, time-series, sensor data, whatever you have
    2. **Describe the problem** in one sentence — no technical terms needed
    3. **ARIA profiles the data**, classifies the task type (with a built-in Devil's Advocate check),
       runs AutoML, computes feature importance, and generates three persona-specific reports

    **What makes it different from standard AutoML:**
    - It questions its own classification before confirming it
    - Metrics are weighted by the consequence of being wrong, not just statistics
    - Three simultaneous outputs: Engineer, Manager, Regulator

    Upload a dataset in the sidebar to get started.
    """)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.markdown("**Stage 1**\nProfile")
    with col2: st.markdown("**Stage 2**\nClassify")
    with col3: st.markdown("**Stage 3**\nAutoML")
    with col4: st.markdown("**Stage 4**\nExplain")
    with col5: st.markdown("**Stage 5**\nReport")