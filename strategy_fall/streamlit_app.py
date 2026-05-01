"""
Interactive Strategy Explorer — Streamlit demo for Phase 2.

Run from repo root (cs639Team10-main):
    streamlit run strategy_fall/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from build_graph import StrategyAnalyzer
from utils import Graph
from visualize import draw_fancy_graph, draw_graph_side_by_side

STRATEGY_FALL = Path(__file__).resolve().parent


@st.cache_resource
def get_analyzer(cluster_map: str, cluster_tags: str) -> StrategyAnalyzer:
    return StrategyAnalyzer(cluster_map, cluster_tags)


def clean_trajectory(steps: List[str]) -> str:
    """Removes thinking tags and cleans up raw artifacts."""
    cleaned = []
    for s in steps:
        # Remove tags and normalize whitespace
        s = s.replace("<think>", "").replace("</think>", "").strip()
        # Escape dollar signs for currency to prevent incorrect LaTeX rendering
        s = s.replace("$", "\\$")
        if s:
            cleaned.append(s)
    return "\n\n---\n\n".join(cleaned)


def render_step_card(step_text: str, node_id: int, tag: str) -> None:
    """Renders a single reasoning step with a colored badge and node ID."""
    TAG_COLORS = {
        "Planning": "#87CEEB",                # Sky Blue
        "Uncertainty Management": "#FFA500",   # Orange
        "Conclusion": "#90EE90",               # Light Green
        "Active Computation": "#D3D3D3",       # Light Gray
        "Other": "#F0F2F6"                     # Light Sidebar Gray
    }
    color = TAG_COLORS.get(tag, "#F0F2F6")
    
    # Clean text
    step_text = step_text.replace("<think>", "").replace("</think>", "").strip().replace("$", "\\$")
    
    if not step_text:
        return

    st.markdown(f"""
        <div style="border-left: 5px solid {color}; padding: 10px; margin-bottom: 10px; background-color: rgba(255,255,255,0.05); border-radius: 5px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                <span style="background-color: {color}; color: black; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: bold;">
                    {tag}
                </span>
                <span style="font-family: monospace; font-size: 0.8rem; color: #888;">
                    #Node {node_id}
                </span>
            </div>
            <div style="font-size: 0.95rem;">
                {step_text}
            </div>
        </div>
    """, unsafe_allow_html=True)


def _clustered_paths(version: str) -> Tuple[Path, Path, Path, Path]:
    """Return (cluster_map, cluster_tags, sft_traces, rl_traces) for a given version.

    Naming convention produced by clustering.py:
        {model}_traces-{version}_clustered.json
    Fallback (older runs without version suffix in filename):
        {model}_clustered.json
    """
    base = STRATEGY_FALL / "data" / f"clustered_{version}"
    cmap = base / "cluster_map.json"
    ctags = base / "cluster_tags.json"

    SFT_MODEL = "Qwen2.5-7B-Instruct-AWQ"
    RL_MODEL  = "DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ"

    sft = base / f"{SFT_MODEL}_traces-{version}_clustered.json"
    rl  = base / f"{RL_MODEL}_traces-{version}_clustered.json"

    # Fallback: legacy filenames without the version infix
    if not sft.exists():
        sft = base / f"{SFT_MODEL}_clustered.json"
    if not rl.exists():
        rl = base / f"{RL_MODEL}_clustered.json"

    return cmap, ctags, sft, rl


@st.cache_data
def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_causal_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def intervention_tag_for_q(df: pd.DataFrame, qid: int) -> Optional[str]:
    if df is None or df.empty:
        return None
    qcol = df["qid"].astype(int)
    row = df[(qcol == int(qid)) & (df["type"] == "intervention")]
    if row.empty:
        return None
    return str(row.iloc[0]["tag"])


def apply_anchor_tags(analyzer: StrategyAnalyzer, node_map) -> None:
    for cid, node in node_map.items():
        node.node_type = analyzer.cluster_tags.get(str(cid), "Other")


def build_pair(
    analyzer: StrategyAnalyzer,
    sft_item: Dict,
    rl_item: Dict,
) -> Tuple[Graph, Graph, Any, Any]:
    g_sft, nm_sft = analyzer.build_question_graph(sft_item["trajectories"])
    g_rl, nm_rl = analyzer.build_question_graph(rl_item["trajectories"])
    apply_anchor_tags(analyzer, nm_sft)
    apply_anchor_tags(analyzer, nm_rl)
    return g_sft, g_rl, nm_sft, nm_rl


def main() -> None:
    st.set_page_config(
        page_title="Strategy Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Interactive Strategy Explorer")
    st.caption("Browse questions, compare SFT vs RL reasoning graphs, and highlight causal intervention anchors.")

    with st.sidebar:
        st.header("Data")
        version = st.selectbox(
            "Trace set",
            options=["q1000", "q50", "math_l5", "math_l1", "math_l3"],
            index=0,
            help="math_l5 = MATH-500 Level 5 (hardest) | q1000/q50 = GSM8K",
        )
        cmap, ctags, sft_path, rl_path = _clustered_paths(version)

        st.header("Mode")
        mode = st.radio("Analysis Mode", ["Deep-Dive (Question View)", "Global Statistics"], index=0)

        st.header("View")
        layout = st.radio("Layout", ["Side-by-side", "Single model"], horizontal=False)
        single_model = st.radio("Model (single view)", ["SFT (Instruct)", "RL (DeepSeek-R1)"], horizontal=True) if layout == "Single model" else None

        st.header("Causal highlight")
        highlight_causal = st.checkbox(
            "Outline causal intervention tag",
            value=True,
            help="Uses causal_details.csv (intervention row) for the selected question.",
        )

    missing = [p for p in (cmap, ctags, sft_path, rl_path) if not p.exists()]
    if missing:
        st.error("Missing required files:")
        for p in missing:
            st.code(str(p))
        st.info("Run clustering + tagging + graph pipeline for this version, or pick another trace set.")
        return

    if mode == "Global Statistics":
        render_global_stats()
        return

    causal_rl_path = STRATEGY_FALL / "results" / "causal" / "causal_details.csv"
    causal_sft_path = STRATEGY_FALL / "results" / "causal_sft" / "causal_details.csv"
    causal_rl = load_causal_csv(str(causal_rl_path)) if causal_rl_path.exists() else None
    causal_sft = load_causal_csv(str(causal_sft_path)) if causal_sft_path.exists() else None

    sft_data = load_json_array(str(sft_path))
    rl_data = load_json_array(str(rl_path))
    n = min(len(sft_data), len(rl_data))

    if n == 0:
        st.warning("No questions in trace JSON.")
        return

    labels = [f"Q{i}: {(sft_data[i].get('question') or '')[:90]}…" for i in range(n)]
    q_idx = st.selectbox("Question", options=list(range(n)), format_func=lambda i: labels[i])

    analyzer = get_analyzer(str(cmap), str(ctags))
    g_sft, g_rl, _, _ = build_pair(analyzer, sft_data[q_idx], rl_data[q_idx])

    tag_rl = intervention_tag_for_q(causal_rl, q_idx) if highlight_causal else None
    tag_sft = intervention_tag_for_q(causal_sft, q_idx) if highlight_causal else None

    with st.expander("Question text", expanded=False):
        st.write(sft_data[q_idx].get("question", ""))

    cols = st.columns(2)
    with cols[0]:
        st.metric("Causal tag (RL intervention)", tag_rl or "—")
    with cols[1]:
        st.metric("Causal tag (SFT intervention)", tag_sft or "—")

    if layout == "Side-by-side":
        fig = draw_graph_side_by_side(
            g_sft,
            g_rl,
            title_a=f"Instruct (SFT) — Q{q_idx}",
            title_b=f"DeepSeek-R1 (RL) — Q{q_idx}",
            figsize=(18, 8),
            highlight_tag_a=tag_sft,
            highlight_tag_b=tag_rl,
        )
        st.pyplot(fig)
        plt.close(fig)
    else:
        use_rl = single_model.startswith("RL")
        g = g_rl if use_rl else g_sft
        tag = tag_rl if use_rl else tag_sft
        title = f"{'DeepSeek-R1 (RL)' if use_rl else 'Instruct (SFT)'} — Q{q_idx}"
        fig, ax = plt.subplots(figsize=(12, 9), facecolor="#F8F9FA")
        draw_fancy_graph(g, title=title, ax=ax, highlight_tag=tag)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Live Reasoning Transcripts (Tagged Steps)", expanded=False):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**SFT (Instruct)**")
            for i, traj in enumerate(sft_data[q_idx].get("trajectories", [])):
                with st.container(height=450):
                    st.caption(f"Trajectory {i}")
                    cids = traj.get("cluster_ids", [])
                    steps = traj.get("text_steps", [])
                    for cid, step_txt in zip(cids, steps):
                        tag = analyzer.cluster_tags.get(str(cid), "Other")
                        render_step_card(step_txt, cid, tag)
        with t2:
            st.markdown("**RL (DeepSeek-R1)**")
            for i, traj in enumerate(rl_data[q_idx].get("trajectories", [])):
                with st.container(height=450):
                    st.caption(f"Trajectory {i}")
                    cids = traj.get("cluster_ids", [])
                    steps = traj.get("text_steps", [])
                    for cid, step_txt in zip(cids, steps):
                        tag = analyzer.cluster_tags.get(str(cid), "Other")
                        render_step_card(step_txt, cid, tag)

    st.divider()
    st.subheader("Causal Verification (Control vs Intervention)")
    st.markdown("""
        Compare the **Control** (baseline) to the **Intervention** (where we masked the anchor).
        - A **large accuracy drop** proves the node was a **Critical Hub**.
        - A **small/zero drop** proves the model is **Resilient** and found an alternative path.
    """)
    c1, c2 = st.columns(2)
    with c1:
        if causal_rl is not None:
            sub = causal_rl[(causal_rl["qid"].astype(int) == q_idx)]
            if not sub.empty:
                st.write("**RL**")
                st.dataframe(sub[["type", "tag", "accuracy"]], use_container_width=True, hide_index=True)
            else:
                st.caption("No RL causal row for this qid.")
        else:
            st.caption("No `results/causal/causal_details.csv`")
    with c2:
        if causal_sft is not None:
            sub = causal_sft[(causal_sft["qid"].astype(int) == q_idx)]
            if not sub.empty:
                st.write("**SFT**")
                st.dataframe(sub[["type", "tag", "accuracy"]], use_container_width=True, hide_index=True)
            else:
                st.caption("No SFT causal row for this qid.")
        else:
            st.caption("No `results/causal_sft/causal_details.csv`")


def _short_model_name(model_col_val: str) -> str:
    if "DeepSeek" in model_col_val:
        return "RL (R1)"
    if "Instruct" in model_col_val:
        return "SFT (Instruct)"
    return "Base"


def _load_report(version: str) -> "pd.DataFrame | None":
    p = STRATEGY_FALL / "results" / version / f"strategy_collapse_report_{version}.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["model_short"] = df["model"].apply(_short_model_name)
        df["version"] = version
        return df
    return None


def render_global_stats() -> None:
    """Renders high-level aggregate metrics, Level-5 complexity stress-test, and causal summaries."""

    # ── Baseline (GSM8K / q1000) ──────────────────────────────────────────────
    st.subheader("Systemic Strategy Comparison — Baseline (GSM8K, n=1000)")
    st.info(
        "**Summary**: RL-trained models (DeepSeek-R1) consistently exhibit more 'Thought Anchors' "
        "than SFT counterparts, visible in higher Planning and Uncertainty Management intensities. "
        "This suggests RL incentivises meta-cognitive verification."
    )

    df_base = _load_report("q1000")
    if df_base is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Planning Intensity**")
            st.bar_chart(df_base.set_index("model_short")[["planning_intensity"]])
        with c2:
            st.markdown("**Uncertainty Management Intensity**")
            st.bar_chart(df_base.set_index("model_short")[["uncertainty_intensity"]])
        st.markdown("**Structural Diversity (Entropy & Branching)**")
        st.line_chart(df_base.set_index("model_short")[["mean_strategy_entropy", "mean_branching_factor"]])
        st.caption("Higher entropy and branching indicate a 'Reasoning Web' with multiple redundant paths.")
    else:
        st.info("Baseline report not found at `results/q1000/strategy_collapse_report_q1000.csv`. "
                "Run the pipeline with version `q1000` first.")

    # ── Phase 1: Complexity Threshold (MATH Level 5) ──────────────────────────
    st.divider()
    st.subheader("Phase 1 — Complexity Threshold Stress-Test (MATH Level 5)")
    st.markdown(
        """
        **Hypothesis**: At the hardest difficulty, does the RL model's 'Logical Web' **scale** 
        (maintain high entropy + branching), or does it **collapse** to a single linear chain?
        Conversely, does the SFT model's already-brittle structure shatter entirely?

        | Metric | Expected SFT | Expected RL |
        |---|---|---|
        | Strategy Entropy | ↓ drops sharply (fewer unique paths) | ↔ / ↑ maintained or grows |
        | Branching Factor | ↓ collapses toward 1 | ↔ stays distributed |
        | Planning Intensity | ↓ model skips meta-reasoning | ↑ RL doubles down on structure |
        | Uncertainty Mgmt | ↑ spikes (confusion) | ↑ but controlled |
        """
    )

    df_l5 = _load_report("math_l5")
    df_base_for_compare = _load_report("q1000")

    if df_l5 is not None:
        st.markdown("#### Level 5 Results")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Planning Intensity (L5)**")
            st.bar_chart(df_l5.set_index("model_short")[["planning_intensity"]])
        with c2:
            st.markdown("**Uncertainty Management Intensity (L5)**")
            st.bar_chart(df_l5.set_index("model_short")[["uncertainty_intensity"]])
        st.markdown("**Structural Diversity — L5 (Entropy & Branching)**")
        st.line_chart(df_l5.set_index("model_short")[["mean_strategy_entropy", "mean_branching_factor"]])

        # Cross-difficulty delta comparison (requires baseline)
        if df_base_for_compare is not None:
            st.markdown("#### Δ Difficulty Scaling  (Level 5 − GSM8K baseline)")
            st.caption(
                "Positive = metric *grew* at Level 5 (good for RL). "
                "Negative = metric *collapsed* under pressure (bad for SFT)."
            )
            metrics = ["mean_strategy_entropy", "mean_branching_factor",
                       "planning_intensity", "uncertainty_intensity"]
            rows = []
            for model_short in df_l5["model_short"].unique():
                l5_row   = df_l5[df_l5["model_short"] == model_short][metrics].iloc[0]
                base_row = df_base_for_compare[df_base_for_compare["model_short"] == model_short]
                if base_row.empty:
                    continue
                base_row = base_row[metrics].iloc[0]
                delta = (l5_row - base_row).to_dict()
                delta["model"] = model_short
                rows.append(delta)
            if rows:
                delta_df = pd.DataFrame(rows).set_index("model")
                st.dataframe(delta_df.style.background_gradient(cmap="RdYlGn", axis=None),
                             use_container_width=True)
    else:
        st.info(
            "Level 5 report not found at `results/math_l5/strategy_collapse_report_math_l5.csv`.\n\n"
            "**To generate it, run:**\n"
            "```bash\n"
            "# 1. Generate traces (needs GPU + vLLM)\n"
            "python strategy_fall/data/generate_traces.py \\\n"
            "    --model Qwen/Qwen2.5-7B-Instruct-AWQ \\\n"
            "    --dataset math --math_level 5 --num_questions 50\n\n"
            "python strategy_fall/data/generate_traces.py \\\n"
            "    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\\n"
            "    --dataset math --math_level 5 --num_questions 50\n\n"
            "# 2. Run the analysis pipeline\n"
            "bash strategy_fall/run_analysis.sh math_l5\n"
            "```"
        )

    # ── Causal Anchor Impact ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Causal Anchor Impact (Avg Accuracy Drop)")
    st.markdown(
        """
        **How to read this chart**:
        - **Positive Drop**: Removing the anchor caused accuracy to fall → **Critical Hub**.
        - **Negative Drop**: Forcing a rethink *improved* accuracy → **Brittle SFT loop**.
        """
    )

    crl_path  = STRATEGY_FALL / "results" / "causal"     / "causal_summary.csv"
    csft_path = STRATEGY_FALL / "results" / "causal_sft" / "causal_summary.csv"
    crl5_path = STRATEGY_FALL / "results" / "causal_math_l5" / "causal_summary.csv"

    if crl_path.exists() and csft_path.exists():
        crl  = pd.read_csv(crl_path).set_index("type")
        csft = pd.read_csv(csft_path).set_index("type")
        drop_data = {
            "RL (R1) Drop":       crl.loc["control"]  - crl.loc["intervention"],
            "SFT (Instruct) Drop": csft.loc["control"] - csft.loc["intervention"],
        }
        if crl5_path.exists():
            crl5 = pd.read_csv(crl5_path).set_index("type")
            drop_data["RL (R1) Drop — L5"] = crl5.loc["control"] - crl5.loc["intervention"]
        st.bar_chart(pd.DataFrame(drop_data))
        st.caption(
            "Positive values = accuracy LOSS when anchor removed. "
            "Negative = accuracy GAIN (brittle logic). "
            "L5 column appears once `results/causal_math_l5/` is populated."
        )
    else:
        st.info("Causal summaries not found in `results/causal/` or `results/causal_sft/`.")


if __name__ == "__main__":
    main()
