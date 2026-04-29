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


def _clustered_paths(version: str) -> Tuple[Path, Path, Path, Path]:
    """cluster_map, cluster_tags, sft traces, rl traces."""
    base = STRATEGY_FALL / "data" / f"clustered_{version}"
    cmap = base / "cluster_map.json"
    ctags = base / "cluster_tags.json"
    sft = base / f"Qwen2.5-7B-Instruct-AWQ_traces-{version}_clustered.json"
    rl = base / f"DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_traces-{version}_clustered.json"
    if not sft.exists():
        sft = base / "Qwen2.5-7B-Instruct-AWQ_clustered.json"
    if not rl.exists():
        rl = base / "DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_clustered.json"
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
        version = st.selectbox("Trace set", options=["q1000", "q50"], index=0)
        cmap, ctags, sft_path, rl_path = _clustered_paths(version)

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

    st.divider()
    st.subheader("Causal run (intervention row)")
    c1, c2 = st.columns(2)
    with c1:
        if causal_rl is not None:
            sub = causal_rl[(causal_rl["qid"].astype(int) == q_idx) & (causal_rl["type"] == "intervention")]
            if not sub.empty:
                st.write("**RL**")
                st.dataframe(sub[["qid", "type", "tag", "accuracy"]], use_container_width=True, hide_index=True)
            else:
                st.caption("No RL causal row for this qid.")
        else:
            st.caption("No `results/causal/causal_details.csv`")
    with c2:
        if causal_sft is not None:
            sub = causal_sft[(causal_sft["qid"].astype(int) == q_idx) & (causal_sft["type"] == "intervention")]
            if not sub.empty:
                st.write("**SFT**")
                st.dataframe(sub[["qid", "type", "tag", "accuracy"]], use_container_width=True, hide_index=True)
            else:
                st.caption("No SFT causal row for this qid.")
        else:
            st.caption("No `results/causal_sft/causal_details.csv`")


if __name__ == "__main__":
    main()
