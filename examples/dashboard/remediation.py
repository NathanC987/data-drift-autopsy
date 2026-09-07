"""Remediation view: the localisation-driven triage plus an optional LLM agent.

The core is the triage from ``drift_autopsy.remediation`` (rendered by
``story.render_remediation_step``): run the cheapest label-free repairs as
probes, then recommend escalation only if one recovers a meaningful slice of
the gap. An optional expander lets an LLM narrate the same evidence.
"""

import json
import os

import streamlit as st

from examples.dashboard import story

try:
    from google import genai
except ImportError:
    genai = None

try:
    from dotenv import load_dotenv
except ImportError:  # optional
    def load_dotenv(*_a, **_k):
        return False


def render_remediation_dashboard(loader=None, scope: str = "acs") -> None:
    """Render the remediation step: triage, strategy detail, and an optional agent."""
    story.step_header(6)
    story.render_remediation_step(scope)

    st.markdown("---")
    with st.expander("Optional: let an LLM agent narrate the remediation decision"):
        _render_agent(scope)


def _render_agent(scope: str) -> None:
    if genai is None:
        st.caption("`google-genai` not installed - skip, or `pip install google-genai` to enable.")
        return

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        st.caption("Set `GEMINI_API_KEY` in a `.env` file to enable the agent.")
        return

    rem = story._load(story.REMEDIATION_RESULTS)
    if not rem:
        st.info("Run the remediation demo first.")
        return

    context = {
        s["shift_name"]: {
            "gap": round(s["reference_accuracy"] - s["baseline_before"], 4),
            "triage": s["triage"]["will_retraining_help"],
            "rationale": s["triage"]["rationale"],
            "best_label_free_recovery": max(
                (r["fraction_of_gap_recovered"] for r in s["results"]
                 if r["n_production_labels_required"] == 0), default=0.0,
            ),
        }
        for s in rem.get("settings", [])
    }
    prompt = (
        "You are an MLOps engineer. Given this remediation triage evidence, in 4 sentences advise "
        "whether to retrain and which strategy to try first, and say plainly when retraining is not "
        "worth it.\n\n" + json.dumps(context, indent=2)
    )

    if st.button("Ask the agent", key=f"agent_ask_{scope}"):
        try:
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            st.session_state[f"agent_remediation_{scope}"] = resp.text
        except Exception as exc:
            st.error(f"Agent call failed: {exc}")

    if f"agent_remediation_{scope}" in st.session_state:
        st.write(st.session_state[f"agent_remediation_{scope}"])
        st.caption(
            "To run a strategy for real, use "
            "`python examples/quickstart/remediation_demo.py --shift all` from a terminal - "
            "the dashboard reads the result it writes."
        )
