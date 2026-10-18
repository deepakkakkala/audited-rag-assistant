import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="Audited RAG Assistant", layout="wide")

@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()

st.title("🛡️ Audited RAG Knowledge Assistant")
st.markdown("Enterprise-grade question-answering with strict similarity gating and source citation.")

col1, col2 = st.columns([2, 1])

with col1:
    user_query = st.text_input(
        "Ask a technical question based on indexed documentation:",
        placeholder="e.g., What is the default regional concurrency limit for Lambda?"
    )
    submit_button = st.button("Query Knowledge Base", type="primary")

    if submit_button and user_query.strip():
        with st.spinner("Retrieving semantic matches & validating thresholds..."):
            result = engine.query(user_query)

        if result["status"] == "verified":
            st.success("Verification Succeeded: Answer Grounded in Context")
            st.markdown(f"### Answer\n{result['answer']}")

            st.markdown("### Citations & Supporting Evidence")
            for idx, cit in enumerate(result["citations"]):
                st.info(f"**Source:** `{cit['source_file']}`\n\n> *\"{cit['quoted_snippet']}\"*")

            st.caption(f"**Rationale:** {result.get('confidence_rationale', 'N/A')}")

        elif result["status"] == "refused":
            st.error("Guardrail Refusal: Query Out-of-Bounds")
            st.warning(result["answer"])

        else:
            st.warning("Structured output formatting error. Raw output returned:")
            st.write(result["answer"])

with col2:
    st.subheader("System Telemetry")
    if 'result' in locals() and submit_button and user_query.strip():
        st.metric(
            label="Nearest Chunk Cosine Distance",
            value=f"{result['best_distance']:.4f}",
            delta="PASS" if result["status"] == "verified" else "TRIGGERED GUARDRAIL",
            delta_color="normal" if result["status"] == "verified" else "inverse"
        )
        st.markdown("**Guardrail Threshold:** `<= 0.99`")
        
        with st.expander("Inspected Raw Retrieved Chunks"):
            if "retrieved_chunks" in result:
                for i, chunk in enumerate(result["retrieved_chunks"]):
                    st.text(f"Chunk {i+1}:\n{chunk}\n")
    else:
        st.write("Awaiting query execution to display vector distance and retrieval stats.")