# app.py
import os
import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase, basic_auth

# --- LLMs ---
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from openai import OpenAI

# -----------------------------
# ENV / CONFIG
# -----------------------------
URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
USER = os.environ.get("NEO4J_USERNAME", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
DB = os.environ.get("NEO4J_DATABASE", "neo4j")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

EMBED_MODEL = "text-embedding-3-small"  # upgrade to -large if needed

# -----------------------------
# HELPERS
# -----------------------------
def make_llm():
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    return ChatOpenAI(
        temperature=0,
        model_name="gpt-4o",
        api_key=OPENAI_API_KEY
    )

def make_openai_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    return OpenAI(api_key=OPENAI_API_KEY)

def make_driver():
    driver = GraphDatabase.driver(URI, auth=basic_auth(USER, PASSWORD))
    driver.verify_connectivity()
    return driver

TOPIC_SYS_PROMPT = """You extract the single most important keyword/topic that the paragraph is about.
Respond with ONLY the topic text, no punctuation or quotes. Keep it concise (max 8 words).
Examples:
- "The fourth season of Chicago Fire..." -> Chicago Fire Season 4
- "Apple unveiled the iPhone 16 Pro..." -> iPhone 16 Pro
"""

def extract_main_topic(llm: ChatOpenAI, paragraph: str) -> str:
    resp = llm.invoke([
        SystemMessage(content=TOPIC_SYS_PROMPT),
        HumanMessage(content=paragraph)
    ])
    topic = (resp.content or "").strip()
    topic = topic.replace('"', '').replace("'", "").strip()
    return topic

def embed_texts(oa_client: OpenAI, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    emb = oa_client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [d.embedding for d in emb.data]
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

def cosine_sim_matrix(vec_query: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    return vecs @ vec_query

GET_NODE_NAMES_CYPHER = """
MATCH (n)
OPTIONAL MATCH (n)-[:HAS_NAME]->(v:value_text)
WITH n, v
RETURN
  coalesce(n.id, elementId(n)) AS id,
  coalesce(v.id, n.id, elementId(n)) AS name_id_like,
  labels(n) AS graph_labels
"""

def prettify_name(name_id_like: str) -> str:
    if not name_id_like:
        return ""
    x = name_id_like
    if x.startswith("val_"):
        x = x[4:]
    x = x.replace("_", " ")
    return x.strip()

def fetch_all_nodes_with_names(driver) -> List[Dict[str, Any]]:
    with driver.session(database=DB) as session:
        rows = session.run(GET_NODE_NAMES_CYPHER).data()

    results = []
    for r in rows:
        node_id = r.get("id")
        raw_name = r.get("name_id_like") or node_id
        pretty = prettify_name(raw_name)
        graph_labels = r.get("graph_labels", [])
        results.append({
            "id": node_id,
            "raw_name": raw_name,
            "name": pretty if pretty else (node_id or ""),
            "graph_labels": graph_labels,
        })
    return results

def find_most_similar_node(oa_client: OpenAI, topic: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {}

    texts = [c["name"] for c in candidates]
    vecs = embed_texts(oa_client, texts)           # (N, D)
    q_vec = embed_texts(oa_client, [topic])[0]     # (D,)
    sims = cosine_sim_matrix(q_vec, vecs)          # (N,)

    best_idx = int(np.argmax(sims))
    best = candidates[best_idx].copy()
    best["similarity"] = float(sims[best_idx])
    return best

GET_NEIGHBORS_CYPHER = """
MATCH (center {id: $id})
OPTIONAL MATCH (center)-[r]-(nbr)
RETURN
  center.id AS center_id,
  type(r)   AS rel_type,
  CASE
    WHEN r IS NULL THEN NULL
    WHEN startNode(r) = center THEN 'OUT'
    ELSE 'IN'
  END AS direction,
  coalesce(nbr.id, elementId(nbr)) AS neighbor_id,
  labels(nbr) AS neighbor_graph_labels
ORDER BY rel_type, neighbor_id
"""

def fetch_connections(driver, node_id: str) -> List[Dict[str, Any]]:
    with driver.session(database=DB) as session:
        rows = session.run(GET_NEIGHBORS_CYPHER, {"id": node_id}).data()
    return [r for r in rows if r.get("rel_type") is not None]

def neighbors_to_dataframe(neighbors: List[Dict[str, Any]]) -> pd.DataFrame:
    def _type_from_labels(labels: List[str]) -> str:
        return ",".join(labels or [])
    def _pretty_id(s: str) -> str:
        return prettify_name(s or "")
    records = []
    for r in neighbors:
        records.append({
            "relationship": r["rel_type"],
            "direction": r["direction"],
            "neighbor_id": r["neighbor_id"],
            "neighbor_name": _pretty_id(r["neighbor_id"]),
            "neighbor_labels": _type_from_labels(r.get("neighbor_graph_labels")),
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=["relationship", "neighbor_name"]).reset_index(drop=True)
    return df

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.set_page_config(page_title="KG Paragraph Checker", page_icon="🔥", layout="wide")
st.title("🔎 KG Paragraph Analyzer")

with st.sidebar:
    st.header("Neo4j Connection")
    st.write("These read from environment variables if left blank.")
    uri_in = st.text_input("NEO4J_URI", value=URI)
    user_in = st.text_input("NEO4J_USERNAME", value=USER)
    pwd_in = st.text_input("NEO4J_PASSWORD", value=PASSWORD, type="password")
    db_in = st.text_input("NEO4J_DATABASE", value=DB)

    st.header("OpenAI")
    openai_in = st.text_input("OPENAI_API_KEY", value=OPENAI_API_KEY, type="password")

    st.caption("Click **Apply** if you change any connection settings.")
    apply = st.button("Apply")

if apply:
    URI = uri_in
    USER = user_in
    PASSWORD = pwd_in
    DB = db_in
    OPENAI_API_KEY = openai_in
    st.success("Applied new settings.")

# Paragraph input
st.subheader("Input Paragraph")
default_para = (
    "The fourth season of Chicago Fire, an American drama television series with executive producer "
    "Dick Wolf, and producers Derek Haas, Michael Brandt, and Matt Olmstead, was ordered on February 5, 2015, "
    "by NBC, and premiered on October 13, 2015 and concluded on May 17, 2016. The season contained 24 episodes."
)
paragraph = st.text_area("Enter paragraph to analyze", value=default_para, height=160)

run = st.button("Analyze")

if run:
    try:
        with st.spinner("Connecting to Neo4j..."):
            driver = make_driver()
            st.success("Connected to Neo4j.")

        with st.spinner("Preparing LLM & embeddings..."):
            llm = make_llm()
            oa_client = make_openai_client()

        # Extract topic
        with st.spinner("Extracting main topic..."):
            topic = extract_main_topic(llm, paragraph)
        st.info(f"**Extracted Topic:** {topic}")

        # Fetch candidates and choose best match
        with st.spinner("Fetching candidate nodes..."):
            candidates = fetch_all_nodes_with_names(driver)
        if not candidates:
            st.warning("No nodes found in the graph.")
        else:
            best = find_most_similar_node(oa_client, topic, candidates)
            if not best:
                st.warning("Could not determine the most similar node.")
            else:
                st.success("Best match found.")
                c1, c2, c3, c4 = st.columns([2,2,1,2])
                with c1:
                    st.metric("Best Node ID", best["id"])
                with c2:
                    st.metric("Best Node Name", best["name"])
                with c3:
                    st.metric("Similarity", f"{best['similarity']:.3f}")
                with c4:
                    st.write("Labels:", ", ".join(best.get("graph_labels", [])))

                # Fetch neighbors
                with st.spinner("Fetching connections..."):
                    neighbors = fetch_connections(driver, best["id"])
                df = neighbors_to_dataframe(neighbors)

                st.subheader("Connections")
                if df.empty:
                    st.info("No connections found for this node.")
                else:
                    st.dataframe(df, use_container_width=True)

                # Optional: show raw JSON
                with st.expander("Raw neighbors JSON"):
                    st.code(json.dumps(neighbors, indent=2))

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        try:
            driver.close()
        except Exception:
            pass
