import os
import traceback
import streamlit as st
import requests
from datetime import timedelta
from couchbase.exceptions import CouchbaseException
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, QueryOptions
from couchbase.auth import PasswordAuthenticator

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
ENDPOINT = os.getenv("CB_ENDPOINT", "couchbases://cb.et5osacve0qgiuxo.cloud.couchbase.com")
USER = os.getenv("CB_USERNAME", "")
PASS = os.getenv("CB_PASSWORD", "")
BUCKET_NAME = os.getenv("CB_BUCKET", "retail")
SCOPE_NAME = os.getenv("CB_SCOPE", "category")
COLLECTION_NAME = os.getenv("CB_COLLECTION", "bb_products")
AI_SERVICE_URL = ""
AI_KEY        = ""
EMBED_MODEL   = "nvidia/llama-3.2-nv-embedqa-1b-v2"

# ─────────────────────────────────────────────
#  Page setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="AI Discovery", page_icon="🥭", layout="wide")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Sora', sans-serif; }}

    .bb-header {{ background: linear-gradient(90deg, #002e5b 0%, #689f38 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem; }}
    .product-card {{ background: white; border: 1px solid #e0e0e0; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: transform 0.15s; }}
    .product-card:hover {{ transform: translateY(-2px); border-color: #689f38; }}
    .product-name {{ color: #1e293b; font-size: 1.1rem; font-weight: 700; }}
    .product-price {{ color: #689f38; font-size: 1.2rem; font-weight: bold; }}
    .product-meta {{ color: #64748b; font-size: 0.82rem; margin-top: 6px; }}
    .product-desc {{ color: #475569; font-size: 0.88rem; margin-top: 8px; line-height: 1.5; }}
   
    /* Badges & Scores */
    .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; color: white; margin-right: 8px; }}
    .badge-vector {{ background: #002e5b; }}
    .score-pill {{ background: #f1f5f9; padding: 4px 10px; border-radius: 6px; font-family: monospace; font-size: 0.85rem; color: #334155; float: right; border: 1px solid #e2e8f0; }}

    .tip-box {{ background: #f8fafc; border-left: 3px solid #689f38; border-radius: 0 10px 10px 0; padding: 0.75rem 1rem; margin-bottom: 1rem; color: #475569; font-size: 0.85rem; }}
</style>
<div class="bb-header">
    <h1>Vector Discovery 🥭</h1>
    <p>Unified AI Architecture with Capella & NVIDIA</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Couchbase connection (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Couchbase Capella...")
def connect_to_couchbase():
    if not PASS:
        raise ValueError(
            "Missing Couchbase password. Set the CB_PASSWORD environment variable "
            "or update PASS in demo.py."
        )

    auth = PasswordAuthenticator(USER, PASS)
    options = ClusterOptions(auth)
    options.apply_profile("wan_development")

    cluster = Cluster(ENDPOINT, options)
    cluster.wait_until_ready(timedelta(seconds=10))

    # Validate the data path early so connectivity issues show up clearly.
    cluster.bucket(BUCKET_NAME).scope(SCOPE_NAME).collection(COLLECTION_NAME)
    return cluster


try:
    cluster = connect_to_couchbase()
except CouchbaseException as err:
    st.error(
        "Couchbase connection failed. Check the connection string, username, password, "
        "cluster allowlist, and whether your network can reach Capella."
    )
    st.code("".join(traceback.format_exception_only(type(err), err)).strip())
    st.stop()
except Exception as err:
    st.error("App startup failed while configuring Couchbase.")
    st.code("".join(traceback.format_exception_only(type(err), err)).strip())
    st.stop()

# ─────────────────────────────────────────────
#  Embedding helper
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def get_embedding(text: str) -> list[float]:
    headers = {"Authorization": f"Bearer {AI_KEY}"}
    payload = {"input": text, "model": EMBED_MODEL}
    resp = requests.post(AI_SERVICE_URL, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]

# ─────────────────────────────────────────────
#  Result renderer
# ─────────────────────────────────────────────
def render_product(row: dict, badge_html: str = "", score_html: str = ""):
    """Fixed signature to accept 3 arguments"""
    name  = row.get("name", "—")
    price = row.get("price", "—")
    desc  = row.get("description", "")
    brand = row.get("brand", "")
    cat   = row.get("category", "")
    meta_str = " · ".join(p for p in [brand, cat] if p)

    st.markdown(f"""
    <div class="product-card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <span class="product-name">{name}</span>
        <span class="product-price">₹{price}</span>
      </div>
      <div style="margin-top:8px">{badge_html}{score_html}</div>
      {f'<div class="product-meta">{meta_str}</div>' if meta_str else ''}
      {f'<div class="product-desc">{desc}</div>' if desc else ''}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🔵  Pure Vector Search",
    "🟢  Vector + Category Filter",
    "🟣  FTS-keyword search",
])

# ══════════════════════════════════════════════
#  TAB 1 — Pure Vector Search
# ══════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="tip-box">
      <strong>How it works:</strong> Ranked by semantic similarity using <code>VECTOR_DISTANCE</code>.
      Lower distance values indicate a stronger conceptual match.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        v_query = st.text_input("Semantic query", value="probiotic strawberry greek yogurt thick creamy high protein cultures", key="v_query")
    with col2:
        v_limit = st.number_input("Results", min_value=1, max_value=20, value=5, key="v_limit")

    if st.button("🔍 Run Vector Search", key="btn_vector"):
        with st.spinner("Embedding query..."):
            vector = get_embedding(v_query)
        with st.spinner("Searching Couchbase..."):
            sql = """
                SELECT p.name, p.price, p.description, p.brand, p.category,
                       APPROX_VECTOR_DISTANCE(p.embedding, $vector, "l2") AS vec_dist
                FROM `retail`.`category`.`bb_products` AS p
                USE INDEX (`vector_index`)
                WHERE p.embedding IS NOT MISSING
                ORDER BY APPROX_VECTOR_DISTANCE(p.embedding, $vector, "l2")
                LIMIT $lim
            """
            try:
                results = cluster.query(sql, QueryOptions(named_parameters={"vector": vector, "lim": int(v_limit)}))
                for row in results:
                    dist = row.get("vec_dist")
                    dist_str = f"{dist:.4f}" if dist is not None else "n/a"
                    render_product(
                        row,
                        '<span class="badge badge-vector">Vector</span>',
                        f'<span class="score-pill">dist&nbsp;{dist_str}</span>'
                    )
            except Exception as e:
                st.error(f"Vector search failed: {e}")

# ══════════════════════════════════════════════
#  TAB 2 — Vector + Category Filter
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="tip-box">
      <strong>How it works:</strong> Combines a hard SQL <code>WHERE</code> filter with vector similarity ranking.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        comp_query = st.text_input("Semantic intent", value="high protein healthy option", key="comp_query")
    with col2:
        cat_filter = st.selectbox("Category", ["Fresh Vegetables", "Fresh Fruits", "Dairy", "Staples", "Snacks", "Health & Wellness"], key="cat_filter")
    with col3:
        c_limit = st.number_input("Results", min_value=1, max_value=20, value=5, key="c_limit")

    if st.button("🔍 Run Filtered Vector Search", key="btn_filter"):
        with st.spinner("Embedding query..."):
            vector = get_embedding(comp_query)
        with st.spinner("Searching Couchbase..."):
            # Fixed SQL to include VECTOR_DISTANCE in SELECT
            sql = """
                SELECT p.name, p.price, p.category, p.brand, p.description,
                       APPROX_VECTOR_DISTANCE(p.embedding, $vector, "l2") AS vec_dist
                FROM `retail`.`category`.`bb_products` AS p
                USE INDEX (`vector_index`)
                WHERE p.category = $cat
                  AND p.embedding IS NOT MISSING
                ORDER BY APPROX_VECTOR_DISTANCE(p.embedding, $vector, "l2")
                LIMIT $lim
            """
            try:
                results = cluster.query(sql, QueryOptions(named_parameters={"vector": vector, "cat": cat_filter, "lim": int(c_limit)}))
                for row in results:
                    dist = row.get("vec_dist")
                    dist_str = f"{dist:.4f}" if dist is not None else "n/a"
                    render_product(
                        row,
                        '<span class="badge badge-vector">Vector</span>',
                        f'<span class="score-pill">dist&nbsp;{dist_str}</span>'
                    )
            except Exception as e:
                st.error(f"Filtered vector search failed: {e}")

# ══════════════════════════════════════════════
#  TAB 3 — Keyword Search (FTS)
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="tip-box">
      <strong>Full Text Search (FTS):</strong><br>
      This search uses the Couchbase Search engine to find exact keyword matches.
      It uses <code>conjuncts</code> (AND) for the brand and <code>disjuncts</code> (OR) for the keywords.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        brand_val = st.text_input("Brand keyword:", value="Amul", key="fts_brand")
    with col2:
        match_val = st.text_input("Search for:", value="milk", key="fts_match")
    with col3:
        f_limit = st.number_input("Results", min_value=1, max_value=20, value=5, key="fts_limit")

    if st.button("Run Keyword Search", key="btn_fts_search"):
        with st.spinner("Searching keywords..."):
            # The simplified SQL++ query using SEARCH()
            sql = """
                SELECT p.name, p.brand, p.category, p.description, p.price
                FROM `retail`.`category`.`bb_products` AS p
                WHERE SEARCH(p, {
                    "query": {
                        "conjuncts": [
                            {"field": "brand", "match": $brand},
                            {
                                "disjuncts": [
                                    {"field": "description", "match": $keyword}
                                ]
                            }
                        ]
                    }
                })
                LIMIT $lim
            """
            try:
                # Execute using named parameters for safety
                results = list(cluster.query(
                    sql,
                    QueryOptions(named_parameters={
                        "brand": brand_val,
                        "keyword": match_val,
                        "lim": int(f_limit)
                    })
                ))

                if not results:
                    st.info(f"No bb_products found matching brand '{brand_val}' and keyword '{match_val}'.")
               
                for row in results:
                    render_product(
                        row,
                        '<span class="badge badge-filter">FTS Search</span>',
                        '<span class="score-pill score-vector">Keyword Match</span>'
                    )

            except Exception as e:
                st.error(f"Search failed: {e}")

