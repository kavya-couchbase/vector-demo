# vector-demo

# 🥭 AI Discovery: Unified Vector and AI Architecture

This repository demonstrates how to build a high-performance semantic discovery engine for e-commerce. By combining **Couchbase Capella** and **NVIDIA Llama 3.2**, we eliminate the "Sync-Gap" between transactional data and AI intelligence.

## 🚀 The Three Discovery Patterns
This demo showcases three distinct architectural patterns within a single platform:

1. **Pure Vector Search:** Ranked conceptual matching using `VECTOR_DISTANCE`. Perfect for broad intent queries like *"healthy morning energy"* where exact keywords don't exist.
2. **Vector + Category Filter:** Combines hard SQL metadata filtering (`WHERE category = $cat`) with semantic ranking. Ensures results stay within a specific "Aisle."
3. **Full Text Search (FTS):** Traditional high-speed keyword discovery, invoked directly via SQL++ for brand or specific attribute matching.

## 🛠️ Tech Stack
- **Data Platform:** [Couchbase Capella](https://www.couchbase.com/products/capella/) (Vector + Search + Query)
- **AI Service:** Hosted NVIDIA Llama 3.2-nv-embedqa-1b-v2 (1024-dim embeddings)
- **UI:** Streamlit
- **Model Integration:** Hosted natively on Capella AI Services
- **Dataset:** The repository includes data.json, a curated list of e-commerce products designed to showcase the power of semantic search.

## ⚙️ Installation & Setup
Follow these specific steps to get the environment running on your local machine.

### 1. Clone the Repository
Open your terminal and run:
```bash
git clone https://github.com/kavya-couchbase/vector-demo.git
cd vector-demo
```

### 2. Set Up Virtual Environment
This step is essential to isolate your project dependencies.
```bash
# Create the virtual environment
python -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Activate it (Windows)
.\venv\Scripts\activate
```

### 3. Install Dependencies
Once the environment is active, install the required Python SDKs:
```bash
pip install couchbase streamlit requests
```

### 4. Configuration
Before running the app, update your credentials in demo.py:
Open demo.py in your code editor.
Update the ENDPOINT, USER, PASS, and AI_KEY variables.

Ensure data.json is in the same directory as demo.py.
Ensure a primary index and the vector index is created and reference correctly

### 5. Launch the Demo
```bash
# Start the Streamlit application:
streamlit run demo.py
```

