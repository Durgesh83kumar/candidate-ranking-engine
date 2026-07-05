import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import time
import tempfile
import re
import uuid
import faiss
import docx
from pypdf import PdfReader

# Add local root to sys.path
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Create temporary upload directory relative to project root
TEMP_DIR = str(ROOT / "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# Define absolute paths based on project root to be safe from working directory variations
local_index_path = str(ROOT / "output" / "faiss.index")
local_metadata_path = str(ROOT / "output" / "processed_candidates.parquet")
local_search_docs_path = str(ROOT / "output" / "search_documents.parquet")
hiring_specification_path = str(ROOT / "output" / "hiring_specification.json")
search_queries_path = str(ROOT / "output" / "search_queries.json")
index_metadata_path = str(ROOT / "output" / "index_metadata.json")

# Backend imports
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.reader import CandidateReader
from src.jd_intelligence.parser import JDParser
from src.jd_intelligence.extractor import LlmExtractor
from src.jd_intelligence.validator import SpecificationValidator
from src.jd_intelligence.query_gen import QueryGenerator
from src.document_generation.pipeline import DocumentGenerationPipeline
from src.indexing.backends.sentence_transformers import SentenceTransformersBackend
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.config import RetrievalConfig
from src.reranker.scorer import BatchRerankingScorer
from src.ranking.scorer import HybridScorer
from src.ranking.honeypot import HoneypotDetector
from src.verification.calibrator import ScoreCalibrator
from src.verification.config import VerificationConfig
from src.verification.cli import validate_and_save_submission


# ---------------------------------------------------------
# ST PAGE CONFIG & RICH AESTHETICS STYLE
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Recruiter Match & Screening Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Gradients, Glassmorphism, Rounded Cards)
st.markdown("""
<style>
    .main {
        background-color: #0f111a;
        color: #ffffff;
    }
    .stApp {
        background: radial-gradient(circle at 30% 30%, #1a1e36 0%, #0c0e17 100%);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .badge {
        background-color: #6366f1;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-alert {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-success {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL CACHING RESOURCES
# ---------------------------------------------------------
@st.cache_resource
def get_embedding_backend():
    """Instantiates and caches the sentence-transformers embedding backend."""
    try:
        from src.indexing.backends.sentence_transformers import SentenceTransformersBackend
        backend = SentenceTransformersBackend()
        backend.initialize("BAAI/bge-small-en-v1.5", {})
        return backend
    except Exception as e:
        st.error(f"Error loading embedding backend model: {str(e)}")
        return None

@st.cache_resource
def get_reranker_backend():
    """Instantiates and caches the CrossEncoder model on CPU."""
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    except Exception as e:
        st.error(f"Error loading Cross-Encoder reranker: {str(e)}")
        return None

@st.cache_resource
def load_cached_faiss_index(index_path):
    """Loads and caches the local FAISS index file."""
    if os.path.exists(index_path):
        try:
            return faiss.read_index(index_path)
        except Exception as e:
            st.error(f"Error reading FAISS Index file: {str(e)}")
            return None
    return None

# ---------------------------------------------------------
# CONVENIENCE HELPERS
# ---------------------------------------------------------
def save_uploaded_file(uploaded_file):
    """Saves streamlit file upload to local temp directory."""
    if uploaded_file is None:
        return None
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_text_from_pdf(pdf_path):
    """Reads all pages of a PDF and returns raw compiled text."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF file: {str(e)}"

def extract_text_from_docx(docx_path):
    """Reads docx file paragraph text."""
    try:
        doc = docx.Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"Error reading Word document: {str(e)}"

def extract_text_from_any(file_path):
    """Parses text content from PDF, DOCX, or TXT formats."""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def extract_experience_years(text: str) -> float:
    """Parses text dynamically to extract years of work experience, avoiding the Education Trap."""
    text_lower = text.lower()
    
    # Check for explicit "X years of experience" mention
    exp_match = re.search(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:of\s+)?experience", text_lower)
    if exp_match:
        return float(exp_match.group(1))

    # Parse lines to identify sections and calculate date spans
    lines = text.split("\n")
    in_education = False
    in_experience = False
    
    experience_spans = []
    
    # Common date regexes:
    # 1. YYYY - YYYY (e.g., 2021 - 2024)
    # 2. MMM YYYY - MMM YYYY (e.g., Jan 2021 - Dec 2023) or Jan 2021 - Present
    # 3. MM/YYYY - MM/YYYY
    date_range_regex = re.compile(
        r"\b((?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*|\d{1,2})[\s/,-]*)?(\d{4})\s*(?:-|to|until|—)\s*((?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*|\d{1,2})[\s/,-]*)?(\d{4}|present)\b",
        re.IGNORECASE
    )

    for line in lines:
        line_clean = line.strip().lower()
        if not line_clean:
            continue
            
        # Detect Section Headers
        edu_headers = ["education", "academic", "university", "college", "degree", "school", "coursework"]
        exp_headers = ["experience", "employment", "work history", "professional", "career", "job", "position"]
        
        # If line is short and contains header keywords, toggle context
        if len(line_clean) < 50:
            if any(h in line_clean for h in edu_headers):
                in_education = True
                in_experience = False
                continue
            elif any(h in line_clean for h in exp_headers):
                in_experience = True
                in_education = False
                continue
                
        # If we are in education, skip scanning this line for experience dates
        if in_education:
            continue
            
        # Find date ranges
        matches = date_range_regex.findall(line)
        for match in matches:
            start_year = int(match[1])
            end_val = match[3].lower()
            
            if end_val == "present":
                end_year = 2026
            else:
                try:
                    end_year = int(end_val)
                except ValueError:
                    end_year = start_year
                    
            if 1980 < start_year <= 2026 and 1980 < end_year <= 2026:
                diff = end_year - start_year
                if diff >= 0:
                    experience_spans.append(diff)

    if experience_spans:
        total_exp = sum(experience_spans)
        return min(30.0, float(total_exp))
        
    return 0.0

def parse_uploaded_resumes(file_paths):
    """Processes uploaded resumes into structured mock candidate formats."""
    candidates = []
    for path in file_paths:
        name = os.path.basename(path)
        text = extract_text_from_any(path)
        cid = f"CAND_UP_{uuid.uuid4().hex[:6].upper()}"
        
        # Heuristically discover skills list
        skills = []
        tech_words = [
            "python", "pytorch", "tensorflow", "keras", "transformers", "huggingface", 
            "langchain", "llamaindex", "faiss", "milvus", "pinecone", "scikit-learn", 
            "numpy", "pandas", "scipy", "sql", "aws", "gcp", "docker", "kubernetes", 
            "java", "c++", "golang", "react", "fastapi", "flask", "django"
        ]
        for w in tech_words:
            if re.search(rf"\b{w}\b", text.lower()):
                skills.append(w)
                
        # Discover experience years dynamically
        years_exp = extract_experience_years(text)
            
        # Parse timeline anomalies to simulate honeypots
        triggered_honeypots = []
        # Mock timeline inconsistency checks
        if "gpt-4" in text.lower() and re.search(r"2018|2019", text.lower()) and "experience" in text.lower():
            triggered_honeypots.append("GPT-4 experience claimed in 2018 (Prior to GPT-4 release)")
        if "llama" in text.lower() and re.search(r"2015|2016", text.lower()) and "experience" in text.lower():
            triggered_honeypots.append("Llama model tenure claimed in 2015 (Prior to Llama release)")

        # Prepare experience positions
        experience_positions = []
        if years_exp > 0.0:
            start_year = int(2026 - years_exp)
            experience_positions.append({
                "title": "AI Engineer" if "ai" in text.lower() or "machine learning" in text.lower() else "Software Engineer",
                "start_date": f"{start_year}-01-01",
                "end_date": "2025-12-31",
                "is_ongoing": False,
                "description": text[:600]
            })

        candidate_record = {
            "candidate_id": cid,
            "display_name": name,
            "years_of_experience": years_exp,
            "skills": [{"name_normalized": s, "name_raw": s, "duration_months": int(years_exp * 12) if years_exp > 0 else 6} for s in skills],
            "skills_raw": ", ".join(skills),
            "search_document_v2": text[:3000],  # Compile profile text
            "profile": {
                "years_of_relevant_ai_experience": years_exp if any(x in skills for x in ["pytorch", "tensorflow", "transformers", "langchain"]) else 0.0
            },
            "experience_positions": experience_positions,
            "triggered_honeypot_checks": triggered_honeypots
        }
        candidates.append(candidate_record)
    return candidates

# ---------------------------------------------------------
# SIDEBAR CONFIGURATION WIDGETS
# ---------------------------------------------------------
st.sidebar.title("⚙️ Engine Configuration")

# Core selection
mode = st.sidebar.selectbox(
    "Execution Mode",
    ["Search 100K Candidate Database", "Screen Uploaded Resumes"],
    help="Select whether to search our offline candidate pool or screen new uploaded resumes on the fly."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Weighted Scoring Parameters")
w_retrieval = st.sidebar.slider("Retrieval Score Weight (%)", 0, 100, 20)
w_ce = st.sidebar.slider("Cross-Encoder Weight (%)", 0, 100, 40)
w_career = st.sidebar.slider("Career Quality Weight (%)", 0, 100, 25)
w_profile = st.sidebar.slider("Profile Quality Weight (%)", 0, 100, 15)

total_weight = w_retrieval + w_ce + w_career + w_profile
if total_weight != 100:
    st.sidebar.warning(f"Total weights must equal 100%. Current sum: {total_weight}%")
    # Normalize weights internally
    w_retrieval_n = w_retrieval / total_weight
    w_ce_n = w_ce / total_weight
    w_career_n = w_career / total_weight
    w_profile_n = w_profile / total_weight
else:
    w_retrieval_n = w_retrieval / 100.0
    w_ce_n = w_ce / 100.0
    w_career_n = w_career / 100.0
    w_profile_n = w_profile / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("Recruitment Soft Filters")
min_years_exp = st.sidebar.slider("Min Years of Experience", 0.0, 15.0, 5.0)
max_notice_days = st.sidebar.slider("Max Notice Period (Days)", 15, 90, 60)

# ---------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------
st.title("🎯 AI Candidate Discovery & Ranking System")
st.markdown("Offline, CPU-optimized hybrid search, neural re-ranking, and deterministic candidate verification dashboard.")

# Tab navigation
tab1, tab2, tab3, tab4 = st.tabs(["📥 Upload & Ingest", "📊 Rankings Table", "🔎 Candidate Insights", "⚙️ System Engine Stats"])

# Initialize session states
if "processed_results" not in st.session_state:
    st.session_state.processed_results = None
if "hiring_spec" not in st.session_state:
    st.session_state.hiring_spec = None
if "search_queries" not in st.session_state:
    st.session_state.search_queries = None
if "metadata_db" not in st.session_state:
    st.session_state.metadata_db = None

# ---------------------------------------------------------
# DEPLOYMENT DIAGNOSTICS & VERIFICATION CHECKLIST (CHECKLIST 3 & 8)
# ---------------------------------------------------------
output_dir = ROOT / "output"
output_exists = output_dir.exists()
output_contents = os.listdir(str(output_dir)) if output_exists else []

# Compute checklist items
chk_output_exists = output_exists
chk_faiss_exists = os.path.exists(local_index_path)
chk_metadata_exists = os.path.exists(local_metadata_path)
chk_search_docs_exists = os.path.exists(local_search_docs_path)
chk_index_meta_exists = os.path.exists(index_metadata_path)

# 1. Print diagnostics to container logs (stdout)
print("=== DEPLOYMENT STARTUP DIAGNOSTICS ===", flush=True)
print(f"Current Working Directory (CWD): {os.getcwd()}", flush=True)
print(f"Application Root Path: {ROOT}", flush=True)
print(f"Output Directory Exists: {chk_output_exists}", flush=True)
print(f"Output Directory Contents: {output_contents}", flush=True)
print(f"Absolute Path Checked for faiss.index: {local_index_path} (Exists: {chk_faiss_exists})", flush=True)
print(f"Absolute Path Checked for processed_candidates.parquet: {local_metadata_path} (Exists: {chk_metadata_exists})", flush=True)
print(f"Absolute Path Checked for search_documents.parquet: {local_search_docs_path} (Exists: {chk_search_docs_exists})", flush=True)
print(f"Absolute Path Checked for index_metadata.json: {index_metadata_path} (Exists: {chk_index_meta_exists})", flush=True)
print("======================================", flush=True)

# 2. Render diagnostics and checklist in Streamlit Sidebar
with st.sidebar:
    st.markdown("---")
    with st.expander("🔍 Deployment Verification Checklist", expanded=True):
        st.markdown("### Verification Status")
        
        def render_status(label, passed):
            if passed:
                st.markdown(f"**✓ {label}**: :green[PASS]")
            else:
                st.markdown(f"**✗ {label}**: :red[FAIL]")
                
        render_status("output directory exists", chk_output_exists)
        render_status("faiss.index exists", chk_faiss_exists)
        render_status("processed_candidates.parquet exists", chk_metadata_exists)
        render_status("search_documents.parquet exists", chk_search_docs_exists)
        render_status("index_metadata.json exists", chk_index_meta_exists)
        
        st.markdown("---")
        st.markdown("**Diagnostics Data:**")
        st.caption(f"**CWD:** `{os.getcwd()}`")
        st.caption(f"**App Root:** `{ROOT}`")
        st.caption(f"**Output Files:** `{output_contents}`")

# Pre-load local metadata DB if mode requires it
if st.session_state.metadata_db is None and os.path.exists(local_metadata_path):
    try:
        st.session_state.metadata_db = pd.read_parquet(local_metadata_path)
    except Exception:
        pass

# ---------------------------------------------------------
# TAB 1: INPUT & PROCESSING ZONE
# ---------------------------------------------------------
with tab1:
    if not os.path.exists(local_index_path):
        st.info("ℹ️ Database not found. Please ensure faiss.index and parquet files are uploaded to the Space Files section.")
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Job Description File")
        jd_file = st.file_uploader("Upload JD Document (docx, txt)", type=["docx", "txt"])
        jd_text_input = st.text_area("Or paste Job Description text directly...", height=200)

    with col2:
        st.subheader("2. Upload Resume Profiles")
        resumes_uploaded = st.file_uploader("Upload Candidates Resumes (PDF, Word, or TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        if resumes_uploaded:
            st.success(f"Loaded {len(resumes_uploaded)} candidates resumes successfully.")
            
        if mode == "Search 100K Candidate Database":
            st.info("ℹ️ Database Search Mode Active: The engine will query the offline candidate database. Uploaded resumes will be ignored in this mode.")
            
            # Show availability indicators
            if os.path.exists(local_index_path):
                st.success(f"✔️ FAISS Index found ({os.path.basename(local_index_path)}) - Loaded in cache.")
            else:
                st.warning("⚠️ FAISS index file not found in 'output/faiss.index'. Run the index builder runner first.")
                
            if os.path.exists(local_metadata_path):
                st.success(f"✔️ Processed candidates database found ({os.path.basename(local_metadata_path)}).")
            else:
                st.warning("⚠️ Candidate metadata parquet file not found in 'output/'. Run preprocessing first.")

    st.markdown("---")
    
    # Run analysis trigger
    analyze_clicked = st.button("🚀 Execute Analysis Pipeline", use_container_width=True)
    
    if analyze_clicked:
        # Extract JD text
        jd_text = ""
        if jd_file is not None:
            temp_jd_path = save_uploaded_file(jd_file)
            jd_text = extract_text_from_any(temp_jd_path)
        elif jd_text_input.strip():
            jd_text = jd_text_input.strip()
            
        if not jd_text:
            st.error("No Job Description specified. Please upload a JD file or paste JD text.")
            st.stop()
            
        # Check files requirements
        # Check files requirements
        if mode == "Search 100K Candidate Database":
            if not os.path.exists(local_index_path) or not os.path.exists(local_metadata_path) or not os.path.exists(local_search_docs_path):
                st.error(f"Missing local vector database files in 'output/'.\n"
                         f"Checked Absolute Paths:\n"
                         f"- FAISS Index: {local_index_path} (Exists: {os.path.exists(local_index_path)})\n"
                         f"- Metadata Parquet: {local_metadata_path} (Exists: {os.path.exists(local_metadata_path)})\n"
                         f"- Search Docs Parquet: {local_search_docs_path} (Exists: {os.path.exists(local_search_docs_path)})\n"
                         f"Please ensure indexing and preprocessing runs successfully locally before uploading output files.")
                st.stop()
        else:
            if not resumes_uploaded:
                st.error("Screener mode requires at least one candidate resume upload.")
                st.stop()

        # Execute Pipeline using st.status for beautiful analysis logging
        with st.status("Initializing AI Discovery Engine...", expanded=True) as status:
            
            # Step 1: JD Extraction & Intelligence
            status.update(label="Phase 2: Extracting structural spec from Job Description...")
            jd_temp_path = os.path.join(TEMP_DIR, "uploaded_jd.txt")
            with open(jd_temp_path, "w", encoding="utf-8") as f:
                f.write(jd_text)
                
            extractor = LlmExtractor()
            try:
                spec = extractor.extract(jd_text)
                validator = SpecificationValidator()
                validator.validate(spec)
                query_gen = QueryGenerator()
                queries = query_gen.generate(spec)
                
                # Save parsed deliverables to output folder
                os.makedirs(str(ROOT / "output"), exist_ok=True)
                with open(hiring_specification_path, "w", encoding="utf-8") as f:
                    json.dump(spec.model_dump(), f, indent=2)
                with open(search_queries_path, "w", encoding="utf-8") as f:
                    json.dump(queries, f, indent=2)
                
                st.session_state.hiring_spec = spec
                st.session_state.search_queries = queries
                st.write("✔️ Successfully extracted structured Spec Schema.")
            except Exception as e:
                st.error(f"Failed parsing JD spec: {str(e)}")
                status.update(label="Analysis failed.", state="error")
                st.stop()

            # Load models from cache
            status.update(label="Loading pre-trained HuggingFace sentence transformer models...")
            emb_backend = get_embedding_backend()
            ce_backend = get_reranker_backend()
            
            if emb_backend is None or ce_backend is None:
                st.error("Failed loading models into memory.")
                status.update(label="Models initialization failed.", state="error")
                st.stop()
                
            st.write("✔️ Cached embedding models successfully initialized.")

            # Processing candidates based on mode
            if mode == "Search 100K Candidate Database":
                # Step 2: Vector Search & RRF Fusion (Phase 5)
                status.update(label="Phase 5: Querying FAISS index and running RRF fusion...")
                
                # Setup Retriever
                retrieval_filters = {
                    "min_experience": min_years_exp,
                    "max_notice_period_days": max_notice_days
                }
                retrieval_config = RetrievalConfig(
                    top_k_per_query=200,
                    output_pool_size=1000,
                    filters=retrieval_filters
                )
                
                retriever = SemanticRetriever(
                    config=retrieval_config,
                    spec_path=hiring_specification_path,
                    queries_path=search_queries_path,
                    index_path=local_index_path,
                    index_metadata_path=index_metadata_path,
                    candidates_parquet_path=local_metadata_path
                )
                
                retrieved_candidates, stats = retriever.retrieve_candidates()
                st.write(f"✔️ Retrieval completed. Ingested {len(retrieved_candidates)} candidates.")
                
                if len(retrieved_candidates) == 0:
                    st.warning("No candidates matched filters. Relax the filters in the sidebar and run again.")
                    status.update(label="Retrieval pool empty.", state="warning")
                    st.stop()

                # Step 3: Deep Cross-Encoder scoring (Phase 6)
                status.update(label="Phase 6: Scoring candidate search documents with Cross-Encoder...")
                
                from src.reranker.config import RerankerConfig
                from src.reranker.pair_builder import RerankingPairBuilder
                from src.reranker.explainability import ExplainabilityEngine
                
                reranker_config = RerankerConfig(
                    model_name="BAAI/bge-reranker-base",
                    batch_size=32,
                    device="cpu"
                )
                pair_builder = RerankingPairBuilder(spec_path=hiring_specification_path)
                scorer = BatchRerankingScorer(config=reranker_config)
                scorer.model = ce_backend # Bind cached CrossEncoder instance
                
                # Load search documents
                search_docs_df = pd.read_parquet(local_search_docs_path)
                
                # Build text pairs (Query + Candidate Doc)
                pairs = pair_builder.construct_pairs(retrieved_candidates, search_docs_df)
                
                # Execute inference
                scored_candidates = scorer.score_pairs(pairs)
                
                # Enrich candidate pool records
                explain_engine = ExplainabilityEngine()
                enriched_pool = []
                for cand in retrieved_candidates:
                    cid = cand["candidate_id"]
                    if cid in scored_candidates:
                        logit, prob = scored_candidates[cid]
                        cand_copy = dict(cand)
                        cand_copy["cross_encoder_logit"] = logit
                        cand_copy["cross_encoder_probability"] = prob
                        
                        # Map semantic similarity score
                        sims = cand.get("query_similarities", {})
                        cand_copy["semantic_similarity"] = float(pd.Series(list(sims.values())).mean()) if sims else 0.0
                        cand_copy["retrieval_score"] = cand.get("rrf_score", 0.0)
                        
                        # Generate explainability evidence
                        evidence = explain_engine.extract_evidence(cand_copy)
                        cand_copy["explainability_evidence"] = evidence
                        
                        enriched_pool.append(cand_copy)
                        
                # Sort and slice
                enriched_pool.sort(key=lambda x: x["cross_encoder_probability"], reverse=True)
                re_ranked = enriched_pool[:300]
                st.write(f"✔️ Cross-Encoder scoring completed for {len(re_ranked)} candidates.")

                # Step 4: Hybrid Scorer & Honeypot Checks (Phase 7)
                status.update(label="Phase 7: Calculating Unified Hybrid Scores and scanning for Honeypots...")
                from src.ranking.config import RankingConfig
                from src.ranking.scorer import HybridScorer
                from src.ranking.reasoning import ReasoningEngine
                
                max_rrf = max([c["rrf_score"] for c in re_ranked]) if re_ranked else 0.10
                if max_rrf <= 0:
                    max_rrf = 0.10
                
                ranking_config = RankingConfig(
                    weight_retrieval=w_retrieval_n,
                    weight_cross_encoder=w_ce_n,
                    weight_career=w_career_n,
                    weight_profile=w_profile_n
                )
                hybrid_scorer = HybridScorer(config=ranking_config)
                reasoning_engine = ReasoningEngine()
                
                scored_list = []
                for cand in re_ranked:
                    cid = cand["candidate_id"]
                    cand_copy = dict(cand)
                    
                    # Fetch metadata record
                    meta_rows = st.session_state.metadata_db[st.session_state.metadata_db["candidate_id"] == cid]
                    if not meta_rows.empty:
                        row_proc = meta_rows.iloc[0]
                        career_history = row_proc.get("career_history")
                        education = row_proc.get("education")
                        skills = row_proc.get("skills")
                        certifications = row_proc.get("certifications")
                        profile = row_proc.get("profile")
                    else:
                        career_history, education, skills, certifications, profile = [], [], [], [], {}
                    
                    if hasattr(career_history, "tolist"):
                        career_history = career_history.tolist()
                    if hasattr(education, "tolist"):
                        education = education.tolist()
                    if hasattr(skills, "tolist"):
                        skills = skills.tolist()
                    if hasattr(certifications, "tolist"):
                        certifications = certifications.tolist()
                        
                    cand_copy["career_history"] = career_history
                    cand_copy["education"] = education
                    cand_copy["skills"] = skills
                    cand_copy["certifications"] = certifications
                    cand_copy["profile"] = profile
                    
                    # Compute hybrid score
                    scores_breakdown = hybrid_scorer.compute_hybrid_score(cand_copy, max_rrf_score=max_rrf)
                    cand_copy.update(scores_breakdown)
                    
                    # Generate reasoning
                    reasoning = reasoning_engine.generate_reasoning(cand_copy)
                    cand_copy["reasoning"] = reasoning
                    
                    scored_list.append(cand_copy)
                    
                st.write("✔️ Score calculations finished.")

                # Step 5: Verification & Calibration (Phase 8)
                status.update(label="Phase 8: Calibrating multipliers and verifying mandatory requirements...")
                ver_config = VerificationConfig()
                calibrator = ScoreCalibrator(config=ver_config, spec_path=hiring_specification_path)
                
                calibrated_list = []
                for s in scored_list:
                    cid = s["candidate_id"]
                    cand_copy = dict(s)
                    
                    # Ingest search document and skills
                    meta_rows = st.session_state.metadata_db[st.session_state.metadata_db["candidate_id"] == cid]
                    if not meta_rows.empty:
                        row_proc = meta_rows.iloc[0]
                        cand_copy["skills"] = row_proc.get("skills")
                        cand_copy["search_document_v2"] = row_proc.get("search_document_v2")
                    else:
                        cand_copy["skills"] = []
                        cand_copy["search_document_v2"] = ""
                        
                    # Calibrate
                    cal_res = calibrator.calibrate_candidate(cand_copy)
                    cand_copy.update(cal_res)
                    calibrated_list.append(cand_copy)
                    
                # Re-sort and take Top 100
                calibrated_list.sort(key=lambda x: x["calibrated_score"], reverse=True)
                # Assign ranks
                for idx, c in enumerate(calibrated_list):
                    c["rank"] = idx + 1
                    
                st.session_state.processed_results = calibrated_list
                st.write("✔️ Candidate Verification and Ranks calibration complete.")
                status.update(label="Analysis complete! View results in the Rankings Table tab.", state="complete")
                
            else:
                # Screening Uploaded Resumes
                status.update(label="Parsing uploaded resume documents...")
                uploaded_files_paths = [save_uploaded_file(f) for f in resumes_uploaded]
                parsed_candidates = parse_uploaded_resumes(uploaded_files_paths)
                
                st.write(f"✔️ Preprocessed {len(parsed_candidates)} candidate profiles.")

                # On-the-fly similarity and neural scoring
                status.update(label="Encoding profiles and evaluating match similarities...")
                
                query_texts = list(queries.values())
                query_embeddings = emb_backend.compute_embeddings(query_texts)
                
                scored_list = []
                for pc in parsed_candidates:
                    # Compute similarity scores
                    doc_emb = emb_backend.compute_embeddings([pc["search_document_v2"]])[0]
                    sims = [np.dot(q_emb, doc_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb)) for q_emb in query_embeddings]
                    retrieval_score = float(np.max(sims))
                    
                    # Cross-Encoder neural score
                    ce_pairs = [(q_text, pc["search_document_v2"]) for q_text in query_texts]
                    ce_scores = ce_backend.predict(ce_pairs)
                    if np.isscalar(ce_scores):
                        ce_scores = np.array([ce_scores])
                    ce_score = float(np.max(ce_scores))
                    # Map to probability
                    ce_prob = 1.0 / (1.0 + np.exp(-ce_score))
                    
                    # Compute hybrid score
                    hybrid_score = (w_retrieval_n * retrieval_score) + (w_ce_n * ce_prob) + (w_career_n * 0.8) + (w_profile_n * 0.9)
                    
                    pc["retrieval_score"] = retrieval_score
                    pc["ce_score"] = ce_prob
                    pc["cross_encoder_probability"] = ce_prob
                    pc["final_score"] = hybrid_score
                    
                    # Run calibration directly
                    ver_config = VerificationConfig()
                    calibrator = ScoreCalibrator(config=ver_config, spec_path=hiring_specification_path)
                    res = calibrator.calibrate_candidate(pc)
                    pc.update(res)
                    scored_list.append(pc)
                    
                # Sort and Rank
                scored_list.sort(key=lambda x: x["calibrated_score"], reverse=True)
                for idx, c in enumerate(scored_list):
                    c["rank"] = idx + 1
                    
                st.session_state.processed_results = scored_list
                status.update(label="Resumes screening complete! View Rankings Table.", state="complete")
                
        # Render dynamic visual results in Tab 1 immediately
        if st.session_state.processed_results:
            if mode == "Screen Uploaded Resumes":
                st.write("---")
                st.subheader("Candidate Evaluation Results")
                for cand in st.session_state.processed_results:
                    score_pct = round(cand["calibrated_score"] * 100, 1)
                    
                    with st.container():
                        st.markdown(f"### 📄 Profile: {cand.get('display_name', cand['candidate_id'])}")
                        
                        col_m1, col_m2 = st.columns([1, 3])
                        with col_m1:
                            st.metric(label="Overall Match Score", value=f"{score_pct}%")
                        with col_m2:
                            st.info(cand["reasoning"])
                            
                        is_honeypot = cand.get("is_flaged_honeypot", False)
                        honeypots = cand.get("triggered_honeypot_checks", [])
                        if is_honeypot or honeypots:
                            for hp in honeypots:
                                st.error(f"🚨 FRAUD/HONEYPOT DETECTED: {hp}")
                            if not honeypots:
                                st.error("🚨 FRAUD/HONEYPOT DETECTED: Critical timeline or requirement anomaly identified.")
            else:
                st.write("---")
                st.success("✔️ Search complete! Top 100 matched profiles have been loaded into the Rankings Table tab.")

# ---------------------------------------------------------
# TAB 2: RANKINGS TABLE
# ---------------------------------------------------------
with tab2:
    if st.session_state.processed_results is None:
        st.info("Please upload your Job Description and execute analysis under the 'Upload & Ingest' tab.")
    else:
        results = st.session_state.processed_results
        st.subheader("🏆 Calibrated Candidate Rankings (Top 100)")
        
        # Build display table
        display_data = []
        for r in results[:100]:
            # Pull display name fallback if present
            display_name = r["candidate_id"]
            display_data.append({
                "Rank": r["rank"],
                "Candidate ID": display_name,
                "Calibrated Score": f"{r['calibrated_score']:.4f}",
                "Honeypot Risk": "⚠️ Flagged" if r["is_flaged_honeypot"] else "✔️ Clean",
                "Reasoning Description": r["reasoning"]
            })
            
        df_rankings = pd.DataFrame(display_data)
        st.dataframe(df_rankings, use_container_width=True)
        
        # Save and Download CSV
        csv_submission = []
        for r in results[:100]:
            csv_submission.append({
                "candidate_id": r["candidate_id"],
                "rank": r["rank"],
                "score": round(r["calibrated_score"], 4),
                "reasoning": r["reasoning"]
            })
        df_csv = pd.DataFrame(csv_submission)
        
        # Re-verify and save to csv string
        csv_buffer = df_csv.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download submission.csv",
            data=csv_buffer,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---------------------------------------------------------
# TAB 3: CANDIDATE INSIGHTS
# ---------------------------------------------------------
with tab3:
    if st.session_state.processed_results is None:
        st.info("Candidate insights will load once the analysis runs.")
    else:
        results = st.session_state.processed_results
        
        st.subheader("🔍 Selected Candidate Scorecard Details")
        candidate_ids = [r["candidate_id"] for r in results]
        selected_cid = st.selectbox("Select Candidate ID to audit:", candidate_ids)
        
        # Find candidate record
        cand_record = next((c for c in results if c["candidate_id"] == selected_cid), None)
        
        if cand_record:
            # Layout scorecard columns
            sc_col1, sc_col2, sc_col3 = st.columns(3)
            
            with sc_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Match Scoring</h4>
                    <p style="font-size:24px; font-weight:bold; color:#6366f1;">{cand_record['calibrated_score']:.4f}</p>
                    <small>Original Score: {cand_record['original_hybrid_score']:.4f}</small><br/>
                    <small>Phrase Multiplier: {cand_record['phrase_multiplier']:.2f}</small><br/>
                    <small>Skills Multiplier: {cand_record['skills_multiplier']:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
                
            with sc_col2:
                # Skill and framework badges
                st.markdown('<div class="metric-card"><h4>Skills and Frameworks Validation</h4>', unsafe_allow_html=True)
                if cand_record.get("matched_ai_frameworks"):
                    st.write("Matched AI Specialist Tools:")
                    for tool in cand_record["matched_ai_frameworks"]:
                        st.markdown(f'<span class="badge-success">{tool}</span>', unsafe_allow_html=True)
                else:
                    st.write("No specialized frameworks matched.")
                    
                if cand_record.get("missing_mandatory_skills"):
                    st.write("Missing Core Requirements:")
                    for skill in cand_record["missing_mandatory_skills"]:
                        st.markdown(f'<span class="badge-alert">{skill}</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-success">All mandatory requirements met</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with sc_col3:
                # Honeypot checks
                st.markdown('<div class="metric-card"><h4>Timeline & Security Audit</h4>', unsafe_allow_html=True)
                if cand_record["is_flaged_honeypot"]:
                    st.markdown('<span class="badge-alert">⚠️ Flagged Honeypot Check</span>', unsafe_allow_html=True)
                    st.write("Timeline / Claim inconsistencies detected. Candidate is disqualified from Top 100 selection.")
                else:
                    st.markdown('<span class="badge-success">✔️ Verification Audit Clean</span>', unsafe_allow_html=True)
                    st.write("No suspicious timeline overlaps or pre-release framework experience claimed.")
                st.markdown('</div>', unsafe_allow_html=True)

            # Show reasoning description
            st.info(f"**AI Generated Calibration Reasoning:** {cand_record['reasoning']}")

# ---------------------------------------------------------
# TAB 4: SYSTEM ENGINE STATS
# ---------------------------------------------------------
with tab4:
    st.subheader("📊 MLOps Engine Health & Parameters")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        import psutil
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        st.metric(label="CPU Core Load", value=f"{cpu_usage}%")
        st.metric(label="RAM Usage", value=f"{ram_usage}%")
        
    with col_stat2:
        st.write("**Active Model Specifications:**")
        st.markdown("*   **Retrieval Model:** `BAAI/bge-small-en-v1.5` (Cached)")
        st.markdown("*   **Re-ranking model:** `BAAI/bge-reranker-base` (Cached)")
        
    with col_stat3:
        if st.session_state.hiring_spec is not None:
            spec = st.session_state.hiring_spec
            st.write("**Extracted Role Metadata:**")
            st.write(f"- Role Title: `{spec.role.title}`")
            st.write(f"- Seniority Target: `{spec.role.seniority}`")
            st.write(f"- Required Experience: `{spec.experience.min_years} to {spec.experience.max_years} years`")
        else:
            st.write("Upload a Job Description to parse role metadata.")
