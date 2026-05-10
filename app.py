"""
GitBot - GitLab Handbook & Direction Chatbot
Main Streamlit Application
"""

import streamlit as st
import os
import sys
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Start model server in background (loads models once, stays alive) ─────────
# This runs before Streamlit renders anything — models are ready by the time
# the user enters their API key
try:
    from src.model_server import ensure_server_running
    ensure_server_running()
except Exception:
    pass  # Non-fatal — falls back to direct model loading

# Page configuration - MUST be first Streamlit call
st.set_page_config(
    page_title="GitBot - GitLab AI Assistant",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://handbook.gitlab.com",
        "About": "GitBot - AI-powered GitLab Handbook & Direction Assistant"
    }
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global styles */
* { font-family: 'Inter', sans-serif; }

/* Smooth Chat Animations */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

.stChatMessage {
    animation: slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="stHeader"] {background: transparent !important;}
.block-container {padding-top: 2rem !important;}

/* Animated Interactive Background */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp {
    background: linear-gradient(-45deg, #0f0f23, #1a1a3e, #251330, #0f0f23);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    min-height: 100vh;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #151528 0%, #1e1e3f 100%);
    border-right: 1px solid rgba(252, 109, 38, 0.2);
    box-shadow: 4px 0 20px rgba(0, 0, 0, 0.4);
}

[data-testid="stSidebar"] .stMarkdown {
    color: #e0e0e0;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #fc6d26 0%, #e24329 50%, #fca326 100%);
    padding: 20px 30px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(252, 109, 38, 0.3);
    display: flex;
    align-items: center;
    gap: 15px;
    flex-wrap: wrap;
}

.main-header h1 {
    color: white;
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.main-header p {
    color: rgba(255,255,255,0.9);
    margin: 5px 0 0 0;
    font-size: 0.95rem;
}



/* Source cards */
.source-card {
    background: rgba(252, 109, 38, 0.08);
    border: 1px solid rgba(252, 109, 38, 0.25);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.source-card:hover {
    background: rgba(252, 109, 38, 0.15);
    border-color: rgba(252, 109, 38, 0.5);
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
}

.source-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-handbook {
    background: rgba(66, 133, 244, 0.2);
    color: #4285f4;
    border: 1px solid rgba(66, 133, 244, 0.3);
}

.badge-direction {
    background: rgba(52, 168, 83, 0.2);
    color: #34a853;
    border: 1px solid rgba(52, 168, 83, 0.3);
}

/* Confidence meter */
.confidence-bar {
    height: 6px;
    border-radius: 3px;
    background: rgba(255,255,255,0.1);
    margin-top: 6px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}

/* Suggestion chips */
.suggestion-chip {
    display: inline-block;
    background: rgba(252, 109, 38, 0.1);
    border: 1px solid rgba(252, 109, 38, 0.3);
    color: #fc6d26;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    cursor: pointer;
    margin: 4px;
    transition: all 0.2s ease;
}

.suggestion-chip:hover {
    background: rgba(252, 109, 38, 0.25);
    border-color: rgba(252, 109, 38, 0.6);
    transform: translateY(-1px);
}

/* Stats cards */
.stat-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: all 0.2s ease;
}

.stat-card:hover {
    background: rgba(255, 255, 255, 0.08);
    transform: translateY(-2px);
}

.stat-number {
    font-size: 1.8rem;
    font-weight: 700;
    color: #fc6d26;
}

.stat-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.6);
    margin-top: 4px;
}

/* Input area */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1px solid rgba(252, 109, 38, 0.3) !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(252, 109, 38, 0.7) !important;
    box-shadow: 0 0 0 2px rgba(252, 109, 38, 0.15) !important;
}

.stTextInput > div > div > input::placeholder {
    color: rgba(255,255,255,0.35) !important;
}

/* Buttons */
[data-testid="stBaseButton-secondary"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(252, 109, 38, 0.3) !important;
    color: #e0e0e0 !important;
    border-radius: 20px !important;
    padding: 10px 24px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(252, 109, 38, 0.15) !important;
    border-color: rgba(252, 109, 38, 0.6) !important;
    transform: translateY(-2px) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(252, 109, 38, 0.15) !important;
}

[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #fc6d26, #e24329) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 4px 15px rgba(252, 109, 38, 0.3) !important;
}

[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(252, 109, 38, 0.4) !important;
}

/* Dividers */
hr {
    border-color: rgba(255,255,255,0.1) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: rgba(252, 109, 38, 0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(252, 109, 38, 0.7); }

/* Alert boxes */
.stAlert {
    border-radius: 10px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
}

/* Select box */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(252, 109, 38, 0.3) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.6) !important;
    border-radius: 8px !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #fc6d26, #e24329) !important;
    color: white !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #fc6d26 !important;
}

/* Welcome card */
.welcome-card {
    background: linear-gradient(135deg, rgba(252,109,38,0.1), rgba(226,67,41,0.05));
    border: 1px solid rgba(252,109,38,0.2);
    border-radius: 16px;
    padding: 24px;
    margin: 10px 0;
    text-align: center;
    animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}

/* Typing indicator */
.typing-indicator {
    display: flex;
    gap: 4px;
    padding: 10px 14px;
    background: rgba(255,255,255,0.07);
    border-radius: 18px;
    width: fit-content;
    margin: 10px 0;
}

.typing-dot {
    width: 8px;
    height: 8px;
    background: #fc6d26;
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-8px); opacity: 1; }
}

/* Streamlit chat input */
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1px solid rgba(252, 109, 38, 0.4) !important;
    border-radius: 14px !important;
}

[data-testid="stChatInput"] textarea {
    color: white !important;
    font-size: 0.95rem !important;
    background: transparent !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,0.35) !important;
}

[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #fc6d26, #e24329) !important;
    border-radius: 10px !important;
}

/* Feedback buttons */
.feedback-btn {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.5);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s;
}

.feedback-btn:hover {
    background: rgba(255,255,255,0.08);
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "messages": [],
        "chatbot": None,
        "index": None,
        "chunks": None,
        "initialized": False,
        "data_loaded": False,
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "total_queries": 0,
        "session_start": datetime.now().strftime("%H:%M"),
        "feedback": {},
        "show_sources": True,
        "selected_category": "All",
        "theme": "dark",
        "loading_data": False,
        "env_file_key": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ─── .env Auto-Reload ──────────────────────────────────────────────────────────
def sync_api_key_from_env():
    """
    Read GEMINI_API_KEY directly from .env on every render.
    If it changed since last render, update session state and
    clear the cached chatbot so it re-initializes with the new key.
    """
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    try:
        key_from_file = ""
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    key_from_file = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

        if key_from_file and key_from_file != st.session_state.get("env_file_key", ""):
            st.session_state.env_file_key = key_from_file
            st.session_state.api_key = key_from_file
            st.session_state.initialized = False
            st.cache_resource.clear()

    except OSError:
        pass  # File read error — silently ignore


# Call once at module load
sync_api_key_from_env()

# ─── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_chatbot_resources(api_key: str):
    """Load and cache ALL resources — index, models, BM25, reranker."""
    import concurrent.futures
    from src.scraper import load_scraped_data
    from src.chunker import create_chunks_from_documents, load_chunks, save_chunks
    from src.embeddings import (build_index, load_index, index_exists,
                                 get_embedding_model, get_reranker,
                                 build_bm25_index)
    from src.chatbot import GitLabChatbot

    # ── 1. Load FAISS index (fast, ~40ms) ─────────────────────────────────
    chunks_path = "data/chunks.json"
    index_path = "data/faiss_index"

    if index_exists(index_path):
        index, chunks = load_index(index_path, chunks_path)
    else:
        raise RuntimeError(f"Knowledge base not found at {index_path}. Please run the scraper script before deploying.")

    # ── 2. Load models — use server if running, else parallel load ────────
    try:
        from src.model_server import ModelClient
        if ModelClient()._is_server_up():
            # Server already has models loaded — just build BM25 and warm socket
            build_bm25_index(chunks)
            # Warm up the socket connection with a dummy encode
            ModelClient().encode(["warmup"])
        else:
            raise RuntimeError("Server not up")
    except Exception:
        # Fallback: load models in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f_embed   = executor.submit(get_embedding_model)
            f_rerank  = executor.submit(get_reranker)
            f_bm25    = executor.submit(build_bm25_index, chunks)
            f_embed.result()
            f_rerank.result()
            f_bm25.result()

    # ── 3. Initialize chatbot ──────────────────────────────────────────────
    bot = GitLabChatbot()
    bot.api_key = api_key
    success = bot.initialize(index, chunks)

    return bot, index, chunks, success


def get_confidence_color(confidence: float) -> str:
    """Return color based on confidence score."""
    if confidence >= 70:
        return "#34a853"  # Green
    elif confidence >= 45:
        return "#fca326"  # Orange
    else:
        return "#ea4335"  # Red


def get_confidence_label(confidence: float) -> str:
    if confidence >= 70:
        return "High"
    elif confidence >= 45:
        return "Medium"
    else:
        return "Low"


def render_message(msg: dict, idx: int):
    """Render a single chat message using native Streamlit chat components."""
    role = msg.get("role", "")
    content = msg.get("content") or ""
    timestamp = msg.get("timestamp") or ""
    sources = msg.get("sources") or []
    confidence = msg.get("confidence") or 0

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
            st.caption(timestamp)
    else:
        with st.chat_message("assistant", avatar="🦊"):
            # Render content
            st.markdown(content)
            
            # Render confidence badge
            if confidence:
                conf_color = get_confidence_color(confidence)
                conf_label = get_confidence_label(confidence)
                st.markdown(
                    f'<span style="font-size:0.75rem; color:{conf_color}; '
                    f'background:rgba(0,0,0,0.2); padding:2px 8px; border-radius:10px; display:inline-block; margin-top:8px;">'
                    f'● {conf_label} confidence</span>',
                    unsafe_allow_html=True
                )

            # Render sources
            if sources and st.session_state.get("show_sources", True):
                with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                    for src in sources:
                        cat = src.get("category", "")
                        badge_class = "badge-handbook" if cat == "handbook" else "badge-direction"
                        badge_label = "Handbook" if cat == "handbook" else "Direction"
                        src_conf = src.get("confidence", 0)
                        src_color = get_confidence_color(src_conf)
                        section_html = (
                            f"<br><small style='color: rgba(255,255,255,0.5);'>"
                            f"{src.get('section', '')}</small>"
                            if src.get('section') else ""
                        )
                        html_payload = f"""<div class="source-card">
<span class="source-badge {badge_class}">{badge_label}</span>
<strong style="color: #e0e0e0; margin-left: 8px;">{src.get('title', 'GitLab')}</strong>{section_html}
<br>
<a href="{src.get('url', '#')}" target="_blank" style="color: #fc6d26; font-size: 0.8rem; text-decoration: none;">
🔗 {src.get('url', '')[:60]}...
</a>
<div class="confidence-bar">
<div class="confidence-fill" style="width: {src_conf}%; background: {src_color};"></div>
</div>
<small style="color: rgba(255,255,255,0.4);">Relevance: {src_conf:.0f}%</small>
</div>"""
                        st.markdown(html_payload, unsafe_allow_html=True)
            
            # Render timestamp
            st.caption(timestamp)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo & Title
        st.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 3rem; animation: slideUp 0.5s ease;">🦊</div>
            <h2 style="color: #fc6d26; margin: 5px 0; font-size: 1.4rem; font-weight: 700;">GitBot</h2>
            <p style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin: 0;">
                GitLab AI Assistant
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Primary Actions at the top for easy access
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear", use_container_width=True, type="primary"):
                st.session_state.messages = []
                st.session_state.total_queries = 0
                st.toast("🧹 Chat history cleared!", icon="✨")
        with col2:
            if st.session_state.messages:
                chat_export = json.dumps(st.session_state.messages, indent=2, default=str)
                st.download_button(
                    "💾 Export",
                    data=chat_export,
                    file_name=f"gitbot_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True,
                    type="primary"
                )
            else:
                st.button("💾 Export", use_container_width=True, disabled=True)

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        # Configuration Expander
        with st.expander("⚙️ Settings & Configuration", expanded=not bool(st.session_state.api_key)):
            
            api_key_input = st.text_input(
                "Gemini API Key",
                value="",
                type="password",
                placeholder="Enter key to override..." if os.getenv("GEMINI_API_KEY") else "Enter your key...",
            )

            if api_key_input and api_key_input != st.session_state.api_key:
                st.session_state.api_key = api_key_input
                st.session_state.initialized = False
                st.cache_resource.clear()
                
            if st.session_state.api_key:
                msg = "✅ Using default API Key (hidden)" if os.getenv("GEMINI_API_KEY") and st.session_state.api_key == os.getenv("GEMINI_API_KEY") else "✅ API Key configured"
                st.markdown(f"<div style='font-size: 0.8rem; color: #34a853; margin-top: -10px; margin-bottom: 10px;'>{msg}</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Enter your Gemini API key.")
                st.markdown("""
                <a href="https://aistudio.google.com/app/apikey" target="_blank"
                   style="color: #fc6d26; font-size: 0.85rem;">
                    🔑 Get free API key →
                </a>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Search Filter**")
            category = st.selectbox(
                "Source",
                ["All", "Handbook", "Direction"],
                help="Filter responses by source type",
                label_visibility="collapsed"
            )
            st.session_state.selected_category = category

            st.markdown("**Display Options**")
            st.toggle(
                "Show Inline Sources",
                key="show_sources",
                help="Show source citations inside chat bubbles"
            )

        # Stats Expander
        with st.expander("📊 Session Stats", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{st.session_state.total_queries}</div>
                    <div class="stat-label">Queries</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                msg_count = len([m for m in st.session_state.messages if m["role"] == "assistant"])
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{msg_count}</div>
                    <div class="stat-label">Responses</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <p style="color: rgba(255,255,255,0.4); font-size: 0.75rem; text-align: center; margin-top: 12px; margin-bottom: 0;">
                Started: {st.session_state.session_start}
            </p>
            """, unsafe_allow_html=True)

        # About Expander
        with st.expander("ℹ️ About GitBot", expanded=False):
            st.markdown("""
            **GitBot** uses Retrieval-Augmented Generation (RAG) to answer questions about GitLab's:

            - 📖 **Handbook** — Culture, values, processes
            - 🗺️ **Direction** — Product roadmap & strategy

            Built with ❤️ for GitLab's "build in public" philosophy.
            """)

        # Footer Links
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <a href="https://handbook.gitlab.com" target="_blank"
               style="color: rgba(255,255,255,0.5); font-size: 0.8rem; text-decoration: none; margin: 0 8px; transition: color 0.2s;"
               onmouseover="this.style.color='#fc6d26'" onmouseout="this.style.color='rgba(255,255,255,0.5)'">
                📖 Handbook
            </a>
            <span style="color: rgba(255,255,255,0.2);">|</span>
            <a href="https://about.gitlab.com/direction/" target="_blank"
               style="color: rgba(255,255,255,0.5); font-size: 0.8rem; text-decoration: none; margin: 0 8px; transition: color 0.2s;"
               onmouseover="this.style.color='#fc6d26'" onmouseout="this.style.color='rgba(255,255,255,0.5)'">
                🗺️ Direction
            </a>
        </div>
        """, unsafe_allow_html=True)


# ─── Main Content ──────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="main-header">
        <div style="font-size: 2.5rem;">🦊</div>
        <div>
            <h1>GitBot</h1>
            <p>Your AI-powered guide to GitLab's Handbook & Direction — Ask anything about GitLab's culture, values, processes, and product strategy.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_welcome():
    """Render welcome screen with suggested questions."""
    st.markdown("""
    <div class="welcome-card">
        <h3 style="color: #fc6d26; margin-bottom: 8px;">👋 Welcome to GitBot!</h3>
        <p style="color: rgba(255,255,255,0.7); margin: 0;">
            I'm your AI assistant for GitLab's Handbook and Direction pages.
            Ask me anything about GitLab's culture, values, engineering practices, product roadmap, and more.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 💡 Try asking:")

    suggested = [
        "What are GitLab's core values?",
        "How does GitLab approach remote work?",
        "What is GitLab's AI/ML product direction?",
        "How does the hiring process work?",
        "What are GitLab's engineering principles?",
        "How does GitLab handle performance reviews?",
        "What is GitLab's approach to transparency?",
        "How are OKRs used at GitLab?",
    ]

    cols = st.columns(2)
    for i, suggestion in enumerate(suggested):
        with cols[i % 2]:
            if st.button(f"💬 {suggestion}", key=f"suggest_{i}", use_container_width=True):
                st.session_state["pending_query"] = suggestion
                st.rerun()


def render_data_status(chunks):
    """Show data loading status."""
    if chunks:
        from src.chunker import get_chunk_stats
        stats = get_chunk_stats(chunks)

        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            (col1, "📄", str(stats.get("total_chunks", 0)), "Knowledge Chunks"),
            (col2, "🌐", str(stats.get("unique_sources", 0)), "Source Pages"),
            (col3, "📖", str(stats.get("by_category", {}).get("handbook", 0)), "Handbook Chunks"),
            (col4, "🗺️", str(stats.get("by_category", {}).get("direction", 0)), "Direction Chunks"),
        ]

        for col, icon, num, label in metrics:
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div class="stat-number">{num}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)


def initialize_chatbot():
    """Initialize the chatbot with a detailed loading UI."""
    if not st.session_state.api_key:
        return False

    if st.session_state.initialized and st.session_state.chatbot:
        return True

    # Wrap entire loading UI in one container so it all clears at once
    loading_container = st.empty()

    with loading_container.container():
        st.markdown("""
        <div style="background: rgba(252,109,38,0.08); border: 1px solid rgba(252,109,38,0.2);
                    border-radius: 16px; padding: 28px; margin: 20px 0; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🦊</div>
            <h3 style="color: #fc6d26; margin: 0 0 8px 0;">Initializing GitBot</h3>
            <p style="color: rgba(255,255,255,0.6); margin: 0; font-size: 0.9rem;">
                Loading AI models and knowledge base — this takes ~15 seconds on first load,
                then stays fast for the rest of your session.
            </p>
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status_text  = st.empty()

    steps = [
        (15,  "📚 Loading GitLab knowledge base (559 chunks)..."),
        (40,  "🧠 Loading embedding model (all-MiniLM-L6-v2)..."),
        (70,  "⚡ Loading reranker + BM25 index in parallel..."),
        (90,  "🔗 Connecting to Gemini API..."),
        (100, "✅ Ready!"),
    ]

    try:
        for pct, msg in steps[:-2]:
            with loading_container.container():
                st.markdown("""
                <div style="background: rgba(252,109,38,0.08); border: 1px solid rgba(252,109,38,0.2);
                            border-radius: 16px; padding: 28px; margin: 20px 0; text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 12px;">🦊</div>
                    <h3 style="color: #fc6d26; margin: 0 0 8px 0;">Initializing GitBot</h3>
                    <p style="color: rgba(255,255,255,0.6); margin: 0; font-size: 0.9rem;">
                        Loading AI models and knowledge base — this takes ~15 seconds on first load,
                        then stays fast for the rest of your session.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                progress_bar = st.progress(pct, text=msg)

        bot, index, chunks, success = load_chatbot_resources(st.session_state.api_key)

        with loading_container.container():
            st.markdown("""
            <div style="background: rgba(252,109,38,0.08); border: 1px solid rgba(252,109,38,0.2);
                        border-radius: 16px; padding: 28px; margin: 20px 0; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">🦊</div>
                <h3 style="color: #fc6d26; margin: 0 0 8px 0;">Initializing GitBot</h3>
            </div>
            """, unsafe_allow_html=True)
            progress_bar = st.progress(90, text="🔗 Connecting to Gemini API...")

        if success:
            st.session_state.chatbot  = bot
            st.session_state.index   = index
            st.session_state.chunks  = chunks
            st.session_state.initialized = True

            with loading_container.container():
                st.markdown("""
                <div style="background: rgba(252,109,38,0.08); border: 1px solid rgba(252,109,38,0.2);
                            border-radius: 16px; padding: 28px; margin: 20px 0; text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 12px;">🦊</div>
                    <h3 style="color: #fc6d26; margin: 0 0 8px 0;">Initializing GitBot</h3>
                </div>
                """, unsafe_allow_html=True)
                st.progress(100, text="✅ Ready!")

            time.sleep(0.5)
            # Clear the entire loading UI in one call
            loading_container.empty()
            return True
        else:
            loading_container.empty()
            st.error("❌ Failed to initialize. Please check your API key.")
            return False

    except Exception as e:
        loading_container.empty()
        st.error(f"❌ Initialization error: {str(e)}")
        return False


# ─── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Check .env for API key changes on every render
    sync_api_key_from_env()
    render_sidebar()
    render_header()

    # Main Chat View
    with st.container():
        # Initialize chatbot
        is_ready = initialize_chatbot()

        if not st.session_state.api_key:
            st.info("👈 Enter your Gemini API key in the sidebar to get started.")
            st.markdown("""
            <div style="background: rgba(252,109,38,0.08); border: 1px solid rgba(252,109,38,0.2);
                        border-radius: 12px; padding: 20px; margin: 20px 0;">
                <h4 style="color: #fc6d26;">🚀 Quick Setup</h4>
                <ol style="color: rgba(255,255,255,0.8); line-height: 2;">
                    <li>Get a free Gemini API key at <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color: #fc6d26;">Google AI Studio</a></li>
                    <li>Paste it in the sidebar under "Configuration"</li>
                    <li>Wait for the knowledge base to load (~1-2 min first time)</li>
                    <li>Start asking questions about GitLab!</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            return

        if not is_ready:
            return

        # Show welcome if no messages
        if not st.session_state.messages:
            render_welcome()

        # Chat history
        if st.session_state.messages:
            st.markdown("---")
            for i, msg in enumerate(st.session_state.messages):
                render_message(msg, i)

        # Handle pending query from suggestion buttons
        pending = st.session_state.pop("pending_query", None)

        # ── Chat input — Enter to send, auto-clears after submit ──────────────
        import random
        placeholders = [
            "Ask anything about GitLab's handbook, values, processes...",
            "e.g. 'How does GitLab approach remote work?'",
            "e.g. 'What is GitLab's mission?'",
            "e.g. 'Tell me about the engineering principles.'",
        ]
        user_input = st.chat_input(
            random.choice(placeholders),
            key="chat_input"
        )

        # Accept input from either the chat box or a suggestion button click
        query = (user_input or "").strip() or (pending or "").strip() or None

        if query:
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.session_state.total_queries += 1

            # Immediately render the user message
            render_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)

            # ── Streaming response ────────────────────────────────────────────
            with st.chat_message("assistant", avatar="🦊"):
                with st.spinner("Thinking..."):
                    # The spinner will automatically disappear once write_stream starts outputting
                    full_response = st.write_stream(
                        st.session_state.chatbot.stream_chat(
                            query,
                            history=st.session_state.messages[:-1],
                            show_sources=st.session_state.get("show_sources", True)
                        )
                    )

                # Retrieve metadata stored by stream_chat
                meta = getattr(st.session_state.chatbot, '_last_stream_meta', {})
                confidence = meta.get("confidence", 0)
                sources = meta.get("sources", [])

                # Render confidence badge directly inline
                if confidence:
                    conf_color = get_confidence_color(confidence)
                    conf_label = get_confidence_label(confidence)
                    st.markdown(
                        f'<span style="font-size:0.75rem; color:{conf_color}; '
                        f'background:rgba(0,0,0,0.2); padding:2px 8px; border-radius:10px; display:inline-block; margin-top:8px;">'
                        f'● {conf_label} confidence</span>',
                        unsafe_allow_html=True
                    )

                # Render sources inline
                if sources and st.session_state.get("show_sources", True):
                    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                        for src in sources:
                            cat = src.get("category", "")
                            badge_class = "badge-handbook" if cat == "handbook" else "badge-direction"
                            badge_label = "Handbook" if cat == "handbook" else "Direction"
                            src_conf = src.get("confidence", 0)
                            src_color = get_confidence_color(src_conf)
                            st.markdown(f"""
                            <div class="source-card">
                                <span class="source-badge {badge_class}">{badge_label}</span>
                                <strong style="color:#e0e0e0; margin-left:8px;">{src.get('title','GitLab')}</strong>
                                <br>
                                <a href="{src.get('url','#')}" target="_blank"
                                   style="color:#fc6d26; font-size:0.8rem; text-decoration:none;">
                                    🔗 {src.get('url','')[:60]}...
                                </a>
                                <div class="confidence-bar">
                                    <div class="confidence-fill"
                                         style="width:{src_conf}%; background:{src_color};">
                                    </div>
                                </div>
                                <small style="color:rgba(255,255,255,0.4);">
                                    Relevance: {src_conf:.0f}%
                                </small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Render timestamp inline
                st.caption(datetime.now().strftime("%H:%M"))

            # Add to message history for re-rendering on next rerun
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response or "",
                "sources": sources,
                "confidence": confidence,
                "timestamp": datetime.now().strftime("%H:%M"),
                "error": meta.get("error")
            })

            st.rerun()




if __name__ == "__main__":
    main()
