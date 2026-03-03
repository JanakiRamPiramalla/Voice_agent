from typing import List, Dict, Optional
import os
from firecrawl import FirecrawlApp
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from fastembed import TextEmbedding
from groq import AsyncGroq          # ✅ replaces OpenAI Agents SDK + AsyncOpenAI
from gtts import gTTS               # ✅ replaces gpt-4o-mini-tts (free, no rate limits)
import tempfile
import uuid
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import asyncio

load_dotenv()

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "initialized":      False,
        "qdrant_url":       "",
        "qdrant_api_key":   "",
        "firecrawl_api_key":"",
        "groq_api_key":     "",        # ✅ Groq key (free at console.groq.com)
        "doc_url":          "",
        "setup_complete":   False,
        "client":           None,
        "embedding_model":  None,
        "groq_client":      None,
        "groq_model":       "llama-3.3-70b-versatile",
        "tts_lang":         "en",      # gTTS language code
        "tts_slow":         False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def sidebar_config():
    with st.sidebar:
        st.title("🔑 Configuration")
        st.markdown("---")

        # ── API keys ──────────────────────────
        st.session_state.qdrant_url       = st.text_input("Qdrant URL",        type="password")
        st.session_state.qdrant_api_key   = st.text_input("Qdrant API Key",    type="password")
        st.session_state.firecrawl_api_key= st.text_input("Firecrawl API Key", type="password")

        st.info("🆓 Get a **free** Groq key at [console.groq.com](https://console.groq.com)")
        st.session_state.groq_api_key     = st.text_input("Groq API Key",      type="password")

        st.markdown("---")

        # ── Documentation URL ─────────────────
        st.session_state.doc_url = st.text_input(
            "Documentation URL",
            placeholder="https://platform.openai.com/docs"
        )

        st.markdown("---")

        # ── Groq model selector ───────────────
        groq_models = [
            "llama-3.3-70b-versatile",   # best quality, generous free tier
            "llama-3.1-8b-instant",      # fastest, ultra-low latency
            "mixtral-8x7b-32768",        # long context window
            "gemma2-9b-it",              # Google Gemma 2
        ]
        st.session_state.groq_model = st.selectbox(
            "🧠 Groq Model",
            groq_models,
            help="llama-3.3-70b gives best answers; llama-3.1-8b is fastest."
        )

        # ── gTTS voice settings ───────────────
        st.markdown("**🔊 TTS Settings** *(powered by gTTS – free)*")
        tts_lang_map = {
            "English 🇬🇧":  "en",
            "Spanish 🇪🇸":  "es",
            "French 🇫🇷":   "fr",
            "German 🇩🇪":   "de",
            "Hindi 🇮🇳":    "hi",
            "Japanese 🇯🇵": "ja",
            "Chinese 🇨🇳":  "zh-CN",
            "Arabic 🇸🇦":   "ar",
        }
        lang_label = st.selectbox("TTS Language", list(tts_lang_map.keys()))
        st.session_state.tts_lang = tts_lang_map[lang_label]
        st.session_state.tts_slow = st.checkbox("Slow speech", value=False)

        st.markdown("---")

        # ── Initialize button ─────────────────
        if st.button("🚀 Initialize System", type="primary"):
            try:
                with st.spinner("Setting up Qdrant..."):
                    client, embedding_model = setup_qdrant_collection(
                        st.session_state.qdrant_url,
                        st.session_state.qdrant_api_key
                    )
                    st.session_state.client          = client
                    st.session_state.embedding_model = embedding_model

                with st.spinner("Crawling documentation..."):
                    pages = crawl_documentation(
                        st.session_state.firecrawl_api_key,
                        st.session_state.doc_url
                    )
                    store_embeddings(client, embedding_model, pages, "docs_embeddings")

                # ── Groq client (replaces OpenAI + Agents) ──
                st.session_state.groq_client = AsyncGroq(
                    api_key=st.session_state.groq_api_key
                )

                st.session_state.setup_complete = True
                st.success("✅ System initialized! No OpenAI, no rate limits 🎉")

            except Exception as e:
                st.error(f"Error during setup: {e}")


# ─────────────────────────────────────────────
# QDRANT
# ─────────────────────────────────────────────
def setup_qdrant_collection(
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str = "docs_embeddings"
):
    client          = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embedding_model = TextEmbedding()                      # fastembed, local, free
    dim             = len(list(embedding_model.embed(["test"]))[0])

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
    except Exception:
        pass  # collection already exists

    return client, embedding_model


# ─────────────────────────────────────────────
# FIRECRAWL
# ─────────────────────────────────────────────
def crawl_documentation(firecrawl_api_key: str, url: str) -> List[Dict]:
    firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
    pages     = []

    response = firecrawl.crawl_url(
        url,
        params={
            "limit": 5,
            "scrapeOptions": {"formats": ["markdown", "html"]}
        }
    )

    for page in response:
        content    = page.get("markdown") or page.get("html", "")
        metadata   = page.get("metadata", {})
        source_url = metadata.get("sourceURL", "")

        pages.append({
            "content":  content,
            "url":      source_url,
            "metadata": {
                "title":       metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "language":    metadata.get("language", "en"),
                "crawl_date":  datetime.now().isoformat()
            }
        })

    return pages


# ─────────────────────────────────────────────
# STORE EMBEDDINGS
# ─────────────────────────────────────────────
def store_embeddings(client, embedding_model, pages: List[Dict], collection_name: str):
    for page in pages:
        embedding = list(embedding_model.embed([page["content"]]))[0]

        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": page["content"],
                        "url":     page["url"],
                        **page["metadata"]
                    }
                )
            ]
        )


# ─────────────────────────────────────────────
# QUERY  (Groq LLM  +  gTTS)
# ─────────────────────────────────────────────
async def process_query(
    query:           str,
    client:          QdrantClient,
    embedding_model: TextEmbedding,
    groq_client:     AsyncGroq,
    groq_model:      str,
    tts_lang:        str,
    tts_slow:        bool
):
    # 1️⃣  Embed the query and retrieve relevant context
    query_embedding = list(embedding_model.embed([query]))[0]
    results = client.query_points(
        collection_name="docs_embeddings",
        query=query_embedding.tolist(),
        limit=3,
        with_payload=True
    ).points

    context = "\n\n---\n\n".join(r.payload["content"] for r in results)

    # 2️⃣  Ask Groq (replaces OpenAI gpt-4o Agent)
    chat_response = await groq_client.chat.completions.create(
        model=groq_model,
        messages=[
            {
                "role":    "system",
                "content": (
                    "You are a helpful customer support agent. "
                    "Answer the user's question clearly and concisely "
                    "using only the provided documentation context."
                )
            },
            {
                "role":    "user",
                "content": f"Documentation context:\n{context}\n\nUser question: {query}"
            }
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    answer = chat_response.choices[0].message.content

    # 3️⃣  Convert answer to speech with gTTS (replaces gpt-4o-mini-tts, 100% free)
    tts       = gTTS(text=answer, lang=tts_lang, slow=tts_slow)
    audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
    tts.save(audio_path)

    return answer, audio_path


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def run_streamlit():
    st.set_page_config(
        page_title="🎙️ Customer Support Voice Agent",
        page_icon="🎙️",
        layout="wide"
    )
    init_session_state()
    sidebar_config()

    st.title("🎙️ Customer Support Voice Agent")
    st.caption("Powered by **Groq** (LLM) + **gTTS** (TTS) + **Qdrant** (vector DB) — no OpenAI needed")

    if not st.session_state.setup_complete:
        st.info("👈 Fill in the configuration panel and click **Initialize System** to begin.")
        return

    query = st.text_input(
        "Ask a question:",
        placeholder="How do I authenticate with the API?",
        disabled=not st.session_state.setup_complete
    )

    if query:
        with st.spinner("🔍 Searching docs and generating answer..."):
            text, audio = asyncio.run(process_query(
                query,
                st.session_state.client,
                st.session_state.embedding_model,
                st.session_state.groq_client,
                st.session_state.groq_model,
                st.session_state.tts_lang,
                st.session_state.tts_slow,
            ))

        st.markdown("### 📝 Answer")
        st.write(text)

        st.markdown("### 🔊 Audio Response")
        st.audio(audio, format="audio/mp3")


if __name__ == "__main__":
    run_streamlit()