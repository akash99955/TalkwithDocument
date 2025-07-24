import streamlit as st
import google.generativeai as genai

from utils.document_loader import load_pdf, load_docx, load_txt, chunk_text
from utils.text_embedder import embed_chunks, retrive_relevent_chunks
import time

# Page config
st.set_page_config(page_title="Talk with Your Document 🤖", layout="wide")
st.title("📄 Talk with Your Document")

# Sidebar for Gemini API Key
st.sidebar.header("🔐 Gemini API Setup")
user_api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

if not user_api_key:
    st.warning("Please enter your Gemini API key in the sidebar to start.")
    st.stop()

try:
    genai.configure(api_key=user_api_key)
    chat_model = genai.GenerativeModel("gemini-2.0-flash-lite")
    st.sidebar.success("✅ Gemini API configured.")
except Exception as e:
    st.sidebar.error(f"❌ Failed to configure Gemini API: {e}")
    st.stop()
# Upload document
uploaded_file = st.file_uploader("📎 Upload your document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Check if new file is uploaded
if uploaded_file:
    # If new document is uploaded, clear everything
    if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
        # New document — reset everything
        st.session_state.clear()
        st.session_state["last_uploaded_file"] = uploaded_file.name

        # Reconfigure Gemini (needed after session_state.clear())
        genai.configure(api_key=user_api_key)
        chat_model = genai.GenerativeModel("gemini-2.0-flash-lite")

        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext == "pdf":
            raw_text = load_pdf(uploaded_file)
        elif file_ext == "docx":
            raw_text = load_docx(uploaded_file)
        elif file_ext == "txt":
            raw_text = load_txt(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        st.info("■ Chunking and embedding document...")
        chunks = chunk_text(raw_text)
        embeddings = embed_chunks(chunks)

        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embeddings
        st.session_state["document_processed"] = True

        st.success(f"💡 Document processed and embedded into {len(chunks)} chunks.")
    else:
        st.session_state["document_processed"] = True
else:
    st.session_state["document_processed"] = False

# Ask a question only after document is processed
if st.session_state.get("document_processed", False):
    st.subheader("@ Ask a question about your document")

    # Ensure fresh key to clear old query when document changes
    query_key = f"query_input_{st.session_state['last_uploaded_file']}"
    query = st.text_input("Enter your question:", key=query_key)

    if query:
        st.info("@ Retrieving relevant chunks...")
        top_chunks = retrive_relevent_chunks(
            query,
            st.session_state["chunks"],
            st.session_state["embeddings"],
            top_k=5
        )
        context = "\n\n".join(top_chunks)
        prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"""

        st.info("Generating answer with Gemini...")
        st.markdown("### Answer")
        response_area = st.empty()
        try:
            response_stream = chat_model.generate_content(prompt, stream=True)
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    response_area.markdown(full_response)
                    time.sleep(0.05)
        except Exception as e:
            st.error(f"Error generating response: {e}")

else:
    st.info("Please upload and process a document to start asking a question.")





