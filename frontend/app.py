import streamlit as st
import requests
from dotenv import load_dotenv
import os
import base64
from datetime import datetime
from fpdf import FPDF
import io
import pandas as pd

# Load environment variables
load_dotenv()

# Backend API URL
BASE_URL = "http://127.0.0.1:8000"

# Streamlit Page Configuration
st.set_page_config(
    page_title="Quantum Computing Tutor",
    page_icon="🧠",
    layout="wide",
)

def create_pdf_from_notes(notes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Quantum Computing Study Notes", ln=True, align="C")
    pdf.ln(10)
    
    for note in notes:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, note["title"], ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, note["content"])
        pdf.ln(10)
    
    return pdf.output(dest="S").encode("latin1")

def get_download_link(pdf_bytes, filename):
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF</a>'

def content_card(title, content):
    st.markdown(f"""
        <div class="content-card">
            <h3>{title}</h3>
            <p>{content}</p>
        </div>
    """, unsafe_allow_html=True)

# Add custom CSS
st.markdown("""
<style>
    /* Global theme colors */
    :root {
        --background-color: #0e1117;
        --text-color: #ffffff;
        --accent-color: #1565c0;
        --message-bg: #1e2329;
        --user-message-bg: #2d3139;
        --assistant-message-bg: #1e2329;
        --border-color: #303540;
    }

    /* Main container styling */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Card styling for all content containers */
    .content-card {
        background-color: var(--message-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Message styling for Q&A */
    .message-container {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .assistant-message {
        background-color: var(--assistant-message-bg);
        border-left: 3px solid var(--accent-color);
    }

    .user-message {
        background-color: var(--user-message-bg);
        border-left: 3px solid #4CAF50;
    }

    /* Input fields styling */
    .stTextInput input, 
    .stTextArea textarea, 
    .stSelectbox select {
        background-color: var(--message-bg);
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-radius: 5px;
    }

    /* Button styling */
    .stButton button {
        background-color: var(--accent-color);
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background-color: #1976d2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* File uploader styling */
    .uploadedFile {
        background-color: var(--message-bg);
        border: 1px dashed var(--border-color);
        border-radius: 5px;
    }

    /* Expander styling */
    .streamlit-expander {
        background-color: var(--message-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Update the message container function
def message_container(role, content):
    icon = "🤖" if role == "assistant" else "👤"
    class_name = "assistant-message" if role == "assistant" else "user-message"
    st.markdown(f"""
        <div class="message-container {class_name}">
            <strong>{icon} {"Personalized Quantum Computing Assistant" if role == "assistant" else "You"}</strong><br>
            {content}
        </div>
    """, unsafe_allow_html=True)

# Update the content card function
def content_card(title, content):
    st.markdown(f"""
        <div class="content-card">
            <h3>{title}</h3>
            <div>{content}</div>
        </div>
    """, unsafe_allow_html=True)


# Custom container for the chat interface
def chat_container():
    st.markdown("""
        <div style="
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm your Quantum Computing Tutor. How can I help you today?"}
    ]
if "current_task" not in st.session_state:
    st.session_state["current_task"] = None

# Sidebar: Saved Chats
with st.sidebar:
    st.title("Menu")
    selected_option = st.radio(
        "Choose an option",
        options=["Home", "Ask the Tutor ?", "Summarize the Learning"]
    )

# Main content area
st.title("🧠 Quantum Computing Tutor")

if selected_option == "Home":
    st.caption("🚀 Learn and interact with Quantum Computing!")
    # Add a GIF related to Quantum Computing using URL
    
    st.markdown(
        """
        **Features:**
        - ❓ **Ask the Tutor ?** : Ask questions based on documents, cahpter or general topics.
        - 📜 **Summarize the Learning**: Summarize the stored or uploaded PDFs.
        - 📝 **Download Summary Notes**: Download and Edit the Summary Notes for self-study.
        """
    )
    st.markdown(
        """
        <div style="text-align:center;">
            <img src="https://media.giphy.com/media/L1R1tvI9svkIWwpVYr/giphy.gif?cid=790b7611vlktz6ez8pjtquu0ud81pzkj4hhr3fwypwqslgz1&ep=v1_gifs_search&rid=giphy.gif&ct=g" width="600">
        </div>
        """,
        unsafe_allow_html=True
    )

# "Ask the Tutor?" option selected
elif selected_option == "Ask the Tutor ?":

    content_card("Interact with the Tutor", """
        Ask questions about Quantum Computing and get detailed answers.
    """)

    chat_container()
    
    # Display chat history
    for message in st.session_state.messages:
        message_container(message["role"], message["content"])
    
    # Document selection and context
    col1, col2 = st.columns([2, 1])
    with col1:
        user_question = st.text_input(
            "Ask your question:", 
            key="user_input", 
            placeholder="Ask anything about quantum computing..."
        )
    
    with col2:
        response = requests.get(f"{BASE_URL}/publications/")
        if response.status_code == 200:
            documents = response.json().get("publications", [])
            if documents:
                doc_options = [{"label": "No specific document", "value": None}] + \
                             [{"label": f"{doc['document_title']}", "value": doc['file_name']} 
                              for doc in documents]
                selected_doc = st.selectbox(
                    "Select document context (optional):",
                    options=doc_options,
                    format_func=lambda x: x["label"]
                )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        ask_button = st.button("🚀 Ask Question", use_container_width=True)
    
    if ask_button and user_question:
        with st.spinner("Researching your question..."):
            params = {
                "query": user_question,
                "context_id": selected_doc["value"] if selected_doc and selected_doc["value"] else None
            }
            
            response = requests.post(f"{BASE_URL}/qa-pdf/", json=params)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "I couldn't find an answer to that question.")
                papers = result.get("papers", [])
                
                st.session_state.messages.append({"role": "user", "content": user_question})
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Display answer
                st.markdown("### Answer")
                st.write(answer)
            else:
                st.error("Failed to fetch the answer. Please try again...")
                    
                    # Add save Q&A button
            if st.button("💾 Next", key="save_qa_note"):
                title = f"Q&A: {user_question[:50]}..."
                content = f"Question: {user_question}\n\nAnswer: {answer}"
                if papers:
                    content += "\n\nRelevant Papers:\n" + "\n".join([f"- {p['title']}" for p in papers])
                note_response = requests.post(
                    f"{BASE_URL}/save-note/",
                    json={"title": title, "content": content}
                )
                if note_response.status_code == 200:
                    st.success("Next question!")
                    st.rerun()

                # Display research papers if available
                if papers:
                    st.markdown("### 📚 Relevant Research Papers")
                    for paper in papers:
                        st.markdown(f"- [{paper['title']}]({paper['link']})")

# "Summarize the Learning" option selected
elif selected_option == "Summarize the Learning":
    #st.subheader("📜 Document Summarizer")
    content_card("📜 Document Summarizer", """
        Upload or Select a document to generate a summary.
    """)
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Fetch stored documents
        response = requests.get(f"{BASE_URL}/publications/")
        if response.status_code == 200:
            documents = response.json().get("publications", [])
            if documents:
                doc_options = [
                    {"label": "Select", "value": None}
                ] + [
                    {"label": f"{doc['document_title']}", "value": doc["file_name"]}
                    for doc in documents
                ]
                selected_doc = st.selectbox(
                    "📚 Select a document:",
                    options=doc_options,
                    format_func=lambda x: x["label"],
                    help="Choose from stored documents or upload a new PDF"
                )
            else:
                st.info("No stored documents found.")
                selected_doc = None
        else:
            st.error("Failed to fetch documents.")
            selected_doc = None

        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "📤 Upload a PDF document",
            type=["pdf"],
            help="Upload a PDF file to generate a Summary"
        )
        st.markdown("**Limit: 5MB per file**")

    with col2:
        st.markdown("### Summary Options")
        summary_type = st.radio(
            "Choose Summary Type:",
            ["Entire Document", "Chapter/Topic Name"],
            help="Select whether to summarize the entire document or a specific part"
        )
        
        if summary_type == "Chapter/Topic Name":
            summary_topic = st.text_input(
                "Enter the Chapter or Topic Name:",
                placeholder="Ex: Quantum Superposition, Chapter 1",
                help="Specify the chapter or topic you want to summarize"
            )
        else:
            summary_topic = None

    # Center the summarize button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        summarize_button = st.button(
            "🔍 Generate Summary ",
            use_container_width=True,
            help="Click to generate the summary"
        )

    if summarize_button:
        with st.spinner("Generating summary..."):
            try:
                if uploaded_file:
                    # Handle uploaded PDF
                    files = {"file": ("uploaded.pdf", uploaded_file.getvalue(), "application/pdf")}
                    params = {"topic": summary_topic} if summary_topic else {}
                    response = requests.post(
                        f"{BASE_URL}/summarize-pdf/",
                        files=files,
                        params=params
                    )
                elif selected_doc:
                    # Handle stored document
                    params = {
                        "document_name": selected_doc["value"],
                        "chapter": summary_topic if summary_type == "Chapter/Topic Name" else None
                    }
                    response = requests.post(
                        f"{BASE_URL}/summarize-pdf/",
                        params=params
                    )
                else:
                    st.error("Please upload a file or select a document")
                    st.stop()

                if response.status_code == 200:
                    # Success - Display the summary
                    summary = response.json().get("summary", "No summary found.")
                    st.success("Summary generated successfully!")
                    st.markdown("### 📝 Summary")
                    st.write(summary)
                    
                    # Add download summary button
                if summary:  # Ensure a summary is generated
                    # Determine the filename
                    file_name = f"Summary - {summary_topic if summary_topic else selected_doc['value'] if selected_doc else uploaded_file.name}.txt"
                
                    # Prepare the summary content for download
                    summary_content = f"Summary of {selected_doc['value'] if selected_doc else uploaded_file.name}\n\n"
                    summary_content += f"Topic: {summary_topic if summary_topic else 'Entire Document'}\n\n"
                    summary_content += summary
                
                    # Display the download button
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary_content,
                        file_name=file_name,
                        mime="text/plain",
                        help="Click to download the summary as a text file."
                    )
                else:
                    st.error("No summary available to download. Please generate a summary first.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


