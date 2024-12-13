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
            <strong>{icon} {"Assistant" if role == "assistant" else "You"}</strong><br>
            {content}
        </div>
    """, unsafe_allow_html=True)

def save_note_to_backend(title, content):
    try:
        response = requests.post(
            f"{BASE_URL}/save-note/",
            json={"title": title, "content": content}
        )
        if response.status_code == 200:
            st.success("Note saved successfully!")
            st.rerun()
        else:
            st.error("Failed to save note.")
    except Exception as e:
        st.error(f"Error saving note: {str(e)}")
def save_note_to_backend(title, content):
    try:
        response = requests.post(
            f"{BASE_URL}/save-note/",
            json={"title": title, "content": content}
        )
        if response.status_code == 200:
            st.success("Summary saved as note!")
            st.rerun()
        else:
            st.error("Failed to save summary as note.")
    except Exception as e:
        st.error(f"Error saving note: {str(e)}")

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
    st.title("Navigation")
    selected_option = st.radio(
        "Choose an option",
        options=["Home", "Summarize PDF", "Q&A", "Manage Notes"],
    )

# Main content area
st.title("🧠 Quantum Computing Tutor")

if selected_option == "Home":
    st.caption("🚀 Learn and interact with Quantum Computing!")
    st.markdown(
        """
        **Features:**
        - 📜 **Summarize PDF**: Summarize stored or uploaded PDFs.
        - ❓ **Q&A Tutor**: Ask questions based on documents or general topics.
        - 📝 **Manage Notes**: Save, view, and manage your research notes.
        """
    )



elif selected_option == "Summarize PDF":
    st.subheader("📜 Document Summarizer")
    content_card("Document Summarizer", """
        Upload or select a PDF to generate summaries.
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
                    {"label": "No specific document (Upload PDF)", "value": None}
                ] + [
                    {"label": f"{doc['document_title']} ({doc['file_name']})", "value": doc["file_name"]} 
                    for doc in documents
                ]
                selected_doc = st.selectbox(
                    "📚 Select a document:",
                    options=doc_options,
                    format_func=lambda x: x["label"],
                    help="Choose from previously stored documents or upload a new PDF"
                )
            else:
                st.info("No stored documents found.")
                selected_doc = None
        else:
            st.error("Failed to fetch documents.")
            selected_doc = None

        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "📤 Or upload a new PDF file",
            type=["pdf"],
            help="Upload a PDF file to generate a summary"
        )

    with col2:
        st.markdown("### Summary Options")
        summary_type = st.radio(
            "Choose summary type:",
            ["Full Document", "Specific Chapter/Topic"],
            help="Select whether to summarize the entire document or a specific part"
        )
        
        if summary_type == "Specific Chapter/Topic":
            summary_topic = st.text_input(
                "Enter chapter or topic:",
                placeholder="e.g., Quantum Superposition, Chapter 1",
                help="Specify the chapter or topic you want to summarize"
            )
        else:
            summary_topic = None

    # Center the summarize button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        summarize_button = st.button(
            "🔍 Generate Summary",
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
                        "chapter": summary_topic if summary_type == "Specific Chapter/Topic" else None
                    }
                    response = requests.post(
                        f"{BASE_URL}/summarize-pdf/",
                        params=params
                    )
                else:
                    st.error("Please upload a file or select a stored document.")
                    st.stop()

                if response.status_code == 200:
                    # Success - Display the summary
                    summary = response.json().get("summary", "No summary found.")
                    st.success("Summary generated successfully!")
                    st.markdown("### 📝 Summary")
                    st.write(summary)
                    
                    # Add save summary as note button
                    if st.button("💾 Save Summary as Note", key="save_summary_note"):
                        title = f"Summary: {selected_doc['value'] if selected_doc else uploaded_file.name}"
                        if summary_topic:
                            title += f" - {summary_topic}"
                        content = f"Summary of {selected_doc['value'] if selected_doc else uploaded_file.name}\n\n"
                        content += f"Topic: {summary_topic if summary_topic else 'Full Document'}\n\n"
                        content += summary
                        save_note_to_backend(title, content)
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


elif selected_option == "Q&A":

    st.subheader("💬 Interactive Q&A")
    content_card("Interactive Q&A", """
        Ask questions about quantum computing and get detailed answers.
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
                doc_options = [{"label": "No specific document (General Q&A)", "value": None}] + \
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
                if answer:
                    st.markdown("### Answer")
                    st.write(answer)
                    
                    # Add save Q&A button
                    if st.button("💾 Save Q&A as Note", key="save_qa_note"):
                        title = f"Q&A: {user_question[:50]}..."
                        content = f"Question: {user_question}\n\nAnswer: {answer}"
                        if papers:
                            content += "\n\nRelevant Papers:\n" + "\n".join([f"- {p['title']}" for p in papers])
                        note_response = requests.post(
                            f"{BASE_URL}/save-note/",
                            json={"title": title, "content": content}
                        )
                        if note_response.status_code == 200:
                            st.success("Q&A saved as note!")
                            st.rerun()

                # Display research papers if available
                if papers:
                    st.markdown("### 📚 Relevant Research Papers")
                    for paper in papers:
                        st.markdown(f"- [{paper['title']}]({paper['link']})")
                

elif selected_option == "Manage Notes":
    st.subheader("📝 Manage Notes")
    
    tab1, tab2, tab3 = st.tabs(["View Notes", "Create Note", "Search Notes"])
    
    with tab1:
        st.subheader("Your Notes")
        try:
            response = requests.get(f"{BASE_URL}/get-notes/")
            if response.status_code == 200:
                notes = response.json().get("notes", [])
                if notes:
                    for i, note in enumerate(notes):
                        with st.expander(f"📔 {note['title']}", expanded=False):
                            st.write(note["content"])
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                if st.button(f"🗑️ Delete Note", key=f"delete_note_{i}"):
                                    del_response = requests.post(
                                        f"{BASE_URL}/delete-note/",
                                        json={"note_id": note["id"]}
                                    )
                                    if del_response.status_code == 200:
                                        st.success("Note deleted successfully.")
                                        st.rerun()
                else:
                    st.info("No notes found.")
        except Exception as e:
            st.error(f"Error loading notes: {str(e)}")

    with tab2:
        st.subheader("Create New Note")
        note_title = st.text_input("Title", key="create_note_title")
        note_content = st.text_area("Content", height=300, key="create_note_content")
        if st.button("Save Note", key="save_new_note"):
            if not note_title or not note_content:
                st.error("Please provide both title and content for the note.")
            else:
                response = requests.post(
                    f"{BASE_URL}/save-note/",
                    json={"title": note_title, "content": note_content}
                )
                if response.status_code == 200:
                    st.success("Note saved successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Error saving note.")
    
    with tab3:
        st.subheader("Search Notes")
        search_query = st.text_input("Search in your notes:", 
                                   placeholder="Enter search terms...",
                                   key="search_notes_input")
        if search_query:
            response = requests.get(f"{BASE_URL}/get-notes/")
            if response.status_code == 200:
                notes = response.json().get("notes", [])
                matching_notes = [
                    note for note in notes
                    if search_query.lower() in note["title"].lower() or 
                       search_query.lower() in note["content"].lower()
                ]
                
                if matching_notes:
                    st.write(f"Found {len(matching_notes)} matching notes:")
                    for i, note in enumerate(matching_notes):
                        with st.expander(f"📔 {note['title']}", expanded=True):
                            st.write(note["content"])
                            if st.button(f"🗑️ Delete Note", key=f"delete_search_{i}"):
                                del_response = requests.post(
                                    f"{BASE_URL}/delete-note/",
                                    json={"note_id": note["id"]}
                                )
                                if del_response.status_code == 200:
                                    st.success("Note deleted successfully.")
                                    st.experimental_rerun()
                else:
                    st.info("No matching notes found.")

    # Save New Note
    st.subheader("Save a New Note")
    note_title = st.text_input("Title")
    note_content = st.text_area("Content")
    if st.button("Save Note"):
        if not note_title or not note_content:
            st.error("Please provide both title and content for the note.")
        else:
            response = requests.post(f"{BASE_URL}/save-note/", json={"title": note_title, "content": note_content})
            if response.status_code == 200:
                st.success("Note saved successfully.")
            else:
                st.error("Error saving note.")
