import streamlit as st
import requests
from dotenv import load_dotenv
import os

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
    st.subheader("📜 Summarize PDF")
    
    # Fetch stored documents
    response = requests.get(f"{BASE_URL}/publications/")
    if response.status_code == 200:
        documents = response.json().get("publications", [])
        if documents:
            doc_options = [{"label": f"{doc['document_title']} ({doc['file_name']})", "value": doc["file_name"]} for doc in documents]
            selected_doc = st.selectbox("Select a document to summarize:", options=doc_options, format_func=lambda x: x["label"])
        else:
            st.write("No stored documents found.")
            selected_doc = None
    else:
        st.error("Failed to fetch documents.")
        selected_doc = None

    uploaded_file = st.file_uploader("Or upload a PDF file for summary", type=["pdf"])
    summary_topic = st.text_input("Enter a topic or leave blank for a full summary:")

    if st.button("Summarize"):
        if uploaded_file:
            files = {"file": uploaded_file.getvalue()}
            params = {"topic": summary_topic} if summary_topic else {}
            response = requests.post(f"{BASE_URL}/summarize-pdf/", files=files, params=params)
        elif selected_doc:
            params = {"document_name": selected_doc["value"], "topic": summary_topic} if summary_topic else {"document_name": selected_doc["value"]}
            response = requests.post(f"{BASE_URL}/summarize-pdf/", json=params)
        else:
            st.error("Please upload a file or select a stored document.")
            st.stop()

        if response.status_code == 200:
            st.write("### Summary:")
            st.write(response.json().get("summary", "No summary found."))
        else:
            st.error("Error generating summary.")

elif selected_option == "Q&A":
    st.subheader("❓ Q&A")
    
    # Fetch stored documents
    response = requests.get(f"{BASE_URL}/publications/")
    if response.status_code == 200:
        documents = response.json().get("publications", [])
        if documents:
            doc_options = [{"label": "None", "value": None}] + [{"label": f"{doc['document_title']} ({doc['file_name']})", "value": doc['file_name']} for doc in documents]
            selected_doc = st.selectbox("Select a document for Q&A:", options=doc_options, format_func=lambda x: x["label"])
        else:
            st.write("No stored documents found.")
            selected_doc = None
    else:
        st.error("Failed to fetch documents.")
        selected_doc = None

    user_question = st.text_input("Enter your question:")

    if st.button("Ask"):
        if not user_question:
            st.error("Please enter a question.")
        else:
            if selected_doc and selected_doc["value"]:
                params = {"query": user_question, "context_id": selected_doc["value"]}
                response = requests.post(f"{BASE_URL}/qa-pdf/", json=params)
            else:
                params = {"query": user_question}
                response = requests.post(f"{BASE_URL}/generic-qa/", json=params)

            if response.status_code == 200:
                st.write("### Answer:")
                st.write(response.json().get("answer", "No answer found."))
            else:
                st.error(f"Error answering question: {response.status_code}")

elif selected_option == "Manage Notes":
    st.subheader("📝 Manage Notes")
    
    # View Notes
    st.subheader("Your Notes")
    response = requests.get(f"{BASE_URL}/get-notes/")
    if response.status_code == 200:
        notes = response.json().get("notes", [])
        if notes:
            for note in notes:
                st.markdown(f"**{note['title']}**")
                st.write(note["content"])
                if st.button(f"Delete Note: {note['id']}"):
                    del_response = requests.post(f"{BASE_URL}/delete-note/", json={"note_id": note["id"]})
                    if del_response.status_code == 200:
                        st.success("Note deleted successfully.")
                    else:
                        st.error("Failed to delete note.")
        else:
            st.write("No notes found.")

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
