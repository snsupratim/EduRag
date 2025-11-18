import streamlit as st
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth
import os

load_dotenv()
API_URL = os.getenv("API_URL")

st.set_page_config(page_title="Enterprise Knowledge Explorer", layout="wide")


# ---------------- Session Init ----------------
if "username" not in st.session_state:
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.role = ""
    st.session_state.logged_in = False
    st.session_state.summary = ""


def get_auth():
    return HTTPBasicAuth(st.session_state.username, st.session_state.password)


# ---------------- Login / Signup ----------------
def auth_ui():
    st.title("Enterprise PDFs → Searchable Knowledge")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    # Login
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                res = requests.get(f"{API_URL}/login", auth=HTTPBasicAuth(username, password))
                if res.status_code == 200:
                    user = res.json()
                    st.session_state.username = username
                    st.session_state.password = password
                    st.session_state.role = user["role"]
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error(res.json().get("detail", "Login failed"))
            except Exception:
                st.error("Server unreachable. Ensure backend is running.")

    # Signup
    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        new_role = st.selectbox("Role", ["admin", "hr", "finance", "engineer", "sales", "support", "other"])
        if st.button("Signup"):
            payload = {"username": new_user, "password": new_pass, "role": new_role}
            try:
                res = requests.post(f"{API_URL}/signup", json=payload)
                if res.status_code == 200:
                    st.success("Signup successful — now login.")
                else:
                    st.error(res.json().get("detail", "Signup failed"))
            except Exception:
                st.error("Server unreachable. Ensure backend is running.")


# ---------------- MAIN DASHBOARD ----------------
def dashboard():
    st.sidebar.title(f"Welcome {st.session_state.username}")
    st.sidebar.caption(f"Role: {st.session_state.role}")

    # Logout
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.password = ""
        st.session_state.role = ""
        st.session_state.summary = ""
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("📄 Upload Enterprise PDF")

    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    role_for_doc = st.sidebar.selectbox(
        "Give access to:", ["admin", "hr", "finance", "engineer", "sales", "support", "other"]
    )

    if st.sidebar.button("Upload"):
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {"role": role_for_doc}
            try:
                res = requests.post(f"{API_URL}/upload_docs", files=files, data=data, auth=get_auth())
                if res.status_code == 200:
                    st.success("File uploaded successfully.")
                else:
                    st.error(res.json().get("detail", "Upload failed"))
            except Exception:
                st.error("Server unreachable. Ensure backend is running.")
        else:
            st.warning("Please upload a PDF file first.")

    st.sidebar.divider()
    st.sidebar.subheader("📌 My Uploaded PDFs")

    # ---- SAFE PDF FETCH ----
    pdfs = []
    try:
        res_list = requests.get(f"{API_URL}/uploaded_docs", auth=get_auth(), timeout=5)
        data = res_list.json() if res_list.status_code == 200 else []
        if isinstance(data, list):
            pdfs = data
    except Exception:
        pdfs = []

    # Dropdown options (always safe)
    if pdfs:
        options = [f"{d.get('doc_id','?')} — {d.get('name','Untitled')}" for d in pdfs]
    else:
        options = ["No documents uploaded yet"]

    selected_pdf = st.sidebar.selectbox("Select document", options)

    # -------- MAIN SCREEN --------
    st.title("Enterprise Knowledge Assistant")

    # If no docs exist, stop here and show helpful message
    if not pdfs:
        st.info("🚀 Upload a PDF from the sidebar to begin exploring knowledge.")
        return

    # -------- SUMMARY BUTTON --------
    if st.button("📌 Extract Important Information"):
        try:
            res = requests.post(
                f"{API_URL}/chat",
                data={"message": "Summarize the key information from this document."},
                auth=get_auth(),
            )
            if res.status_code == 200:
                reply = res.json()
                st.session_state.summary = reply.get("answer", "")
            else:
                st.error("Failed to extract summary.")
        except Exception:
            st.error("Server unreachable. Ensure backend is running.")

    if st.session_state.summary:
        st.markdown("### 📍 Key Information Extracted")
        st.info(st.session_state.summary)

    # -------- CHAT BAR --------
    st.markdown("### 💬 Ask Questions About This Document")
    query = st.text_input("Ask a question...")

    if st.button("Send"):
        if not query.strip():
            st.warning("Enter a question.")
        else:
            try:
                res = requests.post(
                    f"{API_URL}/chat",
                    data={"message": query},
                    auth=get_auth(),
                )
                if res.status_code == 200:
                    reply = res.json()
                    st.success(reply.get("answer", "No answer generated."))
                    if reply.get("sources"):
                        st.caption("Sources: " + ", ".join(reply["sources"]))
                else:
                    st.error("Error fetching reply.")
            except Exception:
                st.error("Server unreachable. Ensure backend is running.")


# ---------------- APP CONTROLLER ----------------
if not st.session_state.logged_in:
    auth_ui()
else:
    dashboard()
