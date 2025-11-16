import streamlit as st
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth
import os
from pathlib import Path
import pandas as pd

# --- Configuration ---
load_dotenv()
API_URL = os.getenv("API_URL")

st.set_page_config(page_title="PDF's INTO SEARCHABLE KNOWLEDGE", layout="centered")

# Set the base directory for local image access
IMAGE_DIR = "./uploaded_docs/page_images"  # Must match backend vectorstore path

# --- Session Initialization ---
if "username" not in st.session_state:
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.role = ""
    st.session_state.logged_in = False
    st.session_state.mode = "auth"

# --- Auth Helper ---
def get_auth():
    return HTTPBasicAuth(st.session_state.username, st.session_state.password)


# ================= AUTH UI =================
def auth_ui():
    st.title("PDF's INTO SEARCHABLE KNOWLEDGE")
    st.subheader("Login or Signup")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    # Login
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            res = requests.get(f"{API_URL}/login", auth=HTTPBasicAuth(username, password))
            if res.status_code == 200:
                user_data = res.json()
                st.session_state.username = username
                st.session_state.password = password
                st.session_state.role = user_data["role"]
                st.session_state.logged_in = True
                st.session_state.mode = "chat"
                st.success(f"Welcome {username}")
                st.rerun()
            else:
                st.error(res.json().get("detail", "Login failed"))

    # Signup
    with tab2:
        new_user = st.text_input("New Username", key="signup_user")
        new_pass = st.text_input("New Password", type="password", key="signup_pass")
        new_role = st.selectbox("Choose Role", ["admin", "doctor", "nurse", "patient", "other"])
        if st.button("Signup"):
            payload = {"username": new_user, "password": new_pass, "role": new_role}
            res = requests.post(f"{API_URL}/signup", json=payload)
            if res.status_code == 200:
                st.success("Signup successful! You can login now.")
            else:
                st.error(res.json().get("detail", "Signup failed"))


# ================= UPLOAD DOCS =================
def upload_docs():
    st.subheader("Upload PDF for specific Role")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    role_for_doc = st.selectbox("Target Role for Document", ["admin", "doctor", "nurse", "patient", "other"])

    if st.button("Upload Document"):
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {"role": role_for_doc}
            res = requests.post(f"{API_URL}/upload_docs", files=files, data=data, auth=get_auth())
            if res.status_code == 200:
                doc_info = res.json()
                st.success(f"Uploaded: {uploaded_file.name}")
                st.info(f"Doc Id : {doc_info['doc_id']}, Access: {doc_info['accessible_to']}")
            else:
                st.error(res.json().get("detail", "Upload failed"))
        else:
            st.warning("Please upload a file first.")


# ================= CHAT INTERFACE =================
def chat_interface():
    st.subheader("Search related to PDF's ")
    msg = st.text_input("Your search query")

    if st.button("Send"):
        if not msg.strip():
            st.warning("Please enter a query")
            return

        res = requests.post(f"{API_URL}/chat", data={"message": msg}, auth=get_auth())

        if res.status_code == 200:
            reply = res.json()

            # ---- Display Answer ----
            st.markdown("### 🔍 Assistant Answer")
            st.success(reply.get("answer", "No answer generated."))

            # ---- Display Tables ----
            if reply.get("tables"):
                st.markdown("### 📊 Relevant Tables Found")
                for i, table_text in enumerate(reply["tables"], start=1):
                    st.markdown(f"**Table {i}:**")
                    try:
                        # Try to parse into a DataFrame (works if rows separated by \n and columns by '|')
                        rows = [row.strip().split("|") for row in table_text.strip().split("\n") if row.strip()]
                        if rows and len(rows[0]) > 1:
                            df = pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]])
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.text(table_text)
                    except Exception:
                        st.text(table_text)

            # ---- Display Images ----
            if reply.get("images"):
                st.markdown("### 🖼️ Relevant Images Found")
                for img_info in reply["images"]:
                    # Extract text + path
                    parts = img_info.split("🖼️ Image Path:")
                    text_part = parts[0].strip()
                    image_path = parts[1].strip() if len(parts) > 1 else None

                    if text_part:
                        st.markdown(f"**Detected Text:** {text_part}")

                    if image_path:
                        local_img_path = Path(image_path)
                        if local_img_path.exists():
                            st.image(str(local_img_path), caption=f"{local_img_path.name}")
                        else:
                            st.warning(f"⚠️ Image not found locally: {image_path}")

            # ---- Display Document Sources ----
            if reply.get("sources"):
                st.markdown("### 📚 Document Sources")
                for src in reply["sources"]:
                    st.write(f"- {src}")
        else:
            st.error(res.json().get("detail", f"Chat failed with status code {res.status_code}."))


# ================= MAIN FLOW =================
if not st.session_state.logged_in:
    auth_ui()
else:
    st.title(f"Welcome, {st.session_state.username}")
    st.markdown(f"**Role:** `{st.session_state.role}`")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.password = ""
        st.session_state.role = ""
        st.session_state.mode = "auth"
        st.rerun()

    if st.session_state.role == "admin":
        upload_docs()
        st.divider()
        chat_interface()
    else:
        chat_interface()
