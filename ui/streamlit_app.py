import streamlit as st
import requests
import os

# Streamlit config
st.set_page_config(
    page_title="File Analysis Assistant",
    page_icon="🚀",
    layout="wide",
    menu_items={
        "Get Help": "https://pathway.com/developers",
        "Report a bug": None,
        "About": "Powered by Pathway + Gemini 2.0",
    },
)

# Pathway API config
PATHWAY_HOST = os.getenv("PATHWAY_REST_HOST", "pathway-backend")
PATHWAY_PORT = os.getenv("PATHWAY_REST_PORT", "8000")
PATHWAY_URL = f"http://{PATHWAY_HOST}:{PATHWAY_PORT}"


def call_pathway_api(query, endpoint="/v1/pw_ai_answer"):
    """Call Pathway LLM App API"""
    try:
        response = requests.post(
            f"{PATHWAY_URL}{endpoint}",
            json={"prompt": query},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API connection error: {e}")
        return None


# Main UI
st.title("🚀 File Analysis Assistant")
st.markdown("---")

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about the file."}
    ]

# Chat history display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # API call and response display
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            response = call_pathway_api(prompt)

            if response:
                # Display answer - Pathway 0.21.4 dictionary format
                answer = None

                # Different response formats
                if isinstance(response, dict):
                    # Dictionary format
                    if "result" in response:
                        answer = response["result"]
                    elif "answer" in response:
                        answer = response["answer"]
                    elif "response" in response:
                        answer = response["response"]
                    elif "text" in response:
                        answer = response["text"]
                    else:
                        # Display dictionary
                        answer = str(response)
                elif isinstance(response, str):
                    # String format
                    answer = response
                else:
                    # Other types
                    answer = str(response)

                if answer:
                    st.write(answer)

                    # Add to session
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                    # Display reference information (if sources exist)
                    if isinstance(response, dict) and response.get("sources"):
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(response["sources"], 1):
                                st.json(source, expanded=False)
                else:
                    error_msg = "Empty response."
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
            else:
                error_msg = "Sorry, the system is currently unavailable."
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

# Sidebar information
with st.sidebar:
    st.header("ℹ️ System information")
    st.info(f"**API endpoint**: {PATHWAY_URL}")

    # System health check
    if st.button("🔍 System health check"):
        try:
            health_response = requests.post(
                f"{PATHWAY_URL}/v2/list_documents",
                json={},
                timeout=5,
            )
            if health_response.status_code == 200:
                st.success("✅ System is healthy")
                st.write(health_response.json())
            else:
                st.write(f"{health_response.text}")
                st.warning("⚠️ System response is abnormal")
        except Exception as e:
            st.write(f"Error: {e}")
            st.error("❌ System connection is not available")

    st.markdown("---")
    st.markdown("**📖 Usage**")
    st.markdown(
        """
        Enter your question in the chat field above
        """
    )
