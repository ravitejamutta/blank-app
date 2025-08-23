import streamlit as st
from llm_service import LLMService

# Initialize service
llm_service = LLMService()

def main():
    st.set_page_config(page_title="Chat: RAG vs Fine-tuned", layout="wide")
    st.title("💬 Chat with RAG or Fine-tuned Model")

    # Sidebar: model selection
    with st.sidebar:
        st.header("⚙️ Options")
        model_choice = st.radio("Choose Model:", ["RAG", "Fine-tuned Model"])

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Your message:", key="user_input")

    if user_input:
        # Add user message
        st.session_state.messages.append(("user", user_input))

        # Get model response
        result = llm_service.get_response(user_input, model_choice)
        st.session_state.messages.append(("bot", result["response"]))

        # Sidebar: metadata (only after response is available)
        with st.sidebar:
            st.subheader("📊 Response Metadata")
            st.markdown(f"**Confidence Score:** {result['confidence'] * 100:.1f}%")
            st.markdown(f"**Method:** {result['method']}")
            st.markdown(f"**Time Taken:** {result['response_time']} sec")

            if "retrieved_docs" in result:
                st.subheader("📄 Top Documents")
                for i, doc in enumerate(result["retrieved_docs"][:3]):
                    text = doc["doc"]["text"]
                    meta = doc["doc"].get("metadata", {})
                    st.markdown(f"**Doc {i+1}**: `{meta.get('filename', '')}`\n\n{text[:150]}...")

    # Show chat messages
    for role, msg in st.session_state.messages:
        if role == "user":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Bot:** {msg}")


if __name__ == "__main__":
    main()
