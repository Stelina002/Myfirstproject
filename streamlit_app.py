import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Make sure this is correct

st.title("Chat with Your Custom Bot")

# Choose personality
personality = st.selectbox("Choose a personality:", ["friendly", "formal", "sarcastic", "enthusiastic"])

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You:", key="input")

if st.button("Send"):
    if user_input:
        # Prepare request
        payload = {
            "prompt": user_input,
            "personality": personality,
            "history": st.session_state.history
        }

        # Send to FastAPI
        response = requests.post(f"{API_URL}/chat", json=payload)

        if response.status_code == 200:
            data = response.json()
            classification = data.get("classification", "unknown")
            bot_response = data.get("response", "")

            if classification == "allowed":
                st.session_state.history.append({"user": user_input, "bot": bot_response})
                st.markdown(f"**Bot ({personality}):** {bot_response}")
            else:
                st.warning("⚠️ This prompt was blocked by the classifier and no response was generated.")
        else:
            st.error("Error from API: " + response.text)
