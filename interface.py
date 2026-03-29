import streamlit as st
import replicate
import os

replicate_api = st.secrets['REPLICATE_API_TOKEN']
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def generate_response_with_context(prompt_input):
    history = ""
    base_prompt = "You are an expert on coral reef systems in the home. Below is the history of this dialogue, only respond once as 'Assistant'."
    for hist_message in st.session_state.messages:
        if hist_message["role"] == "user":
            history += "User: " + hist_message["content"] + "\n\n"
        else:
            history += "Assistant: " + hist_message["content"] + "\n\n"
    return replicate.run(
        'meta/llama-2-7b-chat',
        input={"prompt": f"{history} {prompt_input} Assistant: ",
               "max_length": 500, "repetition_penalty": 1}
    )

if prompt := st.chat_input():
    if 'REPLICATE_API_TOKEN' not in st.secrets:
        st.info("Please add your Replicate API key to continue.")
        st.stop()
         
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response_with_context(prompt)
                em = st.empty()
                message = ''
                for i in response:
                    message += em
                    em.markdown(message)
                st.session_state.messages.append({"role": "assistant", "content": message})