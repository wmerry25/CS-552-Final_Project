import streamlit as st
import replicate
import os

replicate_api = st.secrets['REPLICATE_API_TOKEN']
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if 'REPLICATE_API_TOKEN' not in st.secrets:
        st.info("Please add your Replicate API key to continue.")
        st.stop()
         
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = ''
                for h in st.session_state.messages[:-1]:
                    history += f"{h['role']}: {h['content']}\n\n"
                stream = replicate.stream(
                    "meta/llama-2-7b-chat",
                    input={
                        "top_k": 0,
                        "top_p": 1,
                        "prompt": prompt,
                        "max_tokens": 512,
                        "temperature": 0.75,
                        "system_prompt": "You are an expert on coral reef systems in the home. Use the following history to contextulize your answer: " + history,
                        "length_penalty": 1,
                        "max_new_tokens": 800,
                        "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
                        "presence_penalty": 0,
                        "log_performance_metrics": False
                    },
                )
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})