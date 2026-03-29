import streamlit as st
import replicate
import os

replicate_api = st.secrets['REPLICATE_API_TOKEN']
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


st.title("ReefXpert Monitoring Dashboard")

with st.sidebar:
    st.header("ReefXpert Chat")
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
                placeholder = st.empty() 
                response = ""
                history = ''
                for h in st.session_state.messages[:-1]:
                    history += f"{h['role']}: {h['content']}\n\n"

                stream = replicate.stream(
                    "meta/meta-llama-3-8b-instruct",
                    input={
                        "top_k": 0,
                        "top_p": 0.95,
                        "prompt": prompt,
                        "system_prompt": "You are an expert on coral reef systems in the home. Keep your answers short, especially if the user is not asking a complex question. Use the following history to contextulize your answer: " + history,
                        "length_penalty": 1,
                        "max_new_tokens": 512,
                        "stop_sequences": "<|end_of_text|>,<|eot_id|>",
                        # "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                        "presence_penalty": 0,
                        "log_performance_metrics": False
                    })
                for s in stream:
                    response += str(s)
                    placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})