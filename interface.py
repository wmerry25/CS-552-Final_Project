import streamlit as st
import replicate
import os
from reefdatasetgen import VAE, generate_random_walk, ranges, measures
import torch
import pandas as pd

replicate_api = st.secrets['REPLICATE_API_TOKEN']
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to the ReefXpert Chat. How may I help?"}]

if "params" not in st.session_state:
    vae = VAE(input_dim=len(ranges), hidden_dim=16, latent_dim=32)
    state_dict = torch.load('reef_vae.pth', map_location=torch.device('cpu'))
    vae.load_state_dict(state_dict)
    vae.eval()
    with torch.no_grad():
        data = generate_random_walk(vae, 100, 32, 0.1)
    st.session_state["params"] = data

# meta = [{"Size" : "40 Gallons"}, {"Age" : "3 Years"}, {"Livestock": ["Ocellaris Clownfish", "Ocellaris Clownfish", "Midas Blenny", "Royal Gramma", "Watchman Goby"]}]
meta = "Size:40g,Age:3y,Livestock:(Clowfish,Blenny,Royal Gramma,Goby)"

#Dashboard
st.title("ReefXpert Monitoring Dashboard")

st.subheader("Water Parameters")
for i in range(len(ranges)):
    st.subheader(measures[i] )
    st.line_chart(st.session_state["params"][:, i].numpy())

# Chat
with st.sidebar:
    st.header("ReefXpert Chat")
    v_box = st.container(height = 400)
    with v_box:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input():
        if 'REPLICATE_API_TOKEN' not in st.secrets:
            st.stop()
        with v_box:  
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    placeholder = st.empty() 
                    response = ""
                    history = ''
                    for h in st.session_state.messages[-6:-1]:
                        history += f"{h['role']}: {h['content']}\n\n"

                    stream = replicate.stream(
                        "meta/meta-llama-3-8b-instruct",
                        input={
                            "temperature": 0.1,
                            "top_k": 0,
                            "top_p": 0.9,
                            "prompt": prompt,
                            "system_prompt": "You are an expert on coral reef systems in the home.The user is asking questions regarding their reef with these metaparameters:" + meta +". Keep your answers short, especially if the user is not asking a complex question. Use the following history to contextulize your answer: " + history,
                            "length_penalty": 1,
                            "max_new_tokens": 512,
                            "stop_sequences": "<|end_of_text|>,<|eot_id|>",
                            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                            "presence_penalty": 0,
                            "log_performance_metrics": True
                        })
                    for s in stream:
                        response += str(s)
                        placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})