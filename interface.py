import streamlit as st
import replicate
import os
from reefdatasetgen import VAE, generate_random_walk, ranges, measures
import torch
import pandas as pd
from datetime import datetime, timedelta
import time
import json

def get_data(prompt):
    raw_output = replicate.run(
    "meta/meta-llama-3-8b-instruct",
    input={
        "top_k": 0,
        "top_p": 0.95,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.25,
        "system_prompt": "You are a diagnostic agent for reef aquarium chemistry. Your goal is to determine the specific data needed to analyze a user's concern.\nProtocol:\n1. Identify the core issue.\n2. Determine if the question requires historical data. If the user is asking a general \"How-to\" or a question that doesn't require trend analysis, return: (0, [])\n3. Identify the ideal lookback window in days, between 0 and 100.\n4. Select only the parameters that directly influence the reported symptom.\nCritical Rule for Vague Inputs:\nIf the user's prompt is too vague to require specific data, or if no parameters are relevant, you MUST return exactly: (0, [])\nGuide for times:\nAcute/Emergency (Fish gasping, sudden death, RTN): 1–3 days.\nShort-term (Algae blooms, cloudy water, parameter spikes): 7–14 days.\nLong-term (Coral browning, slow growth, stability checks): 30–100 days.\nOutput Constraint:\nReturn ONLY a tuple (int, list). No text, no markdown blocks, no explanations.\nAvailable Parameters: [Calcium, Alkalinity, Magnesium, Phosphate, Nitrate, Nitrite, Ammonia, Salinity, Temperature, pH, ORP]",
        "length_penalty": 0,
        "max_new_tokens": 512,
        "stop_sequences": "<|end_of_text|>,<|eot_id|>",
        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "presence_penalty": 0,
        "log_performance_metrics": False
    })
    output = json.loads(raw_output)
    lookback = int(output["time"])* -1
    if lookback < -60:
        lookback = -60
    time_window = slice(lookback,None)
    selected_params = output["parameters"]
    index_list = []
    new_data = {}
    if lookback == 0 or len(selected_params) == 0:
        return None
    for param in selected_params:
        try:
            idx = measures.index(param)
            index_list.append(idx)
        except ValueError:
            print("Unknown Parameter Returned by Agent.")
    print(index_list)
    print(lookback)
    for index in index_list:
        new_row =  st.session_state["params"][time_window,index].numpy()
        new_data[measures[index]] = new_row.round(decimals=2)
    return new_data

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

meta = "Size:40g,Age:3y,Livestock:(Clowfish,Blenny,Royal Gramma,Goby)"

#Dashboard
st.title("ReefXpert Monitoring Dashboard")

st.subheader("Water Parameters")
x_start = datetime.now().date() - timedelta(days = len(st.session_state["params"][:, 0]))
x_axis = pd.date_range(x_start, periods=len(st.session_state["params"][:, 0]))

for i in range(len(ranges)):
    st.subheader(measures[i] )
    df = pd.DataFrame(st.session_state["params"][:, i].numpy(), index= x_axis)
    st.line_chart(df)
# Chat
with st.sidebar:
    st.header("ReefXpert Chat")
    v_box = st.container(height = 600)
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
                m_retry = 0
                with st.chat_message("assistant"):
                    placeholder = st.empty() 
                    response = ""
                    history = ''
                    for h in st.session_state.messages[-6:-1]:
                        history += f"{h['role']}: {h['content']}\n\n"
                    relevant_data =get_data(prompt)
                    while m_retry <6:
                        try:
                            stream = replicate.stream(
                                "meta/meta-llama-3-8b-instruct",
                                input={
                                    "temperature": 0.05,
                                    "top_k": 0,
                                    "top_p": 0.9,
                                    "prompt": prompt,
                                    "system_prompt": "You are an expert on coral reef systems in the home.The user is asking questions regarding their reef with these metaparameters:" + meta +". Use the following data to contextualize your answers: " +relevant_data +"Keep your answers short, especially if the user is not asking a complex question. Use the following history to contextulize your answer: " + history,
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
                            break
                        except replicate.exceptions.ReplicateError as e:
                            if "429" in str(e):
                                time.sleep(3)
                            else:
                                raise e
    