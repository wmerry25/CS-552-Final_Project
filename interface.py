import streamlit as st
import replicate
from reefdatasetgen import VAE, generate_random_walk, ranges, measures
import torch
import pandas as pd
from datetime import datetime, timedelta
import time
import json

def reset_chat_history():
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to the ReefXpert Chat. How may I help?"}]
def get_data(prompt):
    raw_output = replicate.run(
    "meta/meta-llama-3-8b-instruct",
    input={
        "top_k": 0,
        "top_p": 0.95,
        "prompt": prompt,
        "max_tokens": 25,
        "temperature": 0.25,
        "system_prompt": """You are a diagnostic agent for reef aquarium chemistry. Your job is to determine the data needed to analyze a user's query.
                            # PROTOCOL:
                            1. Identify the ideal lookback window in days, between 0 and 100.
                            2. Select only the parameters that directly influence the reported symptom. Available Parameters: [Calcium, Alkalinity, Magnesium, Phosphate, Nitrate, Nitrite, Ammonia, Salinity, Temperature, pH, ORP]
                            3. If unsure on the correct parameters to select, be cautious and select more rather than less.
                            # GUIDE FOR TIMES:
                            Acute/Emergency: 1–3 days.
                            Short-term: 7–14 days.
                            Long-term: 30–100 days.
                            Default to 25 days if unsure.
                            # OUTPUT CONSTRAINT:
                            Return ONLY a json with the 'time' and 'parameters'. No text, no markdown blocks, no explanations.
                            """,
        "length_penalty": 0,
        "max_new_tokens": 25,
        "stop_sequences": "<|end_of_text|>,<|eot_id|>",
        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "presence_penalty": 0,
        "log_performance_metrics": False
    })
    raw_output = "".join(raw_output)
    try:
        output = json.loads(raw_output)
        lookback = int(output["time"])* -1
        if lookback < -60:
            lookback = -60
        time_window = slice(lookback,None)
        selected_params = output["parameters"]
    except json.JSONDecodeError:
        print(f"Param Agent returned {raw_output}")
        return "No Data Requested"
    index_list = []
    new_data = {}
    if lookback == 0 or len(selected_params) == 0:
        return "No Data Requested"
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
    st.session_state["stored_params"] = st.session_state["params"].clone()

if "fault" not in st.session_state:
    st.session_state["fault"] = "No Fault Selected"
if "Button Label" not in st.session_state:
    st.session_state["Button Label"] = "Show Fault"

if "meta" not in st.session_state:
    st.session_state['meta'] = {"volume": 80, 
                                "age": 3, 
                                "gph": 100, 
                                "refugium": 'No', 
                                "skimmer": 'Yes', 
                                "reactor": 'No', 
                                "lighting_type": 'T5',
                                "lighting_period": 9, 
                                "water_change_schedule": 20}
    
from fault_injection import *

def param_hist(i, param, length):
    start = round(st.session_state["params"][100-length, i].item(),2)
    end = round(st.session_state["params"][99, i].item(),2)
    diff = end - start
    pct_change = round(100* abs(diff)/end,2)
    change = "stayed flat"
    if diff > 0: change = "increased" 
    elif diff < 0: change = "decreased"
    return f"{param} {length} Day History: From {start} to {end}. It {change} by {pct_change} % \n"

def dashboard():
    st.title("ReefXpert Monitoring Dashboard")

    st.subheader("Water Parameters")
    x_start = datetime.now().date() - timedelta(days = len(st.session_state["params"][:, 0]))
    x_axis = pd.date_range(x_start, periods=len(st.session_state["params"][:, 0]))

    general_param_trends = ""

    for i in range(len(ranges)):
        param = measures[i]
        st.subheader(param)
        df = pd.DataFrame(st.session_state["params"][:, i].numpy(), index= x_axis)
        st.line_chart(df)
        general_param_trends += param_hist(i, param, 50)
        general_param_trends += param_hist(i, param, 10)
    # Chat
    with st.sidebar:
        col = st.columns(2)
        col[0].header("ReefXpert Chat")
        if col[1].button("Reset Chat History", key= "reset_conversation_history"):
            reset_chat_history()
            st.rerun()
        v_box = st.container(height = 300)
        with v_box:
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
        inc_data = st.toggle("Deep Analysis: Include Detailed Parameter History", value=False, help = "When on, specific tailored data is used to generate a response.")
        if prompt := st.chat_input():
            if 'REPLICATE_API_TOKEN' not in st.secrets:
                st.stop()
            with v_box:  
                sp = "Thinking..."
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
                        if inc_data:
                            with st.spinner(sp):
                                relevant_data = get_data(prompt)
                                sp = "Initial analysis complete. Thinking..."
                        else:
                            relevant_data = "No Data Requested"
                        with st.spinner(sp):
                            while m_retry <6:
                                try:
                                    stream = replicate.stream(
                                        "meta/llama-4-maverick-instruct",
                                        input={
                                            "temperature": 0.6,
                                            "use_cache": True,
                                            "top_k": 0,
                                            "top_p": 0.9,
                                            "prompt": prompt,
                                            "system_prompt": f"""
                                            You are an expert on in-home reef aquarium biology. You are to provide scientific and data-driven advice to users based on the following.
                                            # OVERALL DATA TRENDS
                                            {general_param_trends}
                                            # IDENTIFIED RELEVANT DATA
                                            {relevant_data} 
                                            ### CONVERSATION HISTORY
                                            {history}
                                            # TANK PROFILE (Meta-parameters)
                                            {st.session_state['meta']}
                                            #POTENTIAL ISSUES
                                            ATO Failure, Heater Failure On, Heater Failure Off, Dosing Pump Failure, Filter Issue, Flow Issue, Livestock Death, Refugium Light Failure, Protein Skimmer Failure
                                            # INSTRUCTIONS:
                                            1. If recent data is provided, use the RECENT DATA to identify any immediate threats. If the data shows an anomaly, briefly mention it even if the user didn't ask.
                                            2. Explicily state the data you are looking at. Provide answers based on this data.
                                            3. Do not mention the tank profile unless it is causing an issue. Treat this as context to base your answers on.
                                            4. Ensure your answers are rooted in safety.
                                            """,
                                            "length_penalty": 0.7,
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


def fault_injection():
    st.title("Fault Injection")
    st.write("On this page, select a fault to be included in the dataset or select random to select an unknown fault and test the chatbot.")
    if st.session_state["Button Label"] == "Hide Fault":
        st.write(f'The selected fault is: {st.session_state["fault"]}')
    header_buttons = st.columns(3)
    if header_buttons[0].button("Reset Data", key = "Reset", use_container_width=True):
        reset()
        st.rerun()
    if header_buttons[1].button("Random Fault", key= "Random Fault", use_container_width=True):
        st.session_state["fault"] = random_error()
        st.rerun()
    if header_buttons[2].button(st.session_state["Button Label"], key = "Toggle Show", use_container_width=True):
        if st.session_state["Button Label"] == "Show Fault":
            st.session_state["Button Label"] = "Hide Fault"
            st.rerun()
        else:
            st.session_state["Button Label"] = "Show Fault"
            st.rerun()
    rows = [st.columns(3, vertical_alignment="center"), st.columns(3, vertical_alignment="center"), st.columns(3, vertical_alignment="center")]
    i = 0
    for row in rows:
        for col in row:
            if col.button(fault_names[i], key=fault_names[i], use_container_width=True):
                fault_functions[i]()
                st.session_state["fault"] = fault_names[i]
                st.rerun()
            i+=1


def settings():
    bool_selections = ['Yes', 'No']
    lighting_selections = ['Hallogen', 'T5', 'Hybrid']
    st.title("Settings")
    settings_form = st.form('settings')
    volume = settings_form.number_input('What is the volume of the total system? **(gallons)**', 0, value=st.session_state['meta']["volume"])
    age = settings_form.number_input('What is the age of your aquarium? **(years)**', 0,value=st.session_state['meta']["age"])
    gph = settings_form.number_input('What is the estimated flow/water turnover rate in your aquarium? **(GPH)**', 0,value=st.session_state['meta']["gph"])
    refugium = settings_form.selectbox('Are you using a refugium?', bool_selections, index = bool_selections.index(st.session_state['meta']["refugium"]))
    skimmer = settings_form.selectbox('Are you using a protein skimmer?', bool_selections, index = bool_selections.index(st.session_state['meta']["skimmer"]))
    reactor = settings_form.selectbox('Are you using a carbon reactor?', bool_selections, index = bool_selections.index(st.session_state['meta']["reactor"]))
    lighting_type= settings_form.selectbox('What type of lighting do you use?', lighting_selections, index = lighting_selections.index(st.session_state['meta']["lighting_type"]))
    lighting_period = settings_form.number_input('How many hours are your lights on per day', 0, 24, value=st.session_state['meta']["lighting_period"])
    water_change_schedule = settings_form.number_input('Per week, what percentage water change do you do? If you perform waterchanges on a non-weekly basis, please normalize the number to weekly.', 0, 100,value=st.session_state['meta']["water_change_schedule"])
    submit = settings_form.form_submit_button()
    if submit:
        st.session_state['meta']["volume"] = volume
        st.session_state['meta']["age"] = age
        st.session_state['meta']["gph"] = gph
        st.session_state['meta']["refugium"] = refugium
        st.session_state['meta']["skimmer"] = skimmer
        st.session_state['meta']["reactor"] = reactor
        st.session_state['meta']["lighting_type"] = lighting_type
        st.session_state['meta']["lighting_period"] = lighting_period
        st.session_state['meta']["water_change_schedule"] = water_change_schedule
        st.success("Settings Updated")

if __name__ == "__main__":
    dashboard_page = st.Page(dashboard, title="Dashboard", icon="📈")
    fault_page = st.Page(fault_injection, title="Fault Injection", icon="⚠️")
    settings_page = st.Page(settings,title="Settings", icon = "⚙️")

    pg = st.navigation([dashboard_page, fault_page,settings_page])
    pg.run()