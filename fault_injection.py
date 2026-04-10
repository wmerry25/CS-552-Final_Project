import random
import streamlit as st
import torch
time_period = len(st.session_state["params"][:,0])
def random_error():
    fault_functions[random.randint(1,9)]()
def ato_failure():
    pass
def heater_on():
    index = random.randint(time_period-15,time_period-1)
    length = time_period-index
    for i in range(length):
        step = float(random.randint(8,20))/10
        st.session_state["params"][index+i,8] = st.session_state["params"][index+i-1,8] + step
def heater_off():
    index = random.randint(time_period-15,time_period-1)
    length = time_period-index
    for i in range(length):
        step = float(random.randint(8,20))/10
        st.session_state["params"][index+i,8] = st.session_state["params"][index+i-1,8] - step
def dosing_pump():
    index = random.randint(time_period-30,time_period-1)
    length = time_period-index
    for i in range(length):
        step = float(random.randint(2,8))/10
        st.session_state["params"][index+i,1] = st.session_state["params"][index+i-1,1] + step
def filter():
    pass
def flow():
    pass
def death():
    pass
def refugium_light():
    pass
def protein_skimmer():
    pass

fault_functions = [random_error, ato_failure, heater_on, heater_off, 
                  dosing_pump, filter, flow, death,
                  refugium_light, protein_skimmer]

fault_names =["Random", "ATO Failure", "Heater Failure On", "Heater Failure Off", 
                "Dosing Pump Failure", "Filter Issue", "Flow Issue", "Livestock Death",
                "Refugium Light Failure", "Protein Skimmer Failure"]