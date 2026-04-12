import random
import streamlit as st
import torch
time_period = len(st.session_state["params"][:,0])
from reefdatasetgen import measures
from interface import reset_chat_history

def reset():
    st.session_state["params"] = st.session_state["stored_params"].clone()
    reset_chat_history()
def random_error():
    reset()
    random_fault_index = random.randint(1,9)
    fault_functions[random_fault_index]()
    error = measures[random.randint(1,9)]
    st.write(error)
def generate_steps(parameter, max_steps_from_end, min_steps_from_end, increase, max_step, min_step):
    index = random.randint(time_period-max_steps_from_end-1,time_period-min_steps_from_end-1)
    param_index = measures.index(parameter)
    length = time_period-index
    for i in range(length):
        step = float(random.randint(int(10*min_step), int(10*max_step)))/10
        if increase:
            st.session_state["params"][index+i,param_index] = st.session_state["params"][index+i-1,param_index] + step
        else:
            st.session_state["params"][index+i,param_index] = st.session_state["params"][index+i-1,param_index] - step

def ato_failure():
    reset()
    generate_steps("Salinity", 30,0, True, 0.4,0)
def heater_on():
    reset()
    generate_steps("Temperature", 7,0, True, 2,0)
def heater_off():
    reset()
    generate_steps("Temperature", 7,0, False, 1.5,0)
def dosing_pump():
    reset()
    generate_steps("Alkalinity", 30, 0, True, 0.8,0)
def filter():
    reset()
    generate_steps("ORP", 14, 7, False, 0.2,0)
    generate_steps("Ammonia", 21, 14, True, 0.8,0)
    generate_steps("Nitrate", 14, 7, True, 0.5,0)
    generate_steps("Nitrite", 7, 0, True, 0.2,0)
def flow():
    reset()
    generate_steps("pH", 14, 7, False, 0.2,0)
    generate_steps("Ammonia", 21, 14, True, 0.8,0)
    generate_steps("Nitrate", 14, 7, True, 0.5,0)
    generate_steps("Nitrite", 7, 0, True, 0.2,0)
def death():
    reset()
    generate_steps("pH", 14, 7, False, 0.3,0)
    generate_steps("Ammonia", 21, 14, True, 0.7,0)
    generate_steps("Nitrate", 14, 7, True, 0.5,0)
    generate_steps("Nitrite", 7, 0, True, 0.3,0)
def refugium_light():
    reset()
    generate_steps("Nitrate", 30, True, 0.5,0)
    generate_steps("Phosphate", 30, True, 0.1,0)
def protein_skimmer():
    reset()
    generate_steps("ORP", 14, 0, False, 0.2,0)
    generate_steps("Phosphate", 21, 14, True, 0.1,0)
    generate_steps("Nitrate", 14, 0, True, 0.5,0)

fault_functions = [random_error, ato_failure, heater_on, heater_off, 
                  dosing_pump, filter, flow, death,
                  refugium_light, protein_skimmer]

fault_names =["Random", "ATO Failure", "Heater Failure On", "Heater Failure Off", 
                "Dosing Pump Failure", "Filter Issue", "Flow Issue", "Livestock Death",
                "Refugium Light Failure", "Protein Skimmer Failure"]