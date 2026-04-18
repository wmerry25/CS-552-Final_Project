# README - Reef Xpert

## Note
This Repository is the implentation of a final project for CS 552 - Generative Artifical Intelligence. See the final project report [here](Project_Report.pdf).

## Project Abstract
In-home reef-keeping is an ever-growing interest
in the United States and globally. While these aquariums are
amazing to look at, they are expensive investments requiring
significant upkeep and an understanding of water chemistry.
ReefXpert is an LLM-based chat application with a multi-stage
data analysis pipeline to assist new practitioners in maintaining
a healthy aquatic environment. This paper details the creation
and performance evaluation of ReefXpert, including creation and
training of a variational autoencoder to create synthetic data for
the LLMs to analyze. Furthermore, a fault injection simulator
is utilized to analyze the model’s reaction to potential real-world
system failures.

## Repository 
-plots/
-fault_injection.py
-interface.py
-model_eval.py
-reef_vae.pth
-reefdatasetgen.py
-requirements.txt

## Setup + Install

This project was implemented using Python 3.11.9 using the following resources:
- Streamlit
- Replicate
- PyTorch
- Pandas
- Python Datetime and Time
- Python JSON
- Python Random
- Python Torchmetrics

See [requirements.txt](requirements.txt) for required packages.

A Replicate API key is required to run the application locally. One can be obtained [here] (https://replicate.com/). An API key is not needed to run the application on Replicate Cloud.


## Use
To run use the application please visit the cloud version [here](https://reefxpert.streamlit.app/).

To run the application locally (not recommended unless implementing changes), ensure you have populated an API key and generated the reef_vae saved model. To generate this, run the command reefdatasetgen.py in your set up virtual environment terminal. You can then run streamlit run interface.py which will open the application in browser on a local host.