# PHYS 449 Final Project - Systems biology informed deep learning for inferring parameters and hidden dynamics
Group 7 : Richard (Zhifei) Dong, Callum Follett, Nolan Anthony Paul Johnston, Steph Swanson, Jack Zhao

Origninal paper from Yazdani A, Lu L, Raissi M, Karniadakis GE. Systems biology informed deep learning for inferring parameters and hidden dynamics. PLoS computational biology. 2020;16(11):e1007575.

## Introduction
This repo contains the code and data to recreate the work by Yazdani et al., 2020. In their work, Yazdani et al. utilized deep learning to determie the parameters for systems biology models.

## File structure
Three models are tested and validated with the neural network architecture, each with a ``main.py`` that is used to train the model and produce the final report / figure, as well as two source files ``nn_gen.py`` and ``data_gen.py`` that contains the neutral network architecture and the code for generating the data for training. 

The file structre is outlined below:

- ~/apoptosis
  - main.py
  - /src
    - data_gen.py
    - nn_gen.py
- ~/glycolysis
  - main.py
  - /src
    - data_gen.py
    - nn_gen.py
- ~/insulin
  - main.py
  - /src
    - data_gen.py
    - nn_gen.py

## Running each model
