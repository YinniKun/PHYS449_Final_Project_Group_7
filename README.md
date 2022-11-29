# PHYS 449 Final Project - Systems biology informed deep learning for inferring parameters and hidden dynamics
Group 7 : Richard (Zhifei) Dong, Callum Follett, Nolan Anthony Paul Johnston, Steph Swanson, Jack Zhao

Origninal paper from Yazdani A, Lu L, Raissi M, Karniadakis GE. Systems biology informed deep learning for inferring parameters and hidden dynamics. PLoS computational biology. 2020;16(11):e1007575.

## Introduction
This repo contains the code and data to recreate the work by Yazdani et al., 2020. In their work, Yazdani et al. utilized deep learning to determie the parameters for systems biology models.

## File structure
Three models are tested and validated with the neural network architecture, each with a ``main.py`` that is used to train the model and utilizes ``plot.py`` to generate plots from the results. There are also two source files ``nn_gen.py`` and ``data_gen.py`` that contains the neutral network architecture and the code for generating the data for training in the ``./src/`` directory. Lastly, a ``param.json`` can also be found in the ``./src/`` direcotry that contains all the necessary hyperparameters.

The file structre is outlined below:

- ~/apoptosis
  - main.py
  - plot.py
  - ./src/
    - data_gen.py
    - nn_gen.py
    - param.json
- ~/glycolysis
  - main.py
  - plot.py
  - ./src/
    - data_gen.py
    - nn_gen.py
    - param.json
- ~/insulin
  - main.py
  - plot.py
  - ./src/
    - data_gen.py
    - nn_gen.py
    - param.json

## Dependencies
- sys, argparse, os
- math
- numpy
- torch
- matplotlib.pyplot

## Running each model

To run each model, go to the directory of the model, and use:

``python3 main.py``
