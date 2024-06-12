# building-feature-characterization-tool

## Table of Contents

1. [Introduction](#introduction)
2. [Purpose](#purpose)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)


## Introduction

This is a Streamlit web application designed to characterize a building's energy-related features through heating and cooling load signatures.

## Purpose

The app aims to serve professionals and researchers in the field of Building Engineering. Specifically, the tool can be used to:

- Benchmark buildings
- Extract operating hours
- Extract change point models
- Characterize energy related features relating to the envelope, casual heat gains, and HVAC operation
- Determine which energy-related features have retrofit potential


## Requirements
- Python 3.9.19
  
The application requires the following Python packages:

- matplotlib==3.7.1
- numpy==1.24.2
- pandas==1.5.3
- Pillow==10.0.0
- plotly==5.13.1
- ruptures==1.1.7
- scikit_learn==1.2.1
- scipy==1.11.2
- seaborn==0.12.2
- streamlit==1.19.0
- tensorflow==2.11.0

## Installation

1. Clone the GitHub repository:

    ```
    git clone https://github.com/Carleton-DBOM-Research-Group/building-feature-characterization-tool.git
    ```

2. Navigate to the project directory and install the required Python packages:

    ```
    cd building-feature-characterization-tool
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```
    streamlit run BFCT.py
    ```

2. Open the app in your web browser using the link provided in the terminal.

3. Follow the in-app guidelines to input the required building parameters and upload the necessary files.

4. View the results.

