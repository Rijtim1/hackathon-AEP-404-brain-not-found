# Safety Observation Analysis for AEP Hackathon

## Project Overview

This project is part of the American Electric Power (AEP) Safety Observation Challenge. The goal is to analyze safety observation data collected from field workers during "CORE visits" and identify the highest value comments. These comments highlight hazards that pose significant risks, such as those that could lead to serious injuries, damage to property, or environmental harm. The project leverages Large Language Models (LLMs) to extract high-risk insights from thousands of safety records and visualize trends.

## Files Included

- `CORE_HackOhio_subset_cleaned_downsampled.csv`: The dataset containing safety observations.
- `README.md`: Project documentation (this file).
- Python Scripts:
  - `main_analysis.py`: Main script to load data, preprocess, and perform LLM-based analysis.
  - `visualization.py`: Script for plotting graphs, including severity score distribution, confidence score distribution, risk score distribution, and high-risk keyword frequency.
  - `model_initialization.py`: Functions to initialize and manage LLM.
  - `data_processing.py`: Contains all data preprocessing functions such as text cleaning and keyword matching.

- Visualization files:
  - `confidence_score_distribution.png`
  - `severity_score_distribution.png`
  - `risk_score_distribution.png`
  - `high_risk_keyword_frequency.png`

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data](#data)
- [Usage](#usage)
- [Analysis Breakdown](#analysis-breakdown)
- [Model](#model)
- [Visualizations](#visualizations)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Getting Started

### Prerequisites

To run the analysis, ensure you have the following installed:

- Python 3.7+
- Pandas
- Numpy
- Matplotlib
- Langchain
- tqdm

To install these packages, run:

```bash
pip install -r requirements.txt
```

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-repo/aep-safety-observations
   cd aep-safety-observations
   ```

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the provided dataset (`CORE_HackOhio_subset_cleaned_downsampled.csv`) into the root directory.

## Data

The dataset contains safety observations made during "CORE visits," with the following columns:

- **Date**: When the observation was recorded.
- **Observation Type**: The category or safety concern.
- **Comments**: The main field for analysis, containing safety notes provided by supervisors.

## Usage

To run the full analysis, use the following command:

```bash
python main_analysis.py
```

### Core Functions

1. **LLM Initialization**: In `model_initialization.py`, the `initialize_llm` function sets up a Large Language Model (LLM) to process text data from the observations.

2. **Text Preprocessing**: In `data_processing.py`, the `preprocess_text` function cleans the raw text by removing punctuation, converting to lowercase, and removing numerical values.

3. **Keyword Risk Scoring**: The `calculate_risk_score` function calculates a risk score based on the frequency of specific keywords linked to high-risk hazards.

4. **LLM Analysis**: Each observation that passes a risk threshold is passed to the LLM, which responds to structured prompts based on safety assessment questions.

## Analysis Breakdown

### Steps in the Pipeline

1. **Data Loading**: The dataset is read and cleaned.
2. **Text Preprocessing**: Text from safety comments is cleaned using standard preprocessing methods (lowercasing, punctuation removal).
3. **Risk Scoring**: Using predefined high-risk keywords (such as "fall", "explosion", "shock"), a risk score is calculated for each observation.
4. **LLM Prompting**: The LLM is used to assess the riskier comments, responding to prompts with details like whether high-energy is present or whether direct controls are in place.
5. **Visualization**: The results are visualized through histograms and bar charts that help explore key insights.

## Model

We used an open-source LLM (`llama3.2:3b`) to process and analyze the safety observation data. The LLM was queried using structured prompts, based on the Safety Classification and Learning (SCL) Model from the [EEI SCL document](12), to identify the presence of high-energy sources, incidents, injuries, and safety controls.

## Visualizations

### 1. Confidence Score Distribution

- **Graph**: `confidence_score_distribution.png`
- **Description**: Shows the confidence levels of the LLM when classifying observations. Scores range from 0 (low confidence) to 1 (high confidence).

### 2. Severity Score Distribution

- **Graph**: `severity_score_distribution.png`
- **Description**: Displays the distribution of severity scores, ranging from 1 to 5.

### 3. Risk Score Distribution

- **Graph**: `risk_score_distribution.png`
- **Description**: Shows the spread of risk scores based on keyword analysis.

### 4. High-Risk Keyword Frequency

- **Graph**: `high_risk_keyword_frequency.png`
- **Description**: Plots the frequency of high-risk keywords (like "fall", "electric", "fire").

## Results

The main outputs include:

- **Processed CSV File**: The analyzed dataset with added columns for LLM predictions on high-energy presence, incident severity, injury likelihood, and confidence scores.
- **Visualizations**: Plots that visualize the distribution of risk scores, severity scores, and the occurrence of high-risk keywords.

## Acknowledgments

This project was completed for the AEP Safety Observation Challenge. The Safety Classification and Learning (SCL) model by Edison Electric Institute was used as a reference for designing the LLM prompts. Special thanks to the HackOHI/O organizers and AEP for providing the dataset and challenge details【13†source】.

---

Feel free to modify the content based on your specific implementation details!
