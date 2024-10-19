import time
import pandas as pd
import re
import string
import json
from tqdm import tqdm
from langchain_community.chat_models import ChatOllama
import matplotlib.pyplot as plt

class SafetyObservationProcessor:
    def __init__(self, model_name="llama3.2:3b", temperature=0.1, top_k=100, top_p=0.1, seed=None, mirostat_tau=0.1, batch_size=10, risk_threshold=15):
        """
        Initializes the SafetyObservationProcessor with specified model parameters and settings.
        
        Parameters:
        - model_name (str): The name of the LLM model to use.
        - temperature (float): Sampling temperature for the LLM.
        - top_k (int): Top-K sampling for the LLM.
        - top_p (float): Top-p sampling for the LLM.
        - seed (int): Seed for the random generator to ensure reproducibility.
        - mirostat_tau (float): Tau parameter for Mirostat sampling.
        - batch_size (int): The number of prompts to process in one batch.
        - risk_threshold (int): The minimum risk score required to process a row. (15 for testing)
        """
        self.seed = seed if seed else int(time.time())  # Use provided seed or generate based on current time
        self.batch_size = batch_size
        self.risk_threshold = risk_threshold
        
        # Initialize the LLM model with the specified parameters
        self.llama = ChatOllama(model=model_name,
                                format="json",
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                seed=self.seed,
                                mirostat_tau=mirostat_tau)

    def preprocess_text(self, text):
        """
        Preprocesses text by converting to lowercase, removing punctuation, and digits.

        Parameters:
        - text (str): The input text to preprocess.

        Returns:
        - str: Cleaned and preprocessed text.
        """
        text = str(text).lower()  # Convert text to lowercase
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove digits
        return text

    def calculate_risk_score(self, text, keywords):
        """
        Calculates a risk score based on the number of occurrences of specific keywords in the text.

        Parameters:
        - text (str): The text to analyze.
        - keywords (list): A list of keywords related to high-risk scenarios.

        Returns:
        - int: The calculated risk score.
        """
        # Compile regex pattern from keywords to match any of the high-risk terms in the text
        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in keywords) + r')\b')
        return len(re.findall(pattern, text))  # Count the number of keyword matches

    def call_llm(self, question):
        """
        Calls the LLM model with a provided question prompt.

        Parameters:
        - question (str): The prompt to send to the LLM.

        Returns:
        - LLM response object.
        """
        return self.llama.invoke(question)

    def call_llm_with_retry(self, prompt, max_retries=3):
        """
        Calls the LLM with retry logic in case of errors such as JSON parsing errors.

        Parameters:
        - prompt (str): The prompt to send to the LLM.
        - max_retries (int): The number of retry attempts in case of failure.

        Returns:
        - dict or None: The JSON response if successful, otherwise None.
        """
        retries = 0
        while retries < max_retries:
            response = self.call_llm(prompt)
            try:
                response_content = response.content
                json_response = json.loads(response_content)  # Attempt to parse the JSON response
                return json_response  # Return parsed JSON
            except (json.JSONDecodeError, AttributeError) as e:
                retries += 1  # Increment retry counter
                print(f"Retry {retries} for prompt due to: {e}")
        return None  # Return None if all retries fail

    def process_row(self, row, risk_keywords, prompt_template):
        """
        Processes a single row of data, calculates the risk score, and if above the threshold, calls the LLM.

        Parameters:
        - row (pd.Series): A row of the DataFrame to process.
        - risk_keywords (list): A list of keywords related to SIF risks.
        - prompt_template (str): The prompt template to use for the LLM.

        Returns:
        - tuple: (JSON response, LLM status as str) or (None, error status).
        """
        # Extract relevant data from the row
        point_name = row['PNT_NM']
        qualifier_txt = row['QUALIFIER_TXT']
        atrisk_notes = row['PNT_ATRISKNOTES_TX']
        followup_notes = row['PNT_ATRISKFOLWUPNTS_TX']

        # Handle missing follow-up notes
        if pd.isna(followup_notes):
            followup_notes = "No follow-up notes provided."

        # Preprocess and combine the text fields
        combined_text = self.preprocess_text(f"{point_name} {qualifier_txt} {atrisk_notes} {followup_notes}")
        risk_score = self.calculate_risk_score(combined_text, risk_keywords)  # Calculate risk score

        # Check if risk score exceeds the threshold
        if risk_score > self.risk_threshold:
            # Format the prompt with the preprocessed text
            prompt = prompt_template.format(combined_text=combined_text)

            # Call the LLM with retry logic
            response_content = self.call_llm_with_retry(prompt)
            if response_content is None:
                return None, "LLM Failed"  # If LLM fails, return failure status

            # Check if response is already a dictionary (valid JSON)
            if isinstance(response_content, dict):
                return response_content, "Success"  # Return successful response
            else:
                # Attempt to parse JSON if not already a dictionary
                try:
                    json_response = json.loads(response_content)
                    return json_response, "Success"
                except json.JSONDecodeError:
                    return None, "JSON Parsing Error"  # Return parsing error status

    def process_data(self, dataframe, prompt_template, risk_keywords):
        """
        Processes the entire DataFrame, calls the LLM for rows exceeding the risk threshold, and updates the DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame to process.
        - prompt_template (str): The prompt template to use for the LLM.
        - risk_keywords (list): A list of keywords related to SIF risks.

        Returns:
        - pd.DataFrame: The updated DataFrame with LLM responses.
        - list: A list of failed responses for further investigation.
        """
        # Initialize columns for storing results
        dataframe['RESPONSE'] = None
        dataframe['HIGH_ENERGY'] = None
        dataframe['INCIDENT'] = None
        dataframe['INJURY'] = None
        dataframe['CONTROLS_PRESENT'] = None
        dataframe['SEVERITY_SCORE'] = None
        dataframe['LLM_STATUS'] = None
        dataframe['CONFIDENCE_SCORE'] = None
        failed_responses = []

        # Loop through each row in the DataFrame
        for index, row in tqdm(dataframe.iterrows(), desc="Assembling Prompts", total=len(dataframe), unit="row"):
            # Process the row and get the LLM response
            result, status = self.process_row(row, risk_keywords, prompt_template)

            if result:
                # Update DataFrame with LLM results
                dataframe.at[index, 'HIGH_ENERGY'] = result.get("high_energy_present", None)
                dataframe.at[index, 'INCIDENT'] = result.get("high_energy_incident", None)
                dataframe.at[index, 'INJURY'] = result.get("serious_injury_sustained", None)
                dataframe.at[index, 'CONTROLS_PRESENT'] = result.get("direct_controls_present", None)
                dataframe.at[index, 'SEVERITY_SCORE'] = result.get("severity_score", None)
                dataframe.at[index, 'CONFIDENCE_SCORE'] = result.get("confidence_score", None)
            else:
                # Append failed responses for debugging
                failed_responses.append({"index": index, "prompt": row, "error": status})

            dataframe.at[index, 'LLM_STATUS'] = status  # Record LLM status

            # Save progress every 100 rows
            if index % 100 == 0:
                dataframe.to_csv(f"output_partial_{index}.csv", index=False)

        return dataframe, failed_responses

    def save_failed_responses(self, failed_responses, file_name="failed_responses.json"):
        """
        Saves failed LLM responses to a JSON file for later analysis.

        Parameters:
        - failed_responses (list): A list of failed responses.
        - file_name (str): The filename for saving the failed responses.
        """
        with open(file_name, "w") as outfile:
            json.dump(failed_responses, outfile, indent=4)

    def save_output(self, dataframe, file_name="output_with_llm_json_responses.csv"):
        """
        Saves the updated DataFrame with LLM responses to a CSV file.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame with processed results.
        - file_name (str): The output filename.
        """
        dataframe.to_csv(file_name, index=False)
        print(f"Processing complete. LLM responses saved to {file_name}")

    def plot_severity_score_distribution(self, dataframe):
        """Plot distribution of severity scores."""
        plt.figure(figsize=(10, 6))
        plt.hist(dataframe['SEVERITY_SCORE'].dropna(), bins=range(1, 7), edgecolor='black', alpha=0.7)
        plt.title('Severity Score Distribution', fontsize=16)
        plt.xlabel('Severity Score (1 to 5)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.xticks(range(1, 6))  # Severity Score is from 1 to 5
        plt.grid(True)
        plt.show()

    def plot_confidence_score_distribution(self, dataframe):
        """Plot distribution of confidence scores."""
        plt.figure(figsize=(10, 6))
        plt.hist(dataframe['CONFIDENCE_SCORE'].dropna(), bins=10, edgecolor='black', alpha=0.7)
        plt.title('Confidence Score Distribution', fontsize=16)
        plt.xlabel('Confidence Score (0 to 1)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True)
        plt.show()

    def plot_high_risk_keyword_distribution(self, dataframe, risk_keywords):
        """Plot frequency of high-risk keywords."""
        keyword_count = {k: 0 for k in risk_keywords}
        
        # Preprocessing the combined text data and counting keywords
        for _, row in dataframe.iterrows():
            text = self.preprocess_text(f"{row['PNT_NM']} {row['QUALIFIER_TXT']} {row['PNT_ATRISKNOTES_TX']} {row['PNT_ATRISKFOLWUPNTS_TX']}")
            for keyword in risk_keywords:
                keyword_count[keyword] += text.count(keyword)

        # Create a bar plot for keyword frequency
        plt.figure(figsize=(12, 6))
        plt.bar(keyword_count.keys(), keyword_count.values(), color='royalblue', alpha=0.7)
        plt.title('High-Risk Keyword Frequency', fontsize=16)
        plt.xlabel('Keywords', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = SafetyObservationProcessor()

    # Load dataset
    DATA_FILE = r"CORE_HackOhio_subset_cleaned_downsampled 1.csv"
    dataframe = pd.read_csv(DATA_FILE)

    # Define prompt template
    prompt_template = """
    Safety Observation: {combined_text}.
    Based on the safety observation provided, please answer the following questions by providing a **valid JSON object**. All answers should be in exact field names and integer values only (1 for Yes, 0 for No). Ensure consistency with the safety guidelines provided below.

    JSON format:
    {{
    "high_energy_present": 1 or 0, 
    "high_energy_incident": 1 or 0, 
    "serious_injury_sustained": 1 or 0, 
    "direct_controls_present": 1 or 0, 
    "severity_score": 1 to 5, 
    "confidence_score": 0 to 1
    }}

    Guidelines:
    1. **Is high-energy present?**
    2. **Was there a high-energy incident?**
    3. **Was a serious injury sustained?**
    4. **Were direct controls present?**
    5. **Provide a severity score** from 1 (low) to 5 (high).
    6. **Confidence score** from 0 (low) to 1 (high).
    """

    # Keywords related to SIF risks
    SIF_KEYWORDS = [
        "suspend", "load", "fall", "elevation", "mobile", "equipment", "traffic", "motor", "vehicle",
        "heavy", "rotating", "machine", "temperature", "fire", "fuel", "explosion", "electrical", "arc"
    ]

    # Process the data
    processed_df, failed_responses = processor.process_data(dataframe, prompt_template, SIF_KEYWORDS)

    # Save results
    processor.save_output(processed_df)
    processor.save_failed_responses(failed_responses)

    # Visualization of results
    processor.plot_severity_score_distribution(processed_df)
    processor.plot_confidence_score_distribution(processed_df)
    processor.plot_high_risk_keyword_distribution(processed_df, SIF_KEYWORDS)
