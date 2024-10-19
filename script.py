import time
import pandas as pd
import re
import string
import json
from tqdm import tqdm
from langchain_community.chat_models import ChatOllama

class SafetyObservationProcessor:
    def __init__(self, model_name="llama3.2:3b", temperature=0.1, top_k=100, top_p=0.1, seed=None, mirostat_tau=0.1, batch_size=10, risk_threshold=15):
        # Initialize model parameters and other settings
        self.seed = seed if seed else int(time.time())
        self.batch_size = batch_size
        self.risk_threshold = risk_threshold
        
        # Initialize the LLM model
        self.llama = ChatOllama(model=model_name,
                                format="json",
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                seed=self.seed,
                                mirostat_tau=mirostat_tau)

    def preprocess_text(self, text):
        """Preprocesses text data by removing punctuation and converting to lowercase."""
        text = str(text).lower()
        text = text.replace('\n', ' ')
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove digits
        return text

    def calculate_risk_score(self, text, keywords):
        """Calculates risk score based on the presence of specific SIF keywords."""
        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in keywords) + r')\b')
        return len(re.findall(pattern, text))

    def call_llm(self, question):
        """Calls the LLM to process the given prompt."""
        return self.llama.invoke(question)

    def call_llm_with_retry(self, prompt, max_retries=3):
        """Retries LLM call in case of errors."""
        retries = 0
        while retries < max_retries:
            response = self.call_llm(prompt)
            try:
                response_content = response.content
                json_response = json.loads(response_content)
                return json_response  # Return valid JSON response
            except (json.JSONDecodeError, AttributeError) as e:
                retries += 1
                print(f"Retry {retries} for prompt due to: {e}")
        return None  # Return None if all retries fail

    def process_row(self, row, risk_keywords, prompt_template):
        """Processes a single row of data by calling the LLM and updating the DataFrame."""
        point_name = row['PNT_NM']
        qualifier_txt = row['QUALIFIER_TXT']
        atrisk_notes = row['PNT_ATRISKNOTES_TX']
        followup_notes = row['PNT_ATRISKFOLWUPNTS_TX']

        if pd.isna(followup_notes):
            followup_notes = "No follow-up notes provided."

        combined_text = self.preprocess_text(f"{point_name} {qualifier_txt} {atrisk_notes} {followup_notes}")
        risk_score = self.calculate_risk_score(combined_text, risk_keywords)

        if risk_score > self.risk_threshold:
            prompt = prompt_template.format(combined_text=combined_text)

            # Call the LLM with retry logic
            response_content = self.call_llm_with_retry(prompt)
            if response_content is None:
                return None, "LLM Failed"

            # Process the JSON response
            if isinstance(response_content, dict):
                return response_content, "Success"
            else:
                try:
                    json_response = json.loads(response_content)
                    return json_response, "Success"
                except json.JSONDecodeError:
                    return None, "JSON Parsing Error"

    def process_data(self, dataframe, prompt_template, risk_keywords):
        """Processes the entire dataset and applies the LLM model to rows exceeding the risk threshold."""
        dataframe['RESPONSE'] = None
        dataframe['HIGH_ENERGY'] = None
        dataframe['INCIDENT'] = None
        dataframe['INJURY'] = None
        dataframe['CONTROLS_PRESENT'] = None
        dataframe['SEVERITY_SCORE'] = None
        dataframe['LLM_STATUS'] = None
        dataframe['CONFIDENCE_SCORE'] = None
        failed_responses = []

        # Main loop to process each row and call the LLM
        for index, row in tqdm(dataframe.iterrows(), desc="Assembling Prompts", total=len(dataframe), unit="row"):
            result, status = self.process_row(row, risk_keywords, prompt_template)

            if result:
                dataframe.at[index, 'HIGH_ENERGY'] = result.get("high_energy_present", None)
                dataframe.at[index, 'INCIDENT'] = result.get("high_energy_incident", None)
                dataframe.at[index, 'INJURY'] = result.get("serious_injury_sustained", None)
                dataframe.at[index, 'CONTROLS_PRESENT'] = result.get("direct_controls_present", None)
                dataframe.at[index, 'SEVERITY_SCORE'] = result.get("severity_score", None)
                dataframe.at[index, 'CONFIDENCE_SCORE'] = result.get("confidence_score", None)
            else:
                failed_responses.append({"index": index, "prompt": row, "error": status})

            dataframe.at[index, 'LLM_STATUS'] = status

            # Save progress every 100 rows
            if index % 100 == 0:
                dataframe.to_csv(f"output_partial_{index}.csv", index=False)

        return dataframe, failed_responses

    def save_failed_responses(self, failed_responses, file_name="failed_responses.json"):
        """Saves failed LLM responses to a file for later analysis."""
        with open(file_name, "w") as outfile:
            json.dump(failed_responses, outfile, indent=4)

    def save_output(self, dataframe, file_name="output_with_llm_json_responses.csv"):
        """Saves the final processed DataFrame to a CSV file."""
        dataframe.to_csv(file_name, index=False)
        print(f"Processing complete. LLM responses saved to {file_name}")

# Example usage
if __name__ == "__main__":
    processor = SafetyObservationProcessor()
    DATA_FILE = r"CORE_HackOhio_subset_cleaned_downsampled.csv"
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

    SIF_KEYWORDS = [
        "suspend", "load", "fall", "elevation", "mobile", "equipment", "traffic", "motor", "vehicle",
        "heavy", "rotating", "machine", "temperature", "fire", "fuel", "explosion", "electrical", "arc"
    ]

    processed_df, failed_responses = processor.process_data(dataframe, prompt_template, SIF_KEYWORDS)
    processor.save_output(processed_df)
    processor.save_failed_responses(failed_responses)
