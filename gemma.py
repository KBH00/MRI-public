from langchain_community.llms import Ollama

# Initialize GEMMA2 model
llm = Ollama(model="gemma2:2b")

# Function to handle medical question answering
def medical_question_answering(mri_results, patient_data):
    """
    Function to query GEMMA with MRI results and patient metadata.
    Args:
        mri_results (dict): Diagnostic results from MRI (e.g., disease detected).
        patient_data (dict): Patient metadata such as age, sex, etc.
    
    Returns:
        response (str): Natural language explanation of the diagnosis, prognosis, and potential treatment.
    """
    
    prompt = f"""
    Based on the following MRI diagnostic results:
    {mri_results}
    
    And the following patient data:
    Age: {patient_data['age']}
    Sex: {patient_data['sex']}
    Cognitive Scores: {patient_data.get('cognitive_scores', 'Not provided')}
    
    Please provide:
    - A diagnosis explanation
    - Potential prognoses
    - Recommended treatments
    """
    
    try:
        response = llm.invoke(prompt)
        return response
    
    except Exception as e:
        return f"An error occurred: {e}"

# Example MRI results and patient data
mri_results = {
    "disease_detected": "Vascular burden",
    "severity": "Moderate",
    "affected_region": "Frontal lobe",
    "additional_findings": "Minor atrophy"
}

patient_data = {
    "age": 65,
    "sex": "Female",
    "cognitive_scores": "Mild cognitive impairment"
}

response = medical_question_answering(mri_results, patient_data)
print("GEMMA's response:", response)
