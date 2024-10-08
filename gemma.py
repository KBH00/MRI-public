from eval import EvalPolicy 
import pandas as pd
import numpy as np
from langchain.llms import Ollama  # 

class GEMMA:
    def __init__(self, T=11, snapshot_dir=None, log_dir=None, gamma=2.1, gamma_type='fixed', 
                 cog_init=None, adj=None, action_type='delta', action_limit=1.0, 
                 w_lambda=1.0, energy_model='inverse', model_name='gemma2'):
        """
        Initialize GEMMA for diagnosis with evaluation model integration.

        Parameters:
        - T (int): Number of time steps.
        - snapshot_dir (str): Directory for storing snapshots.
        - log_dir (str): Directory for logging.
        - gamma (float): Gamma parameter.
        - gamma_type (str): Type of gamma ('fixed' or 'variable').
        - cog_init: Initial cognition value.
        - adj: Adjacency matrix for the environment.
        - action_type (str): Type of action ('delta' or other).
        - action_limit (float): Action limit.
        - w_lambda (float): Lambda weight.
        - energy_model (str): Energy model type ('inverse' or other).
        - model_name (str): Name of the GEMMA2 model to load from Ollama.
        """
        self.T = T
        self.snapshot_dir = snapshot_dir
        self.log_dir = log_dir
        self.gamma = gamma
        self.gamma_type = gamma_type
        self.cog_init = cog_init
        self.adj = adj
        self.action_type = action_type
        self.action_limit = action_limit
        self.w_lambda = w_lambda
        self.energy_model = energy_model
        
        self.model = Ollama(model_name=model_name)

        self.eval_policy = EvalPolicy(
            T=self.T,
            snapshot_dir=self.snapshot_dir,
            log_dir=self.log_dir,
            gamma=self.gamma,
            gamma_type=self.gamma_type,
            cog_init=self.cog_init,
            adj=self.adj,
            action_type=self.action_type,
            action_limit=self.action_limit,
            w_lambda=self.w_lambda,
            energy_model=self.energy_model
        )

    def evaluate_and_diagnose(self, data):
        """
        Evaluate patient data using EvalPolicy and make a diagnosis.

        Parameters:
        - data: Initial values for X_V, D, alpha1, alpha2, beta, and cognition.

        Returns:
        - diagnosis_df: A DataFrame containing patient diagnosis results.
        """
        state_log, action_log, eval_output, state_log_dict = self.eval_policy.simulate(data=data, data_type='test')

        diagnosis_data = []
        for index, row in eval_output.iterrows():
            rid = row['RID']
            cog_score = row['cogsc_rl']
            severity = self.determine_severity(cog_score)
            diagnosis_data.append([rid, cog_score, severity])

        diagnosis_df = pd.DataFrame(diagnosis_data, columns=['RID', 'Cognition Score', 'Severity'])
        return diagnosis_df

    def determine_severity(self, cog_score):
        """
        Determine severity based on the cognition score.

        Parameters:
        - cog_score (float): Cognition score of the patient.

        Returns:
        - severity (str): Diagnosis severity (Normal, Mild, Moderate, Severe).
        """
        if cog_score < 0.5:
            return "Normal"
        elif 0.5 <= cog_score < 1.0:
            return "Mild"
        elif 1.0 <= cog_score < 1.5:
            return "Moderate"
        else:
            return "Severe"

    def load_and_diagnose(self, data):
        """
        Load the GEMMA2 model and perform diagnosis.

        Parameters:
        - data: Initial values for the model.

        Returns:
        - diagnosis_df: A DataFrame with the diagnosis results.
        """
        prompt = self.create_prompt(data)
        model_output = self.model.generate(prompt)
        
        return self.parse_model_output(model_output)

    def create_prompt(self, data):
        """
        Create a prompt for the model based on patient data.

        Parameters:
        - data: List containing patient data.

        Returns:
        - prompt (str): Structured prompt for diagnosis.
        """
        data_str = ", ".join([f"Value {i+1}: {value:.4f}" for i, value in enumerate(data)])
        prompt = f"Make a diagnosis for the following patient data: {data_str}. Provide the cognition score and severity."
        return prompt

    def parse_model_output(self, output):
        """
        Parse the model output into a DataFrame.

        Parameters:
        - output: Raw output from the GEMMA2 model.

        Returns:
        - diagnosis_df: A DataFrame with the diagnosis results.
        """

        diagnosis_data = []
        for entry in output:
            rid = entry['RID']
            cog_score = entry['cognition_score']
            severity = self.determine_severity(cog_score)
            diagnosis_data.append([rid, cog_score, severity])

        diagnosis_df = pd.DataFrame(diagnosis_data, columns=['RID', 'Cognition Score', 'Severity'])
        return diagnosis_df

if __name__ == "__main__":
    # Example usage of GEMMA
    # Replace with actual data
    data = [
        np.random.rand(10),  # Example for X_V
        np.random.rand(10),  # Example for D
        np.random.rand(10),  # Example for alpha1
        np.random.rand(10),  # Example for alpha2
        np.random.rand(10),  # Example for beta
        np.random.rand(10),  # Example for cognition
        np.arange(10)        # Example for RIDs
    ]

    gemma = GEMMA(T=11, snapshot_dir='snapshots', log_dir='logs')
    diagnosis_df = gemma.load_and_diagnose(data)

    print(diagnosis_df)
