import pandas as pd
import numpy as np

class StudentBaselineBuilder:
    """
    Implements Epic 1: Student Baseline & Radical Change Detection.
    Includes data cleaning to handle string-to-numeric conversion errors.
    """
    
    def __init__(self, window_size=4):
        self.window_size = window_size

    def calculate_baselines(self, df):
        """
        Builds a student-level 'normal state' baseline.
        Ensures columns are numeric before calculating mean and std.
        """
        # Create a copy to avoid SettingWithCopy warnings
        df = df.copy()

        # Convert columns to numeric, forcing errors to NaN (e.g., if there are '%' or text)
        df['Emotion Intensity Percentage'] = pd.to_numeric(df['Emotion Intensity Percentage'], errors='coerce')
        df['Tiredness'] = pd.to_numeric(df['Tiredness'], errors='coerce')
        
        # Drop rows where numeric conversion failed for critical metrics
        df = df.dropna(subset=['Emotion Intensity Percentage', 'Tiredness'])

        # Ensure data is sorted by Student ID and Session ID
        df = df.sort_values(['Student ID', 'Check In Session ID'])
        
        # Aggregate historical metrics
        baselines = df.groupby('Student ID').agg({
            'Emotion Intensity Percentage': ['mean', 'std'],
            'Tiredness': ['mean', 'std']
        }).reset_index()
        
        # Flatten MultiIndex columns
        baselines.columns = [
            'Student ID', 
            'int_mean', 'int_std', 
            'tired_mean', 'tired_std'
        ]
        return baselines

    def detect_radical_difference(self, current_df, baselines):
        """
        Produces a 'radical difference score' using Z-score logic.
        """
        # Ensure current_df columns are also numeric
        current_df = current_df.copy()
        current_df['Tiredness'] = pd.to_numeric(current_df['Tiredness'], errors='coerce')
        
        # Merge with baselines
        merged = pd.merge(current_df, baselines, on='Student ID')
        
        # Calculate Deviation Score (Z-Score)
        # We add 0.1 to std to prevent division by zero
        merged['deviation_score'] = (
            (merged['Tiredness'] - merged['tired_mean']).abs() / (merged['tired_std'] + 0.1)
        )
        
        # Assign Labels
        merged['deviation_label'] = merged['deviation_score'].apply(
            lambda x: 'High' if x > 2.0 else ('Moderate' if x > 1.2 else 'Normal')
        )
        
        merged['explanation'] = merged.apply(self._generate_explanation, axis=1)
        
        return merged[['Student ID', 'deviation_score', 'deviation_label', 'explanation']]

    def _generate_explanation(self, row):
        if row['deviation_label'] == 'High':
            return "Radical spike in tiredness compared to personal baseline."
        elif row['deviation_label'] == 'Moderate':
            return "Moderate shift in wellbeing patterns detected."
        return "Within normal patterns."

if __name__ == "__main__":
    print("Student Baseline Builder module updated with numeric handling.")