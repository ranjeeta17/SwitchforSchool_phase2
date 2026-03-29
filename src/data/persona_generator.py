from pathlib import Path
import numpy as np
import pandas as pd
import datetime

class TemporalController:
    """Handles the mathematical progression of a student's state over time."""
    
    @staticmethod
    def get_modifier(type, day, total_days, variance=0):
        x = day / total_days
        # Add variance to the time progression itself if requested
        x = np.clip(x + np.random.uniform(-variance, variance), 0, 1)
        
        if type == 'linear':
            return x
        elif type == 'sigmoid':
            # K=10 creates a sharp transition at the midpoint (0.5)
            return 1 / (1 + np.exp(-10 * (x - 0.5)))
        elif type == 'sine':
            # One full cycle over the duration
            return (np.sin(2 * np.pi * x) + 1) / 2
        elif type == 'step':
            return 1.0 if x > 0.5 else 0.0
        return 0.5  # Neutral default

class CorrelationEngine:
    """Applies the 'Stress Variable' logic to modulate probabilities."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled

    def modulate_weights(self, base_weights, stress_level, feature_type):
        if not self.enabled:
            return base_weights
        
        # Stress level is 0.0 (Chill) to 1.0 (Crisis)
        modulated = base_weights.copy()
        
        # Logic: If high stress, boost negative outcomes
        if feature_type == 'Emotion':
            negatives = ['SAD', 'ANGRY', 'ANXIOUS']
            positives = ['HAPPY', 'EXCITED']
            for k in modulated:
                if k in negatives: modulated[k] *= (1 + stress_level)
                if k in positives: modulated[k] *= (1 - stress_level)
        
        elif feature_type == 'Tiredness':
            # Higher stress = Higher tiredness weights
            for k in modulated:
                if k >= 4: modulated[k] *= (1 + stress_level)
                if k <= 2: modulated[k] *= (1 - stress_level)

        elif feature_type == 'Eating':
            if 'NO' in modulated: modulated['NO'] *= (1 + stress_level)
            if 'YES' in modulated: modulated['YES'] *= (1 - stress_level)

        # Re-normalize weights so they sum to 1.0
        total = sum(modulated.values())
        return {k: v / total for k, v in modulated.items()}

class PersonaFactory:
    def __init__(self, use_stress=True):
        self.correlator = CorrelationEngine(enabled=use_stress)

    def generate(self, student_id, persona, days=100, trend='linear', variance=0.1):
        data = []
        start_date = datetime.datetime(2026, 1, 1) # Set to current year
        
        for d in range(days):
            # 1. Get the base stress level for this day from the math function
            stress_level = TemporalController.get_modifier(trend, d, days, variance)
            
            row = {'Student_ID': student_id, 'Day': start_date + datetime.timedelta(days=d), 'Latent_Stress': stress_level}
            
            # 2. Generate features using the Correlation Engine
            for attr, weights in persona.items():

                # Populate fields with "UNDEFINED" if student absent for check in session
                if attr != "Absence" and row.get("Absence") == "TRUE":
                    row[attr] = "UNDEFINED"
                    continue
                
                # Modulate weights based on the day's stress
                final_weights = self.correlator.modulate_weights(weights, stress_level, attr)
                
                # Apply Variance: As variance increases, weights flatten (Entropy)
                if variance > 0:
                    choices = list(final_weights.keys())
                    raw_w = np.array(list(final_weights.values()))
                    # Power-scaling: v=0 (no change), v=1 (uniform/random)
                    flat_w = raw_w**(1 - variance)
                    final_weights = dict(zip(choices, flat_w / flat_w.sum()))

                row[attr] = np.random.choice(list(final_weights.keys()), p=list(final_weights.values()))
            
            data.append(row)
        return pd.DataFrame(data)

# --- EXAMPLE USAGE ---

# Define paths
try:
    # .py file
    root = Path(__file__).resolve().parents[2]
except NameError:
    #.ipynb file
    root = Path.cwd().resolve().parent.parent

save_path = root / "datasets"


# Define the student archetype (the baseline probabilities)
BASE_OPTIMIST = {
    "Absence": {"TRUE": 0.1, "FALSE": 0.9},
    "Emotion": {"HAPPY": 0.8, "SAD": 0.1, "ANXIOUS": 0.1},
    "Intensity": {0.14: 0.14, 0.29: 0.14, 0.43: 0.14, 0.57: 0.14, 0.71: 0.14, 0.86: 0.14, 1: 0.16},
    "Tiredness": {1: 0.6, 2: 0.3, 5: 0.1},
    "Request For Chat": {"YES": 0.2, "NO": 0.6, "NOT_YET": 0.2},
    "Eating": {"YES": 0.9, "NO": 0.1}
}

engine = PersonaFactory(use_stress=True)

# Generate a student whose stress gradually increases and then plateaus (Sigmoid)
student_df = engine.generate(
    student_id="Student_01", 
    persona=BASE_OPTIMIST, 
    trend='sigmoid', 
    variance=0.05
)

student_df.to_csv(save_path / "personas.csv", index=False)
print(student_df.head())