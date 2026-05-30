import json
import pandas as pd
from pathlib import Path
from persona_generator import PersonaFactory  # Import your engine

def run_suite(config_path, output_dir, days=100):
    # 1. Load the Configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    suite_name = config['suite_metadata']['name']
    engine = PersonaFactory(use_stress=True)
    all_student_data = []
    manifest = []

    # 2. Iterate through archetypes defined in JSON
    for archetype_name, archetype_data in config['archetypes'].items():
        
        # Extract generation parameters from the new 'config' block
        gen_config = archetype_data.get('config', {})
        student_count = gen_config.get('count', 1)
        trend = gen_config.get('trend', 'linear')
        variance = gen_config.get('variance', 0.05)
        
        # Extract the probability matrices from the 'persona' block
        raw_persona = archetype_data.get('persona', {})

        # Professional Tip: Convert JSON string keys (e.g., "0.33") back to floats
        processed_persona = {}
        for attr, dist in raw_persona.items():
            processed_persona[attr] = {float(k) if isinstance(k, str) and k.replace('.','',1).isdigit() else k: v 
                                       for k, v in dist.items()}

        # 3. Loop to create 'n' number of unique students for this archetype
        for i in range(student_count):
            # Create a unique ID using a formatted string (e.g., "optimist_yr1_001")
            student_id = f"{archetype_name}_{i+1:03d}"
            
            # Generate the data using the engine with dynamic parameters
            df = engine.generate(
                student_id=student_id, 
                persona=processed_persona, 
                days=days, 
                trend=trend,
                variance=variance
            )
            all_student_data.append(df)
            
            # 4. Track Metadata for your team's review
            manifest.append({
                "id": student_id, 
                "archetype": archetype_name, 
                "days": days,
                "trend": trend,
                "variance": variance
            })

    # 5. Save Results
    # Added a check to safely create the output directory if it doesn't exist yet
    Path(output_dir).mkdir(parents=True, exist_ok=True) 
    
    final_df = pd.concat(all_student_data)
    save_name = suite_name.lower().replace(" ", "_")
    final_df.to_csv(Path(output_dir) / f"{save_name}_data.csv", index=False)
    
    print(f"Successfully generated {suite_name} with {len(manifest)} total students.")

if __name__ == "__main__":
    current_file_dir = Path(__file__).resolve().parent
    dataset_path = current_file_dir.parent.parent / "datasets"
    run_suite(current_file_dir / 'bronze_suite.json', dataset_path, days=10000)

    # print(current_file_dir)
    # print(dataset_path)