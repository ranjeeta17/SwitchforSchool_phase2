import pandas as pd
import matplotlib.pyplot as plt
from src.baseline_builder import StudentBaselineBuilder

def plot_deviations(results):
    """
    Generates a bar chart showing the distribution of student risk levels.
    This fulfills the 'Simple Plot' requirement of Epic 1.
    """
    plt.figure(figsize=(10, 6))
    
    # Define professional colors for each risk level
    colors = {'High': '#e74c3c', 'Moderate': '#f39c12', 'Normal': '#2ecc71'}
    
    # Count how many students fall into each category
    counts = results['deviation_label'].value_counts()
    
    # Ensure all categories are shown in the correct order
    counts = counts.reindex(['Normal', 'Moderate', 'High'], fill_value=0)
    
    # Create the bar plot
    counts.plot(kind='bar', color=[colors[x] for x in counts.index])
    
    plt.title('Class Wellbeing Deviation Overview (Epic 1)')
    plt.xlabel('Risk Level (Based on Radical Difference Score)')
    plt.ylabel('Number of Students')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the plot window
    print("\nDisplaying distribution plot... Close the plot window to continue.")
    plt.show()

def run_baseline_test():
    """
    Main execution logic to load data, detect anomalies, and visualize results.
    """
    try:
        # 1. Load your local CSV file
        df = pd.read_csv('studentCheckInDateTime.csv')
        
        # 2. Initialize the builder module you wrote
        builder = StudentBaselineBuilder(window_size=4)
        
        # 3. Calculate historical baselines
        baselines = builder.calculate_baselines(df)
        
        # 4. Get the latest check-in data for each student
        current_checkins = df.groupby('Student ID').tail(1)
        
        # 5. Detect radical differences
        alerts = builder.detect_radical_difference(current_checkins, baselines)
        
        # 6. Print the summary table in the terminal
        print("--- Epic 1: Weekly Anomaly Summary Table ---")
        summary = alerts[alerts['deviation_label'] != 'Normal'].sort_values(
            by='deviation_score', ascending=False
        )
        print(summary.head(10))

        # 7. Generate and show the plot
        plot_deviations(alerts)
            
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    run_baseline_test()
    
