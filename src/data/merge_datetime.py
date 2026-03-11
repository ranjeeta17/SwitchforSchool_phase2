import pandas as pd
from pathlib import Path

def merge_student_sessions(student_df, session_df):
    """
    Joins student check-in data with session metadata and 
    standardizes datetime formatting.
    """
    
    # Append session info where the Check In Session ID matches
    merged_df = pd.merge(
        student_df, 
        session_df[['Check In Session ID', 'Opened Date Time']], 
        on='Check In Session ID', 
        how='left'
    )
    
    # Date Conversion
    merged_df['Opened Date Time'] = pd.to_datetime(merged_df['Opened Date Time'])
    
    # Chronological Sort
    merged_df = merged_df.sort_values(by=['Student ID', 'Opened Date Time'])
    
    return merged_df


if __name__ == "__main__":
    data_folder = Path(__file__).parent.parent.parent / "datasets" 
    student_file = data_folder / "studentCheckIn_cleaned.csv"
    session_file = data_folder / "checkInSession.csv"
    output_file = data_folder / "studentCheckInDateTime.csv"

    if data_folder.exists():
        df_students = pd.read_csv(student_file)
        df_sessions = pd.read_csv(session_file)
        
        final_data = merge_student_sessions(df_students, df_sessions)
        
        final_data.to_csv(output_file, index=False)
        print("Merge complete. Datetime column standardised.")
    else:
        print(f"Error: The folder '{data_folder}' was not found.")
        print(Path(__file__).resolve().parent.parent.parent)
        print(Path.cwd())