# Function to apply justified formatting to any DataFrame
def styled_df(df):
    """
    Apply left-justification for non-numeric columns and right-justification for numeric columns.
    """
    # Create a custom style for the DataFrame
    styled_df = df.style.set_table_styles(
        # Left justify all column headers
        [{'selector': 'th', 'props': [('text-align', 'left')]},  
         # Left justify non-numeric cells (text or string columns)
         {'selector': 'td', 'props': [('text-align', 'left')]}, 
         # Right justify numeric columns (for numbers)
         {'selector': 'td:nth-child(n+1)', 'props': [('text-align', 'right')]},  
        ])
    
    return df

        
        # Example usage with any DataFrame:
        # styled_df = apply_justification(df)
        # styled_df  # This would display your DataFrame with the desired justification in Jupyter


        