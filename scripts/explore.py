
import pandas as pd



# Calculate the percentage of rows with missing imdbId
def calculate_missing_percent_and_clean(datasets, platforms):
    missing_percentages = []

    for platform, df in zip(platforms, datasets):
        # Calculate the percentage of missing imdbId
        missing_count = df["imdbId"].isnull().sum()
        total_count = len(df)
        missing_percentage = (missing_count / total_count) * 100

        # Store the platform and percentage
        missing_percentages.append({"Platform": platform, "Missing Percent": missing_percentage})

        # Remove rows with missing imdbId
        datasets[platforms.index(platform)] = df.dropna(subset=["imdbId"]).reset_index(drop=True)

    # Create a DataFrame for the missing percentages
    missing_df = pd.DataFrame(missing_percentages).sort_values(by="Missing Percent", ascending=False)

    return missing_df, datasets






# Function to generate separate tables for dataset insights
def dataset_summary_separate(datasets, platforms):
    summaries = {"Missing Values": [], "Data Type": [], "Unique Values": [], "Duplicates": []}

    for platform, df in zip(platforms, datasets):
        # Calculate missing values
        missing_values = pd.DataFrame({
            "Column": df.columns,
            platform: df.isnull().sum().values
        }).set_index("Column")

        # Calculate data types
        data_types = pd.DataFrame({
            "Column": df.columns,
            platform: df.dtypes.values
        }).set_index("Column")

        # Calculate unique values
        unique_values = pd.DataFrame({
            "Column": df.columns,
            platform: df.nunique().values
        }).set_index("Column")

        # Calculate duplicate counts
        duplicates = pd.DataFrame({
            "Column": df.columns,
            platform: [
                df.duplicated(subset=[col]).sum() if df[col].nunique() < len(df) else 0
                for col in df.columns
            ]
        }).set_index("Column")

        # Append results to summaries
        summaries["Missing Values"].append(missing_values)
        summaries["Data Type"].append(data_types)
        summaries["Unique Values"].append(unique_values)
        summaries["Duplicates"].append(duplicates)

    # Combine all results into separate tables
    separate_tables = {metric: pd.concat(frames, axis=1) for metric, frames in summaries.items()}
    
    return separate_tables


# Function to reprint the missing values table
def print_missing_values(datasets, platforms):
    # Compile missing values for all platforms
    missing_values = {platform: df.isnull().sum() for platform, df in zip(platforms, datasets)}
    
    # Create a consolidated DataFrame
    missing_values_table = pd.DataFrame(missing_values)
    
    print("----- Missing Values -----")
    return missing_values_table



# Function to drop duplicate rows based on imdbId for each platform's dataset
def drop_duplicates_by_imdbId(platforms_datasets):
    for df in platforms_datasets:
        # Drop duplicate rows based on imdbId column
        df.drop_duplicates(subset='imdbId', keep='first', inplace=True)
    return platforms_datasets

# Function to drop specified columns from all platform datasets
def drop_columns_from_datasets(datasets, columns_to_drop):
    for df in datasets:
        # Drop specified columns from the dataframe
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    return datasets


# Function to drop rows without "US" and calculate the number of rows with "US" and non-"US"
def process_platform_data(datasets, platforms):
    availability_data = []
    cleaned_datasets = {}  # To store cleaned datasets
    
    for platform, df in zip(platforms, datasets):
        # Count rows with "US"
        us_rows = df[df['availableCountries'].str.contains('US', na=False)]
        non_us_rows = df[~df['availableCountries'].str.contains('US', na=False)]
        
        # Calculate percentages
        total_rows = len(df)
        us_count = len(us_rows)
        non_us_count = len(non_us_rows)
        us_percentage = (us_count / total_rows) * 100 if total_rows > 0 else 0
        non_us_percentage = (non_us_count / total_rows) * 100 if total_rows > 0 else 0
        
        # Store availability data
        availability_data.append({
            'platform': platform,
            'us_count': us_count,
            'us_percentage': us_percentage,
            'non_us_count': non_us_count,
            'non_us_percentage': non_us_percentage
        })
        
        # Drop rows without "US"
        cleaned_datasets[platform] = us_rows  # Store cleaned dataframe for the platform
    
    # Save availability data
    availability_df = pd.DataFrame(availability_data)
    availability_df.to_csv(r'C:\Users\kimbe\Documents\StreamingAnalysis\data\cleaned_data\availability.csv', index=False)
    
    return cleaned_datasets


# Function to add platform-specific column to each dataset
def add_platform_column(df, platform_name):
    df[platform_name] = 1  # Mark all rows as available in the platform
    return df

# Function to merge platform datasets on imdbId and handle NaN for availability
def merge_platform_datasets(platforms, platform_dfs):
    # Add platform-specific columns to each dataset
    for i, platform in enumerate(platforms):
        platform_dfs[i] = add_platform_column(platform_dfs[i], platform)
    
    # Combine all datasets into a single dataframe
    all_platforms = pd.concat(platform_dfs, ignore_index=True)
    
    # Perform the groupby operation with custom aggregation logic
    merged_df = all_platforms.groupby('imdbId', as_index=False).agg({
        'type': 'first',        # Take the first non-null value
        'genres': 'first',      # Take the first non-null value
        'releaseYear': 'first', # Take the first non-null value
        'title': 'first',       # Take the first non-null value
        **{platform: 'max' for platform in platforms}  # Max will consolidate availability across datasets
    })
    
    # Replace NaN values in platform-specific columns with 0 and ensure binary representation (0 or 1)
    for platform in platforms:
        merged_df[platform] = merged_df[platform].fillna(0).astype(int)  # Modify directly and assign
    
    return merged_df
