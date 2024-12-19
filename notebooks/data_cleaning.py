# %% [markdown]
# # Datasets Overview

# %% [markdown]
# This project utilizes five datasets from Kaggle, providing comprehensive information on popular streaming platforms and IMDb ratings. Each dataset is updated daily, ensuring accurate and relevant content.
# 
# 1. **Netflix**
# 
#     * Source: [Netflix Movies & TV Series Dataset](https://www.kaggle.com/datasets/octopusteam/full-netflix-dataset)
#     * **Description**: A complete collection of Netflix's available titles (movies and TV series) with IMDb-specific data such as IMDb ID, average rating, and number of votes.
# 
# 2. **Apple TV+**
# 
#     * Source: [Full Apple TV+ Dataset](https://www.kaggle.com/datasets/octopusteam/full-apple-tv-dataset)
#     * Description: A dataset covering all Apple TV+ titles, including key IMDb data for in-depth analysis of content quality.
# 
# 3. **HBO Max**
# 
#     * Source:  [Full HBO Max Dataset](https://www.kaggle.com/datasets/octopusteam/full-hbo-max-dataset)
#     * Description: An extensive collection of titles on HBO Max with associated IMDb data for comparison.
# 
# 3. **Amazon Prime**
# 
#    * Source: [Full Amazon Prime Dataset](https://www.kaggle.com/datasets/octopusteam/full-amazon-prime-dataset)
#     * Description: Comprehensive data on Amazon Prime's movie and TV series offerings, including IMDb-specific metrics.
# 
# 4. **Hulu**
# 
#     * Source: [Full Hulu Dataset](https://www.kaggle.com/datasets/octopusteam/full-hulu-dataset)
#     * Description: A dataset detailing Hulu's catalog with IMDb-related columns for evaluating content quality and popularity.
# 
# Each of the streaming platform datasets includes the following columns:
# 
# * **title**: Name of the content.
# * **type**: Either "movie" or "tv series."
# * **genres**: Genres associated with the title.
# * **releaseYear**: Year the title was released.
# * **imdbId**: Unique IMDb identifier.
# * **imdbAverageRating**: Average user rating on IMDb.
# * **imdbNumVotes**: Number of votes received on IMDb.
# * **availableCountries**: Countries where the title is available.

# %% [markdown]
# ### Imports

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# %% [markdown]
# ### Load Datasets

# %%
# Define the base path for the raw data files
base_path = r"C:\Users\kimbe\Documents\StreamingAnalysis\data\raw_data"

# Load the streaming platform datasets
amazon_df = pd.read_csv(f'{base_path}\\amazon_catalog.csv')
hulu_df = pd.read_csv(f'{base_path}\\hulu_catalog.csv')
netflix_df = pd.read_csv(f'{base_path}\\netflix_catalog.csv')
hbo_df = pd.read_csv(f'{base_path}\\hbo_catalog.csv')
apple_df = pd.read_csv(f'{base_path}\\apple_catalog.csv')

# Load the IMDb datasets
basics_df = pd.read_csv(f"{base_path}\\imdb_basics.csv", low_memory=False)
ratings_df = pd.read_csv(f"{base_path}\\imdb_ratings.csv", low_memory=False)

# Define the list of platform names
platforms = ['Amazon', 'Hulu', 'Netflix', 'HBO', 'Apple']

# Create a list of DataFrames corresponding to each platform
platform_dfs = [amazon_df, hulu_df, netflix_df, hbo_df, apple_df]


# %% [markdown]
# ### Custom Functions

# %%
def clean_and_analyze_df(df):
    """
    Cleans and analyzes a DataFrame by handling missing and duplicate 'imdbId' values.

    This function performs the following steps:
    1. Prints the row count before cleaning.
    2. Replaces occurrences of '\\N' with NaN.
    3. If the 'imdbId' column exists:
        a. Trims white spaces in the 'imdbId' column.
        b. Displays duplicate 'imdbId' values along with their titles.
        c. Displays rows with missing 'imdbId' values along with their titles.
        d. Removes rows with missing 'imdbId' values.
        e. Removes duplicate rows based on 'imdbId', keeping only the first occurrence.
    4. Prints the row count after cleaning.

    Args:
        df (pandas.DataFrame): The DataFrame to be cleaned and analyzed.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    # Print row count before cleaning
    print(f"Row count before cleaning: {len(df)}")
    df.replace('\\N', np.nan, inplace=True)

    # Check if 'imdbId' exists in the DataFrame
    if 'imdbId' in df.columns:
        # Trim white spaces in 'imdbId' column
        df.loc[:, 'imdbId'] = df['imdbId'].str.strip()  # Use .loc to avoid SettingWithCopyWarning

        # Display duplicate imdbIds
        duplicate_imdb_ids = df[df.duplicated(subset=['imdbId'], keep=False)]
        if not duplicate_imdb_ids.empty:
            print("Duplicate imdbIds found:")
            print(duplicate_imdb_ids[['imdbId', 'title']])  # Display duplicate imdbIds with titles

        # Display rows with missing imdbId
        missing_imdb_ids = df[df['imdbId'].isna()]
        if not missing_imdb_ids.empty:
            print("Rows with missing imdbId found:")
            print(missing_imdb_ids[['imdbId', 'title']])  # Display rows with missing imdbIds

        # Remove rows with missing imdbId and duplicates
        df.dropna(subset=['imdbId'], inplace=True)  # Remove rows with missing imdbId
        df.drop_duplicates(subset=['imdbId'], keep='first', inplace=True)  # Keep only first occurrence of duplicates

    # Print row count after cleaning
    print(f"Row count after cleaning: {len(df)}")

    return df



# %% [markdown]
# Display a table with the column name, datatype, a random non-null value, and the number of missing values for each column

# %%
def summarize_columns(df):
    """
    Summarizes the columns of a DataFrame.
    This function generates a summary of each column in the given DataFrame, including the column name, data type, 
    a random non-null value, and the count of missing values.
    Parameters:
    df (pandas.DataFrame): The DataFrame to summarize.
    Returns:
    pandas.DataFrame: A DataFrame containing the summary of each column with the following columns:
        - 'Column Name': The name of the column.
        - 'Data Type': The data type of the column.
        - 'Random Non-Null Value': A random non-null value from the column.
        - 'Missing Values Count': The count of missing values in the column.
    """
    summary = []
    for column in df.columns:
        column_dtype = df[column].dtype
        
        # Find a random row with no null values
        non_null_rows = df.dropna()
        random_row = non_null_rows.sample(1).iloc[0]
        random_non_null_value = random_row[column]

        missing_values_count = df[column].isnull().sum()
        summary.append([column, column_dtype, random_non_null_value, missing_values_count])

    summary_df = pd.DataFrame(summary, columns=['Column Name', 'Data Type', 'Random Non-Null Value', 'Missing Values Count'])
    return summary_df


# %% [markdown]
# Print unique values in the dataset and number of them

# %%
def print_unique_values(df, column_name):
    """Prints all unique values in a specified column of a DataFrame.

    Args:
        df: The DataFrame to analyze.
        column_name: The name of the column to check.
    """
    unique_values = df[column_name].unique()
    print(f"Unique values in column '{column_name}':")
    print(unique_values)


# %% [markdown]
# Drop rows without US but first count and store the numbers

# %%
def availability_split(datasets, platforms):
    """
    Analyzes the availability of content in the US vs Non-US for each platform and visualizes the results.

    Args:
        datasets (list of pd.DataFrame): List of DataFrames for each streaming platform.
        platforms (list of str): List of platform names corresponding to the datasets.

    Returns:
        tuple: A tuple containing:
            - cleaned_datasets (dict): A dictionary with platform names as keys and cleaned DataFrames (US-only content) as values.
            - comparison_df (pd.DataFrame): A DataFrame with comparison data for each platform.
    """
    availability_data = []  # To store availability data for visualization
    cleaned_datasets = {}  # To store cleaned datasets
    comparison_data = []   # To store the comparison data for each platform
    
    for platform, df in zip(platforms, datasets):
        # Count rows with "US" and without "US" in 'availableCountries'
        us_rows = df[df['availableCountries'].str.contains('US', na=False)]
        non_us_rows = df[~df['availableCountries'].str.contains('US', na=False)]
        
        # Calculate percentages
        total_rows = len(df)
        us_count = len(us_rows)
        non_us_count = len(non_us_rows)
        us_percentage = (us_count / total_rows) * 100 if total_rows > 0 else 0
        non_us_percentage = (non_us_count / total_rows) * 100 if total_rows > 0 else 0
        
        # Store availability data for visualization
        availability_data.append({
            'platform': platform,
            'us_count': us_count,
            'us_percentage': us_percentage,
            'non_us_count': non_us_count,
            'non_us_percentage': non_us_percentage
        })
        
        # Store data for comparison_df
        comparison_data.append({
            'Platform': platform,
            'Titles in US': us_count,
            'Percentage in US': us_percentage,
            'Titles not in US': non_us_count,
            'Percent not in US': non_us_percentage
        })
        
        # Save the cleaned dataset for this platform (US-only content)
        cleaned_datasets[platform] = us_rows.reset_index(drop=True)
    
    # Create a DataFrame for availability data
    availability_df = pd.DataFrame(availability_data)
    
    # Create a DataFrame for comparison data
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create the plot with better styling
    plt.figure(figsize=(14, 8))  # Increase figure size for better readability
    
    # Plot stacked bar chart
    ax = availability_df.plot(kind='bar', x='platform', y=['us_percentage', 'non_us_percentage'], stacked=True,
                              color=['#4C8BF5', '#FF6F61'], width=0.75, edgecolor='white', legend=True)

    # Add titles and labels
    plt.title('Content Availability: U.S. vs Non-U.S. for Each Platform', fontsize=18, weight='bold', color='#333')
    plt.xlabel('Platform', fontsize=14, color='#555')
    plt.ylabel('Percentage of Titles', fontsize=14, color='#555')

    # Customize x-axis tick labels
    plt.xticks(rotation=45, ha='right', fontsize=12, color='#555')

    # Add data labels inside each section of the bars
    for p in ax.patches:
        # Get the height and position of each section
        height = p.get_height()
        width = p.get_width()
        x = p.get_x() + width / 2
        y = p.get_y() + height / 2
        
        # Annotate only the US and Non-US percentages inside the bars
        ax.annotate(f'{height:.1f}%', (x, y), ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    # Adjust the legend position inside the plot (in the upper-left corner)
    ax.legend(['US Content', 'Non-US Content'], loc='upper left', fontsize=12)

    # Adjust layout to make sure everything fits (including title and legend)
    plt.tight_layout()

    # Optionally remove gridlines or use more subtle ones
    ax.grid(False)  # Remove gridlines for a cleaner look

    # Save the plot as a PNG file
    plt.savefig(r'C:\Users\kimbe\Documents\StreamingAnalysis\outputs\us_availability.png', dpi=300)
    print("Saved content availability visualization as PNG.")
    plt.show()

    # Return the cleaned datasets and comparison_df
    return cleaned_datasets, comparison_df


# %% [markdown]
# Add a new column with the platform name and fill with 1

# %%
# Define the function to add the platform-specific column
def add_platform_column(df, platform_name):
    """
    Adds a new column to the DataFrame indicating the availability of the platform.

    Parameters:
    df (pandas.DataFrame): The DataFrame to which the platform column will be added.
    platform_name (str): The name of the platform to be added as a column.

    Returns:
    pandas.DataFrame: The DataFrame with the new platform column added, where all rows are marked as available (value 1).
    """
    df[platform_name] = 1  # Mark all rows as available in the platform
    return df


# %% [markdown]
# Merge platform datasets on imdbId and handle NaN for availability

# %%

# Function to merge platform datasets on imdbId and handle NaN for availability
def merge_platform_datasets(platforms, platform_dfs):
    """
    Merges multiple platform-specific datasets into a single consolidated DataFrame.
    Parameters:
    platforms (list of str): A list of platform names (e.g., ['Netflix', 'Hulu', 'AmazonPrime']).
    platform_dfs (list of pd.DataFrame): A list of DataFrames, each containing data for a specific platform.
    Returns:
    pd.DataFrame: A merged DataFrame with consolidated information from all platforms. The DataFrame includes:
        - 'imdbId': Unique identifier for each movie/TV show.
        - 'type': The type of content (e.g., movie, TV show).
        - 'genres': The genres associated with the content.
        - 'releaseYear': The release year of the content.
        - 'title': The title of the content.
        - 'imdbAverageRating': The average IMDb rating.
        - 'imdbNumVotes': The number of votes on IMDb.
        - Platform-specific columns (e.g., 'Netflix', 'Hulu', 'AmazonPrime'): Binary columns indicating availability on each platform (1 if available, 0 otherwise).
    Notes:
    - The function combines all input DataFrames into a single DataFrame.
    - It performs a groupby operation on 'imdbId' with custom aggregation logic to ensure that the first non-null value is taken for most columns.
    - Platform-specific columns are aggregated using the 'max' function to consolidate availability across datasets.
    - NaN values in platform-specific columns are replaced with 0, and the columns are converted to binary representation (0 or 1).
    """
    
    # Combine all datasets into a single dataframe
    all_platforms = pd.concat(platform_dfs, ignore_index=True)
    
    # Perform the groupby operation with custom aggregation logic
    merged_df = all_platforms.groupby('imdbId', as_index=False).agg({
        'type': 'first',        # Take the first non-null value
        'genres': 'first',      # Take the first non-null value
        'releaseYear': 'first', # Take the first non-null value
        'title': 'first',       # Take the first non-null value
        'imdbAverageRating': 'first',
        'imdbNumVotes': 'first',
        **{platform: 'max' for platform in platforms}  # Max will consolidate availability across datasets
    })
    
    # Replace NaN values in platform-specific columns with 0 and ensure binary representation (0 or 1)
    for platform in platforms:
        merged_df[platform] = merged_df[platform].fillna(0).astype(int)  # Modify directly and assign
    
    return merged_df


# %% [markdown]
# Count missing values with percentages

# %%
def count_missing_values(df):
    """
    Calculate the number and percentage of missing values in a DataFrame.
    Parameters:
    df (pandas.DataFrame): The DataFrame for which to calculate missing values.
    Returns:
    pandas.DataFrame: A summary DataFrame containing the count and percentage of missing values for each column.
        - 'Missing Values': The count of missing values in each column.
        - 'Percentage (%)': The percentage of missing values in each column, formatted as a string with a '%' sign.
    """
    # Calculate missing values and percentages
    missing_counts = df.isnull().sum()
    missing_percentage = (df.isnull().mean() * 100).round(2)
    
    # Create the summary DataFrame
    missing_summary = pd.DataFrame({
        'Missing Values': missing_counts,
        'Percentage (%)': missing_percentage.astype(str) + '%'
    })
    
    return missing_summary


# %% [markdown]
# # Combine Platforms

# %%
# List of platform DataFrames
platform_dfs = [amazon_df, hulu_df, netflix_df, hbo_df, apple_df]

# Clean and analyze each platform dataset
# This step involves removing whitespace, dropping rows without 'imdbId', and removing duplicate 'imdbId' values
cleaned_datasets = [clean_and_analyze_df(df) for df in platform_dfs]

# Reassign cleaned datasets back to their respective variables
amazon_df, hulu_df, netflix_df, hbo_df, apple_df = cleaned_datasets


# %%
# Check the shape of all platform DataFrames
# Sum the number of rows in all DataFrames within platform_dfs
total_rows = sum(len(dataset) for dataset in platform_dfs)

# Print the total number of rows across all raw datasets
print(f"Total number of rows across all raw datasets: {total_rows}")


# %% [markdown]
# ## Remove rows where not in US, count and save to csv

# %%
# Call the function
"""
Call the function to split the availability of datasets for different platforms.

Args:
    platform_dfs (dict): A dictionary containing dataframes for each platform.
    platforms (list): A list of platform names.

Returns:
    tuple: A tuple containing:
        - cleaned_datasets (dict): A dictionary of cleaned datasets for each platform.
        - comparison_df (DataFrame): A dataframe for comparing the availability across platforms.
"""
cleaned_datasets, comparison_df = availability_split(platform_dfs, platforms)



# %%
# Extract updated platforms and their corresponding DataFrames
platforms = list(cleaned_datasets.keys())  # Extract keys (platform names) from the dictionary
platform_dfs = list(cleaned_datasets.values())  # Extract values (U.S.-only DataFrames) from the dictionary

# Reassign updated DataFrames to original variables
for platform, df in zip(platforms, platform_dfs):
    if platform == "Amazon":
        amazon_df = df
    elif platform == "Hulu":
        hulu_df = df
    elif platform == "Netflix":
        netflix_df = df
    elif platform == "HBO":
        hbo_df = df
    elif platform == "Apple":
        apple_df = df

# Verify the update by printing the platform names and the number of rows in each updated DataFrame
print("Updated Platforms:", platforms)
print(f"Number of Updated DataFrames: {len(platform_dfs)}")
for platform, df in zip(platforms, platform_dfs):
    print(f"{platform}: {len(df)} rows")


# %%
# Check the shape of platform datasets
# Sum the number of rows in all DataFrames within platform_dfs
total_rows = sum(len(dataset) for dataset in platform_dfs)

# Print the total number of rows across all US datasets
print(f"Total number of rows across all US datasets: {total_rows}")


# %% [markdown]
# Add a new column with the platform name and fill with 1

# %%
# Apply the function for each platform dataset
for platform_name, df in zip(platforms, platform_dfs):
    # Get the index of the platform
    idx = platforms.index(platform_name)
    
    # Add a new column to the DataFrame indicating the availability of the platform
    platform_dfs[idx] = add_platform_column(df, platform_name)


# %%
# Check that it worked
summarize_columns(netflix_df)


# %% [markdown]
# ## Merge the platforms on imdbId

# %%
# Merge the datasets
"""
Merge the datasets from different platforms.

This function takes a list of platform names and a corresponding list of 
dataframes, then merges them into a single dataframe.

Args:
    platforms (list): A list of platform names.
    platform_dfs (list): A list of dataframes corresponding to each platform.

Returns:
    DataFrame: A merged dataframe containing data from all platforms.
"""
platform_merged = merge_platform_datasets(platforms, platform_dfs)


# %%
# Clean and analyze the merged platform dataset
# This step involves removing whitespace, dropping rows without 'imdbId', and removing duplicate 'imdbId' values
platform_merged = clean_and_analyze_df(platform_merged)

# Display the summary of the cleaned DataFrame
summarize_columns(platform_merged)


# %%
# Calculate the sum of each platform column
platform_sums = platform_merged[['Amazon', 'Hulu', 'Netflix', 'HBO', 'Apple']].sum()

# Print the sum for each platform
print(platform_sums)


# %% [markdown]
# Remove release year where outside of 1898 - 2024

# %%
# Remove rows with releaseYear outside the range 1898-2024
platform_merged = platform_merged[(platform_merged['releaseYear'] >= 1898) & 
                                  (platform_merged['releaseYear'] <= 2024)]


# %%
# Print the total number of rows across all US datasets after removing rows with incorrect years
print(f"Total number of rows across all US datasets after incorrect years removed: {platform_merged.shape[0]}")


# %% [markdown]
# Extract imdbIds

# %% [markdown]
# Merge the imbd info datasets

# %%
# Merge the basics_df and ratings_df datasets on 'tconst'
imdb_info = pd.merge(basics_df, ratings_df, on='tconst', how='inner')

# Display the shape of the resulting DataFrame
print("Raw imdb:", imdb_info.shape)

# Rename columns for consistency
imdb_info.rename(columns={"tconst": "imdbId", "primaryTitle": "title"}, inplace=True)


# %%
# Clean and analyze the IMDb information DataFrame
imdb_info = clean_and_analyze_df(imdb_info)

# Display a summary of the cleaned IMDb information DataFrame
summarize_columns(imdb_info)


# %%
# Display a summary of the platform_merged DataFrame
summarize_columns(platform_merged)


# %%
# Merge platform_merged with imdb_info on 'imdbId' using a left join
final_df = platform_merged.merge(imdb_info, on='imdbId', how='left')

# Check the shape of the resulting DataFrame
print("Merged DataFrame shape:", final_df.shape)



# %%
# Display a summary of the merged DataFrame
summarize_columns(final_df)


# %% [markdown]
# ### Investigate missing values

# %%
final_df = clean_and_analyze_df(final_df)


# %%
count_missing_values(final_df)


# %%

# Keep track of platform sums before cleaning
platform_sums_before = final_df[['Amazon', 'Hulu', 'Netflix', 'HBO', 'Apple']].sum()
print("Platform sums before cleaning:")
print(platform_sums_before)

# Drop unnecessary columns
columns_to_drop = ['originalTitle', 'endYear', 'isAdult', 'runtimeMinutes', 'genres_y']
final_df.drop(columns=columns_to_drop, inplace=True)

# Drop rows where genres_x is blank
final_df = final_df[final_df['genres_x'].notna() & (final_df['genres_x'] != "")]

# Drop rows where titleType is "tvEpisode"
final_df = final_df[final_df['titleType'] != "tvEpisode"]

# Map titleType to a more general type
title_type_to_type = {
    "short": "movie",
    "tvMiniSeries": "tv",
    "tvMovie": "movie",
    "tvSeries": "tv",
    "tvShort": "movie",
    "tvSpecial": "movie",
    "video": "movie"
}
final_df['type'] = final_df['titleType'].replace(title_type_to_type)

# Drop additional columns
columns_to_drop = ['startYear', 'titleType', 'title_y']
final_df.drop(columns=columns_to_drop, inplace=True)

# Drop rows where imdbAverageRating is blank or NaN
final_df = final_df[final_df['imdbAverageRating'].notna()]

# Drop additional columns
columns_to_drop = ['averageRating', 'numVotes']
final_df.drop(columns=columns_to_drop, inplace=True)

# Calculate sum of each platform after cleaning
platform_sums_after = final_df[['Amazon', 'Hulu', 'Netflix', 'HBO', 'Apple']].sum()

# Calculate the number of rows dropped for each platform
platform_dropped_counts = platform_sums_before - platform_sums_after

# Calculate the percentage of rows dropped for each platform
platform_dropped_percent = (platform_dropped_counts / platform_sums_before) * 100

# Convert to a DataFrame for easier plotting
platform_dropped_percent_df = platform_dropped_percent.reset_index()
platform_dropped_percent_df.columns = ['Platform', 'Percentage Dropped']

# Set Seaborn style
sns.set_theme(style="whitegrid", palette="pastel")

# Create a bar chart
plt.figure(figsize=(8, 6))
sns.barplot(
    x='Platform',
    y='Percentage Dropped',
    data=platform_dropped_percent_df,
    palette="Blues_d",
    hue='Platform'
)

# Add labels and title
plt.title('Percentage of Rows Dropped for Each Platform', fontsize=14)
plt.ylabel('Percentage Dropped (%)', fontsize=12)
plt.xlabel('Platform', fontsize=12)

# Show values on top of bars
for i, row in platform_dropped_percent_df.iterrows():
    plt.text(
        i, 
        row['Percentage Dropped'] + 0.5,  # Position above the bar
        f"{row['Percentage Dropped']:.1f}%",  # Format percentage
        ha='center', 
        fontsize=10
    )

# Show the plot
plt.tight_layout()
plt.show()


# %%
count_missing_values(final_df)


# %%
# Rename columns for consistency and clarity
final_df.rename(columns={'genres_x': 'genres'}, inplace=True)
final_df.rename(columns={'title_x': 'title'}, inplace=True)
final_df.rename(columns={'imdbAverageRating': 'rating'}, inplace=True)
final_df.rename(columns={'imdbNumVotes': 'numVotes'}, inplace=True)


# %%
# Dictionary to map IMDb IDs to their corrected types
imdb_id_type_correction = {
    "tt0101037": "movie",
    "tt14791494": "movie",
    "tt22297698": "movie",
    "tt22814400": "tv",
    "tt29634610": "movie",
    "tt30495748": "tv",
    "tt32132269": "tv",
    "tt32482847": "tv",
    "tt33356012": "tv",
    "tt4966036": "movie",
    "tt5556110": "movie",
    "tt8660842": "movie",
}

# Update the 'type' column in the dataset based on the mapping
final_df['type'] = final_df.apply(
    lambda row: imdb_id_type_correction[row['imdbId']] if row['imdbId'] in imdb_id_type_correction else row['type'], 
    axis=1
)


# %%
count_missing_values(final_df)


# %%
# Rename Columns for clarity
final_df.rename(columns={'genres_x': 'genres'}, inplace=True)
final_df.rename(columns={'title_x': 'title'}, inplace=True)
final_df.rename(columns={'imdbAverageRating': 'rating'}, inplace=True)
final_df.rename(columns={'imdbNumVotes': 'numVotes'}, inplace=True)


# %% [markdown]
# # Clean up final dataset

# %%
summarize_columns(final_df)


# %%
# Correct datatypes for 'releaseYear' and 'numVotes' columns
final_df['releaseYear'] = final_df['releaseYear'].astype(int)  
final_df['numVotes'] = final_df['numVotes'].astype(int)


# %%
# Rearrange columns in a more logical order
final_df = final_df[['imdbId', 'title', 'type', 'genres', 'releaseYear', 'rating', 'numVotes', 'Amazon', 'Hulu', 'Netflix', 'HBO', 'Apple']]


# %%
# Convert the genres column to lists
final_df['genres'] = final_df['genres'].str.split(',')

# Clean and sort each genre group to standardize representation
final_df['genres'] = final_df['genres'].apply(lambda x: [genre.strip() for genre in x] if isinstance(x, list) else [])

# Flatten the list of all genre groups into individual groups and count occurrences
all_genre_groups = final_df['genres'].apply(tuple)  # Convert to tuple for immutability
genre_group_counts = all_genre_groups.value_counts()

# Create a DataFrame with the unique genre groups and their counts
unique_genre_groups_df = pd.DataFrame({
    'Genre Group': ['/'.join(group) for group in genre_group_counts.index],
    'Count': genre_group_counts.values
})


# %%
# Define the file path
file_path = r"C:\Users\kimbe\Documents\StreamingAnalysis\data\cleaned_data\final_df.csv"

# Save the dataframe to a CSV file
final_df.to_csv(file_path, index=False)


# %% [markdown]
# One hot encode genres

# %%
# Define a dictionary to standardize genre names and mark some genres for removal
genre_replacements = {
    'Reality': 'Reality-TV', 
    'Adult': None,  
    'Sci-Fi': 'Science Fiction', 
    'Action & Adventure': None,
    'Kids': None,
    'Soap': None,
    'Film Noir': None
}

# Apply the genre replacements to the genres column
final_df['genres'] = final_df['genres'].apply(lambda x: [genre_replacements.get(genre, genre) for genre in x])

# Remove any genres that were marked for removal (i.e., 'None')
final_df['genres'] = final_df['genres'].apply(lambda x: [genre for genre in x if genre is not None])

# Flatten the genres column and get the count of each unique genre
all_genres = final_df['genres'].explode().value_counts()

# Calculate the percentage of each genre
total_rows = len(final_df)
genre_percentage = (all_genres / total_rows) * 100

# Calculate the total number of votes by genre
# First, flatten the genres and get the corresponding numVotes
genre_votes = final_df.explode('genres').groupby('genres')['numVotes'].sum()

# Calculate the percentage of numVotes by genre
vote_percentage = (genre_votes / genre_votes.sum()) * 100

# Combine the genre count and numVotes information into one DataFrame
genre_summary = pd.DataFrame({
    'count': all_genres,
    'percentage': genre_percentage,
    'numVotes': genre_votes,
    'vote_percentage': vote_percentage
}).reset_index()

# Rename columns for clarity
genre_summary.columns = ['genre', 'count', 'percentage', 'numVotes', 'vote_percentage']

# Display the result
print(genre_summary)


# %% [markdown]
# Ready to hot encode

# %%
# Perform one-hot encoding on the 'genres' column
one_hot_encoded = final_df['genres'].explode().str.get_dummies().groupby(level=0).sum()

# Merge the one-hot encoded columns back into the original DataFrame
final_df = final_df.join(one_hot_encoded)

# Display the updated DataFrame columns
final_df.columns


# %%
final_df.drop(['genres'], axis=1, inplace=True)


# %%
summarize_columns(final_df)


# %%
# Define the file paths
file_path = r"C:\Users\kimbe\Documents\StreamingAnalysis\data\cleaned_data\final_df.csv"
raw_catalog_path = r"C:\Users\kimbe\Documents\StreamingAnalysis\outputs\raw_catalog_data.csv"

# Save the dataframe to CSV files
final_df.to_csv(file_path, index=False)
final_df.to_csv(raw_catalog_path, index=False)

print(f"Data saved to {file_path}")
print(f"Data saved to {raw_catalog_path}")



