import pandas as pd

# Function to load datasets and print results
def load_and_print_data():
    # Path to the datasets
    base_path = r"C:\Users\kimbe\Documents\StreamingAnalysis\data\raw_data"

    # Try to load each dataset and print the success message with shape
    try:
        amazon_df = pd.read_csv(f"{base_path}\\amazon_catalog.csv")
        print(f"Dataframe amazon_df successfully loaded. {amazon_df.shape}")
    except Exception as e:
        print(f"Error: amazon_df not loaded. {e}")
    
    try:
        hulu_df = pd.read_csv(f"{base_path}\\hulu_catalog.csv")
        print(f"Dataframe hulu_df successfully loaded. {hulu_df.shape}")
    except Exception as e:
        print(f"Error: hulu_df not loaded. {e}")
    
    try:
        netflix_df = pd.read_csv(f"{base_path}\\netflix_catalog.csv")
        print(f"Dataframe netflix_df successfully loaded. {netflix_df.shape}")
    except Exception as e:
        print(f"Error: netflix_df not loaded. {e}")
    
    try:
        hbo_df = pd.read_csv(f"{base_path}\\hbo_catalog.csv")
        print(f"Dataframe hbo_df successfully loaded. {hbo_df.shape}")
    except Exception as e:
        print(f"Error: hbo_df not loaded. {e}")
    
    try:
        apple_df = pd.read_csv(f"{base_path}\\apple_catalog.csv")
        print(f"Dataframe apple_df successfully loaded. {apple_df.shape}")
    except Exception as e:
        print(f"Error: apple_df not loaded. {e}")
    
    try:
        basics_df = pd.read_csv(f"{base_path}\\imdb_basics.csv", low_memory=False)
        print(f"Dataframe basics_df successfully loaded. {basics_df.shape}")
    except Exception as e:
        print(f"Error: basics_df not loaded. {e}")
    
    try:
        ratings_df = pd.read_csv(f"{base_path}\\imdb_ratings.csv", low_memory=False)
        print(f"Dataframe ratings_df successfully loaded. {ratings_df.shape}")
    except Exception as e:
        print(f"Error: ratings_df not loaded. {e}")
