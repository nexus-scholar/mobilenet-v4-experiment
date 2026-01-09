import pandas as pd
import re
import os

# Configuration
INPUT_CSV = "data/PlantWild/metadata.csv"  # The raw file
OUTPUT_CSV = "data/PlantWild/cleaned_metadata.csv" # The file your Dataset class will use

def clean_text(text):
    """
    Basic text cleaning to remove noise.
    """
    if pd.isna(text):
        return ""
    
    # 1. Convert to lowercase
    text = str(text).lower()
    
    # 2. Remove special characters (keep mostly alphanumeric and basic punctuation)
    # This removes things like HTML tags or weird encoding artifacts
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[^a-zA-Z0-9\s.,-]', '', text)
    
    # 3. Remove underscores often found in filenames/labels (e.g., apple_scab -> apple scab)
    text = text.replace('_', ' ')
    
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_clip_prompt(description, label_name):
    """
    Wraps the description in a natural language template for CLIP.
    Source 13/32: We want to capture semantics like 'yellow spots'.
    """
    # If the description is very short or missing, fallback to the label name
    # Ensure label_name is a string before checking length or using it
    label_name_str = str(label_name) if pd.notna(label_name) else ""
    
    if len(description) < 5: 
        content = label_name_str
    else:
        content = description
        
    # TEMPLATE: This is the "Prompt Engineering" part.
    # It tells CLIP exactly what it is looking at.
    return f"A photo of a plant leaf showing symptoms of {content}"

def process_metadata():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}. Make sure you unzipped the data!")
        return

    print(f"Reading raw data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # INSPECT YOUR COLUMNS FIRST!
    # The real CSV might use different column names like 'caption', 'text', 'disease_name'
    # Update these variables based on the real header:
    TEXT_COL = 'description'  # Change this to whatever the real column is named
    LABEL_NAME_COL = 'label' # Changed from 'disease_name' to 'label' based on dummy data schema
    
    # Check if columns exist
    if TEXT_COL not in df.columns:
        print(f"Warning: Column '{TEXT_COL}' not found. Available columns: {df.columns}")
        # Fallback logic could go here
        return

    print("Cleaning text descriptions...")
    
    # Apply cleaning
    df['clean_description'] = df[TEXT_COL].apply(clean_text)
    
    # Apply CLIP templating
    # We use both the specific visual description and the class name to ensure context
    df['clip_text'] = df.apply(
        lambda row: create_clip_prompt(row['clean_description'], row.get(LABEL_NAME_COL, "")), 
        axis=1
    )
    
    # Save the new version
    # We keep the old columns just in case, but 'clip_text' is what the model will read.
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("------------------------------------------------")
    print(f"Success! Processed {len(df)} rows.")
    print(f"Saved to: {OUTPUT_CSV}")
    print("------------------------------------------------")
    print("Sample Output (What CLIP will see):")
    print(df[['clip_text']].head(5))

if __name__ == "__main__":
    process_metadata()

