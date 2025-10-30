import os
import pandas as pd
import re
from pathlib import Path

def extract_year_president(filename):
    """Extract year and president name from filename."""
    # Remove .txt extension
    name = filename.replace('.txt', '')
    
    # Handle special cases with -1, -2 suffixes
    parts = name.split('-')
    year = parts[0]
    president = '-'.join(parts[1:]) if len(parts) > 1 else ''
    
    # Clean up president name (remove -1, -2 suffixes)
    president = re.sub(r'-\d+$', '', president)
    
    return year, president

def split_into_sentences(text):
    """Split text into sentences using basic sentence boundaries."""
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    
    # Basic sentence splitting on period, exclamation, question mark
    # followed by space and capital letter or end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Clean up sentences - remove extra whitespace
    sentences = [' '.join(s.split()).strip() for s in sentences if s.strip()]
    
    return sentences

def split_into_paragraphs(text, target_words=500):
    """Split text into paragraphs of approximately target_words, cutting at sentence boundaries."""
    # First split into sentences
    sentences = split_into_sentences(text)
    
    paragraphs = []
    current_paragraph = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        # If adding this sentence would exceed target, save current paragraph and start new one
        if current_word_count > 0 and current_word_count + sentence_word_count > target_words:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = [sentence]
            current_word_count = sentence_word_count
        else:
            current_paragraph.append(sentence)
            current_word_count += sentence_word_count
    
    # Add the last paragraph if it has content
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return paragraphs

def create_party_mapping(presidents):
    """Create a dictionary mapping presidents to their political parties."""
    party_dict = {
        'Truman': 'Democratic',
        'Eisenhower': 'Republican',
        'Kennedy': 'Democratic',
        'Johnson': 'Democratic',
        'Nixon': 'Republican',
        'Ford': 'Republican',
        'Carter': 'Democratic',
        'Reagan': 'Republican',
        'Bush': 'Republican',
        'Clinton': 'Democratic',
        'GWBush': 'Republican',
        'Obama': 'Democratic',
        'Trump': 'Republican',
        'Biden': 'Democratic'
    }
    return party_dict

def process_state_union_speeches(input_dir, output_file):
    """Process all State of the Union speeches and create a CSV."""
    
    # Initialize list to store all rows
    data_rows = []
    
    # Get all txt files in the directory
    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt') and f != 'README'])
    
    print(f"Found {len(txt_files)} speech files to process...")
    
    # First pass: collect all data
    for filename in txt_files:
        filepath = os.path.join(input_dir, filename)
        
        # Extract year and president
        year, president = extract_year_president(filename)
        
        # Read the file with fallback encoding
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with latin-1 encoding as fallback
            with open(filepath, 'r', encoding='latin-1') as f:
                text = f.read()
        
        # Split into paragraphs of ~500 words
        paragraphs = split_into_paragraphs(text, target_words=500)
        
        # Add each paragraph as a row
        for paragraph in paragraphs:
            data_rows.append({
                'year': year,
                'president': president,
                'paragraph': paragraph
            })
        
        print(f"Processed {filename}: {len(paragraphs)} paragraphs")
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Get unique presidents and create party mapping
    unique_presidents = df['president'].unique()
    party_dict = create_party_mapping(unique_presidents)
    
    # Add party column
    df['party'] = df['president'].map(party_dict)
    
    # Reorder columns
    df = df[['year', 'president', 'party', 'paragraph']]
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Calculate average word count
    df['word_count'] = df['paragraph'].apply(lambda x: len(x.split()))
    avg_words = df['word_count'].mean()
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total paragraphs: {len(df)}")
    print(f"Average words per paragraph: {avg_words:.1f}")
    print(f"Date range: {df['year'].min()} - {df['year'].max()}")
    print(f"Presidents: {len(unique_presidents)}")
    print(f"\nPresident breakdown:")
    print(df.groupby(['president', 'party']).size().to_string())
    print(f"\nOutput saved to: {output_file}")
    print(f"{'='*60}")
    
    return df

if __name__ == "__main__":
    # Set paths
    input_directory = "sources/state_union"
    output_csv = "outputs/state_union_paragraphs.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Process the speeches
    df = process_state_union_speeches(input_directory, output_csv)
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head(5).to_string(max_colwidth=100))
