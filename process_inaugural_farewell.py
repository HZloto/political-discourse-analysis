"""
Political Discourse Analysis - Inaugural & Farewell Speeches
Author: 3DL - Data Driven Decision Lab
Website: https://datadrivendecisionlab.com

This module processes inaugural and farewell speeches into analyzable paragraphs.
Reads text files, extracts metadata, splits into ~500-word segments, and
exports to CSV format.
"""

import os
import pandas as pd
import re
from pathlib import Path

def extract_metadata_from_filename(filename, speech_type):
    """Extract year and president name from filename."""
    # Remove .txt extension
    name = filename.replace('.txt', '')
    
    # Handle different naming patterns
    # Farewell: YYYY-President-Farewell.txt
    # Inaugural: YYYY-President.txt
    parts = name.split('-')
    year = parts[0]
    
    if speech_type == 'farewell':
        # Remove 'Farewell' suffix
        president = '-'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
    else:
        # Inaugural speeches
        president = '-'.join(parts[1:]) if len(parts) > 1 else ''
    
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
        'Washington': 'None',
        'Adams': 'Federalist',
        'Jefferson': 'Democratic-Republican',
        'Madison': 'Democratic-Republican',
        'Monroe': 'Democratic-Republican',
        'VanBuren': 'Democratic',
        'Harrison': 'Whig',
        'Polk': 'Democratic',
        'Taylor': 'Whig',
        'Pierce': 'Democratic',
        'Buchanan': 'Democratic',
        'Lincoln': 'Republican',
        'Grant': 'Republican',
        'Hayes': 'Republican',
        'Garfield': 'Republican',
        'Cleveland': 'Democratic',
        'McKinley': 'Republican',
        'Roosevelt': 'Republican',
        'Taft': 'Republican',
        'Wilson': 'Democratic',
        'Harding': 'Republican',
        'Coolidge': 'Republican',
        'Hoover': 'Republican',
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
        'Biden': 'Democratic',
        'Jackson': 'Democratic'
    }
    return party_dict

def process_speeches(input_dir, output_file, speech_type):
    """Process all speeches of a given type and create rows for CSV."""
    
    # Initialize list to store all rows
    data_rows = []
    
    # Get all txt files in the directory
    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt') and f != 'README'])
    
    print(f"Found {len(txt_files)} {speech_type} files to process...")
    
    # Process each file
    for filename in txt_files:
        filepath = os.path.join(input_dir, filename)
        
        # Extract year and president
        year, president = extract_metadata_from_filename(filename, speech_type)
        
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
                'speech_type': speech_type,
                'paragraph': paragraph
            })
        
        print(f"Processed {filename}: {len(paragraphs)} paragraphs")
    
    return data_rows

def main():
    """Process both inaugural and farewell speeches and combine into single CSV."""
    
    # Set paths
    inaugural_dir = "sources/inaugural"
    farewell_dir = "sources/farewell"
    output_csv = "outputs/inaugural_farewell_paragraphs.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Process inaugural speeches
    print("\n" + "="*60)
    print("Processing Inaugural Speeches")
    print("="*60)
    inaugural_rows = process_speeches(inaugural_dir, output_csv, 'inaugural')
    
    # Process farewell speeches
    print("\n" + "="*60)
    print("Processing Farewell Speeches")
    print("="*60)
    farewell_rows = process_speeches(farewell_dir, output_csv, 'farewell')
    
    # Combine all rows
    all_rows = inaugural_rows + farewell_rows
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Get unique presidents and create party mapping
    unique_presidents = df['president'].unique()
    party_dict = create_party_mapping(unique_presidents)
    
    # Add party column
    df['party'] = df['president'].map(party_dict)
    
    # Reorder columns
    df = df[['year', 'president', 'party', 'speech_type', 'paragraph']]
    
    # Sort by year and speech type
    df = df.sort_values(['year', 'speech_type'], ascending=[True, False])
    
    # Save to CSV
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    # Calculate statistics
    df['word_count'] = df['paragraph'].apply(lambda x: len(x.split()))
    avg_words = df['word_count'].mean()
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total paragraphs: {len(df)}")
    print(f"  - Inaugural: {len(df[df['speech_type'] == 'inaugural'])}")
    print(f"  - Farewell: {len(df[df['speech_type'] == 'farewell'])}")
    print(f"Average words per paragraph: {avg_words:.1f}")
    print(f"Date range: {df['year'].min()} - {df['year'].max()}")
    print(f"Unique presidents: {len(unique_presidents)}")
    print(f"\nSpeech type breakdown:")
    print(df.groupby(['speech_type', 'party']).size().to_string())
    print(f"\nOutput saved to: {output_csv}")
    print(f"{'='*60}")
    
    return df

if __name__ == "__main__":
    df = main()
    
    # Display first few rows
    print("\nFirst 10 rows of the dataset:")
    print(df.head(10).to_string(max_colwidth=100))
