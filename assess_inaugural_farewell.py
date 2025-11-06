"""
Political Discourse Analysis - Inaugural & Farewell Speeches Assessment
Author: 3DL - Data Driven Decision Lab
Website: https://datadrivendecisionlab.com

This module scores paragraphs from inaugural and farewell speeches for hate/violent
speech severity using GPT-4o and the Rabat Plan of Action framework.
Implements async batch processing with retry logic and incremental saving.
"""

# pip install openai==1.* pandas python-dotenv tenacity aiolimiter
import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from typing import Any, Dict
from pathlib import Path
import time

# Load environment variables
load_dotenv()

# ------------- CONFIG -------------
MODEL = "gpt-4o"  # Use GPT-4o for best results
INPUT_CSV = "outputs/inaugural_farewell_paragraphs.csv"
OUTPUT_CSV = "outputs/inaugural_farewell_with_assessment.csv"
PROMPT_FILE = "prompts/inaugural_farewell_prompt.txt"
CONCURRENCY = 10                 # number of in-flight requests
RPS_LIMIT = 10                   # requests per second (tune to your org/project limits)
SAVE_INTERVAL = 50               # Save progress every N rows
# ------------- END CONFIG ----------

# Load the assessment prompt
with open(PROMPT_FILE, 'r') as f:
    ASSESSMENT_PROMPT = f.read()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
limiter = AsyncLimiter(max_rate=RPS_LIMIT, time_period=1)

class TransientOpenAIError(Exception):
    pass

def build_input(row: Dict[str, Any]) -> str:
    """
    Build the prompt for a paragraph with context.
    """
    year = row.get('year', 'Unknown')
    president = row.get('president', 'Unknown')
    party = row.get('party', 'Unknown')
    speech_type = row.get('speech_type', 'Unknown')
    paragraph = str(row.get('paragraph', '')).strip()
    
    # Customize setting description based on speech type
    if speech_type == 'inaugural':
        setting = "Presidential Inaugural Address (public ceremony, national broadcast, high influence)"
    elif speech_type == 'farewell':
        setting = "Presidential Farewell Address (public address, national broadcast, high influence)"
    else:
        setting = "Presidential Address (public, national broadcast, high influence)"
    
    context = f"""Context:
Year: {year}
Speaker: President {president} ({party} Party)
Speech Type: {speech_type.title()}
Setting: {setting}

Paragraph to assess:
{paragraph}

Please assess this paragraph following the instructions above."""
    
    return context

@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(TransientOpenAIError),
)
def parse_score(response: str) -> str:
    """
    Parse the model response and extract just the score.
    Returns the score as a string, or "ERROR" if parsing fails.
    """
    if not response:
        return "ERROR"
    
    # Clean up the response
    cleaned = response.strip()
    
    # Check if it's NA
    if cleaned.upper() in ["NA", "N/A"]:
        return "NA"
    
    # Try to extract just a number
    try:
        # Remove any non-numeric characters except decimal point
        import re
        number_match = re.search(r'\d+\.?\d*', cleaned)
        if number_match:
            score = float(number_match.group())
            # Validate it's in range 0-10
            if 0 <= score <= 10:
                return str(int(score)) if score == int(score) else str(score)
        return "ERROR"
    except:
        return "ERROR"

async def call_model(row: Dict[str, Any]) -> str:
    # Rate-limit guard
    async with limiter:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": ASSESSMENT_PROMPT},
                    {"role": "user", "content": build_input(row)},
                ],
                temperature=0.3,  # Lower temperature for more consistent assessments
            )
        except Exception as e:
            # Map retry-able errors to TransientOpenAIError; let others bubble up
            message = str(e).lower()
            if any(code in message for code in ["429", "timeout", "server error", "temporarily unavailable", "rate limit"]):
                raise TransientOpenAIError(e)
            raise

    # Extract text from Chat API response
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

async def worker(row_index: int, row: Dict[str, Any], total: int) -> Dict[str, Any]:
    """Process a single row and add the assessment score."""
    try:
        raw_response = await call_model(row)
        score = parse_score(raw_response)
        
        out = dict(row)
        out["hate_violence_score"] = score
        
        return out
    except Exception as e:
        print(f"Error processing row {row_index}: {e}")
        out = dict(row)
        out["hate_violence_score"] = "ERROR"
        return out

async def main():
    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    total_rows = len(df)
    print(f"Found {total_rows} paragraphs to process")
    print(f"  - Inaugural: {len(df[df['speech_type'] == 'inaugural'])}")
    print(f"  - Farewell: {len(df[df['speech_type'] == 'farewell'])}")
    print(f"Using model: {MODEL}")
    print(f"Concurrency: {CONCURRENCY}, Rate limit: {RPS_LIMIT} req/s")
    print(f"Starting assessment...\n")
    
    start_time = time.time()
    
    # Initialize results list
    results = []
    processed_count = 0
    error_count = 0
    
    # Limit concurrency explicitly with a semaphore
    sem = asyncio.Semaphore(CONCURRENCY)

    async def run_one(idx, r):
        async with sem:
            return await worker(idx, r, total_rows)

    # Process in batches to save incrementally
    batch_size = SAVE_INTERVAL
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_indices = range(batch_start, batch_end)
        
        print(f"Processing batch {batch_start + 1}-{batch_end}...")
        
        tasks = [run_one(i, df.iloc[i].to_dict()) for i in batch_indices]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process batch results
        for i, res in enumerate(batch_results):
            actual_idx = batch_start + i
            if isinstance(res, Exception):
                row = df.iloc[actual_idx].to_dict()
                row["hate_violence_score"] = "ERROR"
                results.append(row)
                error_count += 1
            else:
                results.append(res)
                if res.get("hate_violence_score") == "ERROR":
                    error_count += 1
            
            processed_count += 1
        
        # Save progress
        out_df = pd.DataFrame(results)
        out_df.to_csv(OUTPUT_CSV, index=False)
        
        elapsed = time.time() - start_time
        rate = processed_count / elapsed if elapsed > 0 else 0
        eta = (total_rows - processed_count) / rate if rate > 0 else 0
        
        print(f"Progress: {processed_count}/{total_rows} ({100*processed_count/total_rows:.1f}%) | "
              f"Rate: {rate:.1f} req/s | ETA: {eta/60:.1f} min | Errors: {error_count}")
        print(f"Saved checkpoint to {OUTPUT_CSV}\n")
    
    # Final save
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Total paragraphs processed: {total_rows}")
    print(f"Successful: {total_rows - error_count}")
    print(f"Errors: {error_count}")
    print(f"Average rate: {total_rows/elapsed:.1f} req/s")
    print(f"\nScore distribution by speech type:")
    for speech_type in ['inaugural', 'farewell']:
        type_df = out_df[out_df['speech_type'] == speech_type]
        print(f"\n{speech_type.title()}:")
        score_counts = type_df['hate_violence_score'].value_counts().sort_index()
        print(score_counts.to_string())
    print(f"\nOutput saved to: {OUTPUT_CSV}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
