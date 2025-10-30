# Political Discourse Analysis
## Analyzing Hate Speech and Violent Rhetoric in State of the Union Addresses

**Part of the [3DL - Data Driven Decision Lab](https://datadrivendecisionlab.com) Research Initiative**

---

## Overview

This research project analyzes hate speech and violent rhetoric in U.S. State of the Union addresses from 1945 to 2024 using natural language processing and GPT-4 assessments based on the Rabat Plan of Action framework.

The project processes 80 historical State of the Union speeches, segments them into analyzable paragraphs, and scores each paragraph for hate/violent speech severity using an evidence-based framework derived from international human rights standards.

## Dataset

- **Source**: State of the Union addresses (1945-2024)
- **Presidents**: Truman, Eisenhower, Kennedy, Johnson, Nixon, Ford, Carter, Reagan, Bush, Clinton, G.W. Bush, Obama, Trump, Biden
- **Total speeches**: 80
- **Total paragraphs analyzed**: 936
- **Average paragraph length**: ~468 words

## Methodology

### Assessment Framework

We use the **Rabat Plan of Action** six-part framework for assessing incitement to hatred:

1. **Context**: Political, social, and historical context of the statement
2. **Speaker**: Status, influence, and authority of the speaker
3. **Intent**: Evidence of intention to incite hatred, discrimination, or violence
4. **Content/Form**: Nature and style of speech (dehumanization, stereotyping, calls to action)
5. **Extent**: Reach and magnitude of the speech act (public/private, media coverage)
6. **Likelihood of Harm**: Imminence and probability of real-world consequences

### Calibrated Scoring Scale (0-10)

The scale has been calibrated specifically for State of the Union addresses to capture nuances in political rhetoric:

- **0**: Pure platitudes, ceremonial language (e.g., "Thank you for gathering tonight")
- **1**: General policy discussion without targeting groups
- **2**: Mild us-vs-them framing, vague external threats
- **3**: Clear in-group/out-group division, identifying adversaries
- **4**: Explicit enemy identification, threat framing of specific groups
- **5**: Strong adversarial language, dehumanizing metaphors
- **6**: Explicit dehumanization, militaristic language toward groups
- **7**: Strong incitement language, calls for hostile action
- **8**: Severe dehumanization, explicit threat construction
- **9**: Direct incitement to hostility/violence
- **10**: Explicit calls for genocide or mass violence

**Note**: The scale is calibrated so that 0 is rare (only ceremonial statements). Most policy discussion scores 1-2, with scores of 3+ indicating increasingly adversarial rhetoric.

### Technical Implementation

- **Model**: OpenAI GPT-4o
- **Processing**: Asynchronous batch processing with rate limiting (10 req/s)
- **Validation**: Strict output format enforcement (numeric scores only)
- **Reliability**: Automatic retry logic with exponential backoff
- **Incremental saving**: Progress checkpoints every 50 paragraphs
- **Visualization**: Professional matplotlib/seaborn charts with party-colored backgrounds

## Project Structure

```
political-discourse-analysis/
├── sources/
│   └── state_union/          # Original speech text files (80 files)
├── outputs/                   # Generated datasets and visualizations
│   ├── state_union_paragraphs.csv           # Processed paragraphs
│   ├── state_union_with_assessment.csv      # Paragraphs with scores
│   ├── violence_scores_paragraph_level.png  # Scatter plot visualization
│   ├── violence_scores_speech_average.png   # Bar chart by speech
│   └── violence_scores_distribution.png     # Score distribution charts
├── prompts/
│   └── assessment_prompt.txt  # LLM assessment instructions
├── process_speeches.py        # Convert speeches to paragraphs
├── assess_speeches.py         # Score paragraphs using GPT-4o
├── generate_visuals.py        # Create professional visualizations
├── run_pipeline.py            # Complete pipeline automation
├── .env                       # API credentials (not in git)
├── .env.example               # Template for environment variables
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key with GPT-4o access

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/HZloto/political-discourse-analysis.git
   cd political-discourse-analysis
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Dependencies include**:
   - `openai` - GPT-4o API access
   - `pandas` - Data processing
   - `python-dotenv` - Environment variables
   - `tenacity` - Retry logic
   - `aiolimiter` - Rate limiting
   - `matplotlib` - Visualization
   - `seaborn` - Statistical graphics
   - `scipy` - Scientific computing
   - `numpy` - Numerical operations

4. **Configure API key**
   
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Quick Start: Run Complete Pipeline

```bash
python run_pipeline.py
```

This will automatically:
1. Process all speeches into ~500-word paragraphs
2. Assess hate/violence scores for each paragraph using GPT-4o
3. Generate professional visualizations
4. Save all outputs to `outputs/` directory

**Runtime**: ~2-3 minutes for 936 paragraphs

To skip speech processing (if already done):
```bash
python run_pipeline.py --skip-processing
```

### Manual Step-by-Step Execution

### Step 1: Process Speeches into Paragraphs

```bash
python process_speeches.py
```

**What it does:**
- Reads all `.txt` files from `sources/state_union/`
- Extracts year and president from filenames
- Splits each speech into ~500-word paragraphs (cut at sentence boundaries)
- Maps presidents to political parties
- Outputs: `outputs/state_union_paragraphs.csv`

**Output columns:**
- `year`: Year of the speech
- `president`: President's last name
- `party`: Political party (Democratic/Republican)
- `paragraph`: ~500-word text segment

### Step 2: Assess Hate/Violence Scores

```bash
python assess_speeches.py
```

**What it does:**
- Loads `outputs/state_union_paragraphs.csv`
- Sends each paragraph to GPT-4o with Rabat framework prompt
- Receives numeric score (0-10) or "NA"
- Saves progress every 50 rows
- Outputs: `outputs/state_union_with_assessment.csv`

**Runtime**: Approximately 1.5-2 hours for 936 paragraphs at 10 req/s

**Output columns:**
- All columns from input CSV
- `hate_violence_score`: Numeric score (0-10), "NA", or "ERROR"

### Configuration Options

Edit `assess_speeches.py` to customize:

```python
MODEL = "gpt-4o"              # OpenAI model
CONCURRENCY = 10              # Parallel requests
RPS_LIMIT = 10                # Requests per second
SAVE_INTERVAL = 50            # Checkpoint frequency
```

## Output Data

### `state_union_paragraphs.csv`
Processed speeches with metadata (936 rows × 4 columns)

### `state_union_with_assessment.csv`
Scored paragraphs with hate/violence assessment (936 rows × 5 columns)

Sample row:
```csv
year,president,party,paragraph,hate_violence_score
2023,Biden,Democratic,"My fellow Americans, we meet tonight...",1
```

### Visualizations

1. **`violence_scores_paragraph_level.png`** - Scatter plot showing all 936 individual paragraph scores over time with party-colored backgrounds and trend line

2. **`violence_scores_speech_average.png`** - Bar chart showing average scores per State of the Union address with president labels

3. **`violence_scores_distribution.png`** - Histogram showing overall score distribution and comparison by political party

## Research Applications

This dataset enables analysis of:

- Temporal trends in political rhetoric (1945-2024)
- Partisan differences in speech patterns
- Historical context of political discourse
- Correlation with social/political events
- Comparative analysis across presidencies

## Limitations

- Automated scoring may not capture all contextual nuances
- Framework optimized for explicit hate speech detection
- Historical speeches require contextual interpretation
- Model outputs are probabilistic, not definitive judgments

## Citation

If you use this dataset or methodology in your research, please cite:

```
Political Discourse Analysis: Hate Speech Assessment in State of the Union Addresses (1945-2024)
3DL - Data Driven Decision Lab
https://datadrivendecisionlab.com
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Contact

**3DL - Data Driven Decision Lab**  
Website: [datadrivendecisionlab.com](https://datadrivendecisionlab.com)

## Acknowledgments

- Rabat Plan of Action framework by the United Nations
- State of the Union archive sources
- OpenAI GPT-4o for assessment capabilities

---

**Research Initiative**: This project is part of the 3DL Data Driven Decision Lab's ongoing research into computational social science and political discourse analysis.
