"""
Political Discourse Analysis - Data Visualization
Author: 3DL - Data Driven Decision Lab
Website: https://datadrivendecisionlab.com

Generate professional visualizations of hate/violence scores over time.
Creates publication-quality charts showing temporal trends in presidential rhetoric.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from datetime import datetime

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom color scheme
COLORS = {
    'Democratic': '#4A90E2',  # Professional blue
    'Republican': '#E84A5F',  # Professional red
    'bg_dem': '#E8F4FF',      # Very light blue for background
    'bg_rep': '#FFE8EC',      # Very light red for background
    'line': '#2C3E50',        # Dark gray for lines
    'grid': '#ECF0F1'         # Light gray for grid
}

def load_data():
    """Load and prepare the assessment data."""
    df = pd.read_csv('outputs/state_union_with_assessment.csv')
    
    # Convert year to numeric
    df['year'] = pd.to_numeric(df['year'])
    
    # Convert scores to numeric, treating NA and ERROR as NaN
    df['score_numeric'] = pd.to_numeric(df['hate_violence_score'], errors='coerce')
    
    # Create a unique identifier for each speech
    df['speech_id'] = df['year'].astype(str) + '-' + df['president']
    
    return df

def create_paragraph_level_chart(df, output_file='outputs/violence_scores_paragraph_level.png'):
    """
    Create a scatter plot showing all individual paragraph scores over time.
    """
    fig, ax = plt.subplots(figsize=(22, 11))
    
    # Add presidential term backgrounds
    add_party_backgrounds(ax, df)
    
    # Create jitter for better visibility
    np.random.seed(42)
    df_plot = df.copy()
    df_plot['year_jittered'] = df_plot['year'] + np.random.uniform(-0.3, 0.3, len(df_plot))
    df_plot['score_jittered'] = df_plot['score_numeric'] + np.random.uniform(-0.08, 0.08, len(df_plot))
    
    # Plot points with color by party - larger, more transparent
    for party in ['Democratic', 'Republican']:
        party_data = df_plot[df_plot['party'] == party]
        ax.scatter(party_data['year_jittered'], party_data['score_jittered'], 
                  alpha=0.35, s=80, c=COLORS[party], label=party,
                  edgecolors='none')
    
    # Add smoothed trend line with rolling average
    valid_data = df.groupby('year')['score_numeric'].mean().reset_index()
    valid_data = valid_data.sort_values('year')
    
    # Smooth with LOWESS
    from scipy.interpolate import make_interp_spline
    x_smooth = np.linspace(valid_data['year'].min(), valid_data['year'].max(), 500)
    spl = make_interp_spline(valid_data['year'], valid_data['score_numeric'], k=3)
    y_smooth = spl(x_smooth)
    
    ax.plot(x_smooth, y_smooth, 
            color='#1a1a1a', linewidth=4, alpha=0.85, 
            label='Trend (smoothed average)', zorder=100)
    
    # Add horizontal reference lines
    ax.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.4, zorder=0)
    ax.axhline(y=2, color='gray', linestyle=':', linewidth=1.5, alpha=0.4, zorder=0)
    ax.axhline(y=3, color='gray', linestyle=':', linewidth=1.5, alpha=0.4, zorder=0)
    ax.axhline(y=4, color='gray', linestyle=':', linewidth=1.5, alpha=0.4, zorder=0)
    
    # Add score level annotations on right side
    ax.text(2026, 0, '0: Ceremonial', fontsize=9, alpha=0.6, va='center')
    ax.text(2026, 1, '1: Policy discussion', fontsize=9, alpha=0.6, va='center')
    ax.text(2026, 2, '2: Us vs them', fontsize=9, alpha=0.6, va='center')
    ax.text(2026, 3, '3: Adversary ID', fontsize=9, alpha=0.6, va='center')
    ax.text(2026, 4, '4: Enemy framing', fontsize=9, alpha=0.6, va='center')
    ax.text(2026, 5, '5: Strong hostile', fontsize=9, alpha=0.6, va='center')
    
    # Styling
    ax.set_xlabel('Year', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel('Hate/Violence Score', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_title('State of the Union Rhetoric: Hate/Violence Assessment (1945-2024)\n' +
                 'Individual paragraphs colored by party • Trend shows yearly average',
                 fontsize=22, fontweight='bold', pad=25)
    
    # Set y-axis limits
    ax.set_ylim(-0.3, 5.5)
    ax.set_xlim(1943, 2028)
    
    # Grid - more prominent
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, axis='y')
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, axis='x')
    ax.set_axisbelow(True)
    
    # Better x-axis ticks
    ax.set_xticks(range(1945, 2025, 5))
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Legend - clearer
    ax.legend(loc='upper left', fontsize=14, frameon=True, shadow=True, 
             fancybox=True, framealpha=0.95, edgecolor='gray')
    
    # Add annotation
    ax.text(0.99, 0.01, 'Source: 3DL - Data Driven Decision Lab | datadrivendecisionlab.com',
            transform=ax.transAxes, fontsize=11, verticalalignment='bottom',
            horizontalalignment='right', alpha=0.6, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved: {output_file}')
    plt.close()

def create_speech_averaged_chart(df, output_file='outputs/violence_scores_speech_average.png'):
    """
    Create a chart showing average scores per State of the Union address.
    """
    # Calculate average score per speech
    speech_avg = df.groupby(['year', 'president', 'party']).agg({
        'score_numeric': 'mean',
        'speech_id': 'first'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(24, 10))
    
    # Add presidential term backgrounds
    add_party_backgrounds(ax, df)
    
    # Plot bars with color by party
    bar_width = 0.7
    for idx, row in speech_avg.iterrows():
        color = COLORS[row['party']]
        ax.bar(row['year'], row['score_numeric'], 
               width=bar_width, color=color, alpha=0.8,
               edgecolor='white', linewidth=1.5)
    
    # Add trend line
    z = np.polyfit(speech_avg['year'], speech_avg['score_numeric'], 2)
    p = np.poly1d(z)
    years_smooth = np.linspace(speech_avg['year'].min(), speech_avg['year'].max(), 300)
    ax.plot(years_smooth, p(years_smooth), 
            color=COLORS['line'], linewidth=3.5, alpha=0.8,
            linestyle='-', label='Trend', zorder=100)
    
    # Add president labels on x-axis
    add_president_labels(ax, df)
    
    # Styling
    ax.set_xlabel('', fontsize=14)
    ax.set_ylabel('Average Hate/Violence Score', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_title('State of the Union Rhetoric: Average Hate/Violence Scores by Address (1945-2024)\n' +
                 'Averaged across all paragraphs in each speech',
                 fontsize=20, fontweight='bold', pad=20)
    
    # Set y-axis limits
    ax.set_ylim(0, max(speech_avg['score_numeric']) * 1.2)
    ax.set_xlim(1944, 2025)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Legend
    dem_patch = mpatches.Patch(color=COLORS['Democratic'], label='Democratic', alpha=0.8)
    rep_patch = mpatches.Patch(color=COLORS['Republican'], label='Republican', alpha=0.8)
    trend_line = mpatches.Patch(color=COLORS['line'], label='Trend')
    ax.legend(handles=[dem_patch, rep_patch, trend_line], 
              loc='upper left', fontsize=12, frameon=True, shadow=True, fancybox=True)
    
    # Add annotation
    ax.text(0.99, 0.01, 'Source: 3DL - Data Driven Decision Lab | datadrivendecisionlab.com',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='right', alpha=0.7, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved: {output_file}')
    plt.close()

def add_party_backgrounds(ax, df):
    """Add subtle colored backgrounds for Democratic and Republican periods."""
    # Get the range of years for each president
    president_periods = df.groupby(['president', 'party']).agg({
        'year': ['min', 'max']
    }).reset_index()
    president_periods.columns = ['president', 'party', 'year_min', 'year_max']
    
    ymin, ymax = ax.get_ylim()
    
    for _, row in president_periods.iterrows():
        color = COLORS[f'bg_{row["party"].lower()[:3]}']
        rect = Rectangle((row['year_min'] - 0.5, ymin), 
                         row['year_max'] - row['year_min'] + 1, 
                         ymax - ymin,
                         facecolor=color, alpha=0.15, zorder=0, edgecolor='none')
        ax.add_patch(rect)

def add_president_labels(ax, df):
    """Add president names and years to x-axis."""
    # Get unique speeches with their info
    speeches = df.groupby(['year', 'president', 'party']).size().reset_index()
    speeches = speeches.sort_values('year')
    
    # Get transition points (where president changes)
    transitions = []
    prev_president = None
    for idx, row in speeches.iterrows():
        if row['president'] != prev_president:
            transitions.append({'year': row['year'], 'president': row['president']})
            prev_president = row['president']
    
    # Add president labels at transition points
    for trans in transitions:
        ax.text(trans['year'], ax.get_ylim()[0] * 0.95, trans['president'],
                rotation=45, ha='right', va='top', fontsize=10, fontweight='bold',
                alpha=0.7)
    
    # Set simple year ticks
    years = speeches['year'].unique()
    # Show every 4th year to avoid overcrowding
    year_ticks = [y for y in years if (int(y) - 1945) % 4 == 0 or y == years[-1]]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([int(y) for y in year_ticks], rotation=0, fontsize=10)

def create_distribution_chart(df, output_file='outputs/violence_scores_distribution.png'):
    """Create a distribution chart showing score frequencies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Overall distribution
    scores = df['score_numeric'].dropna()
    ax1.hist(scores, bins=np.arange(-0.5, 11.5, 1), 
             color=COLORS['line'], alpha=0.7, edgecolor='white', linewidth=2)
    ax1.set_xlabel('Hate/Violence Score', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Paragraphs', fontsize=14, fontweight='bold')
    ax1.set_title('Overall Score Distribution\n(All 936 Paragraphs)', 
                  fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Distribution by party
    dem_scores = df[df['party'] == 'Democratic']['score_numeric'].dropna()
    rep_scores = df[df['party'] == 'Republican']['score_numeric'].dropna()
    
    bins = np.arange(-0.5, 11.5, 1)
    ax2.hist([dem_scores, rep_scores], bins=bins, 
             color=[COLORS['Democratic'], COLORS['Republican']], 
             alpha=0.7, label=['Democratic', 'Republican'],
             edgecolor='white', linewidth=1.5)
    ax2.set_xlabel('Hate/Violence Score', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Paragraphs', fontsize=14, fontweight='bold')
    ax2.set_title('Score Distribution by Party', 
                  fontsize=16, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    
    plt.suptitle('Distribution of Hate/Violence Scores in State of the Union Addresses (1945-2024)',
                 fontsize=18, fontweight='bold', y=1.02)
    
    # Add annotation
    fig.text(0.99, 0.01, 'Source: 3DL - Data Driven Decision Lab | datadrivendecisionlab.com',
             fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', alpha=0.7, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved: {output_file}')
    plt.close()

def generate_summary_stats(df):
    """Print summary statistics."""
    print('\n' + '='*80)
    print('SUMMARY STATISTICS')
    print('='*80)
    
    valid_scores = df['score_numeric'].dropna()
    
    print(f'\nOverall Statistics:')
    print(f'  Mean score: {valid_scores.mean():.2f}')
    print(f'  Median score: {valid_scores.median():.2f}')
    print(f'  Std deviation: {valid_scores.std():.2f}')
    print(f'  Min score: {valid_scores.min():.0f}')
    print(f'  Max score: {valid_scores.max():.0f}')
    
    print(f'\nBy Party:')
    for party in ['Democratic', 'Republican']:
        party_scores = df[df['party'] == party]['score_numeric'].dropna()
        print(f'  {party}:')
        print(f'    Mean: {party_scores.mean():.2f}')
        print(f'    Median: {party_scores.median():.2f}')
        print(f'    Count: {len(party_scores)}')
    
    print(f'\nTop 5 Highest Scoring Speeches (by average):')
    speech_avg = df.groupby(['year', 'president', 'party'])['score_numeric'].mean().reset_index()
    speech_avg = speech_avg.sort_values('score_numeric', ascending=False).head(5)
    for _, row in speech_avg.iterrows():
        print(f'  {row["year"]} - {row["president"]} ({row["party"]}): {row["score_numeric"]:.2f}')
    
    print(f'\nTop 5 Lowest Scoring Speeches (by average):')
    speech_avg = df.groupby(['year', 'president', 'party'])['score_numeric'].mean().reset_index()
    speech_avg = speech_avg.sort_values('score_numeric', ascending=True).head(5)
    for _, row in speech_avg.iterrows():
        print(f'  {row["year"]} - {row["president"]} ({row["party"]}): {row["score_numeric"]:.2f}')

def main():
    """Generate all visualizations."""
    print('='*80)
    print('POLITICAL DISCOURSE ANALYSIS - VISUALIZATION GENERATOR')
    print('3DL - Data Driven Decision Lab')
    print('='*80)
    
    # Load data
    print('\nLoading data...')
    df = load_data()
    print(f'✓ Loaded {len(df)} paragraphs from {len(df["speech_id"].unique())} speeches')
    
    # Generate visualizations
    print('\nGenerating visualizations...')
    create_paragraph_level_chart(df)
    create_speech_averaged_chart(df)
    create_distribution_chart(df)
    
    # Generate statistics
    generate_summary_stats(df)
    
    print('\n' + '='*80)
    print('✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY')
    print('='*80)
    print('\nOutput files:')
    print('  📊 outputs/violence_scores_paragraph_level.png')
    print('  📊 outputs/violence_scores_speech_average.png')
    print('  📊 outputs/violence_scores_distribution.png')
    print()

if __name__ == "__main__":
    main()
