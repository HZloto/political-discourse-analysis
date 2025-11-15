"""
Political Discourse Analysis - Inaugural & Farewell Speeches Visualization
Author: 3DL - Data Driven Decision Lab
Website: https://datadrivendecisionlab.com

Generate professional visualizations comparing hate/violence scores between
inaugural and farewell addresses. Creates publication-quality charts showing
temporal trends, speech type comparisons, and party differences.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy.interpolate import make_interp_spline

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom color scheme
COLORS = {
    'Democratic': '#4A90E2',
    'Republican': '#E84A5F',
    'Federalist': '#9B59B6',
    'Democratic-Republican': '#E67E22',
    'Whig': '#16A085',
    'None': '#95A5A6',
    'inaugural': '#3498DB',
    'farewell': '#E74C3C',
    'bg_dem': '#E8F4FF',
    'bg_rep': '#FFE8EC',
    'line': '#2C3E50',
    'grid': '#ECF0F1'
}

def load_data():
    """Load and prepare the assessment data."""
    df = pd.read_csv('outputs/inaugural_farewell_with_assessment.csv')
    
    # Convert year to numeric
    df['year'] = pd.to_numeric(df['year'])
    
    # Convert scores to numeric, treating NA and ERROR as NaN
    df['score_numeric'] = pd.to_numeric(df['hate_violence_score'], errors='coerce')
    
    # Create a unique identifier for each speech
    df['speech_id'] = df['year'].astype(str) + '-' + df['president'] + '-' + df['speech_type']
    
    return df

def create_speech_type_comparison(df, output_file='outputs/inaugural_farewell_comparison.png'):
    """
    Create a chart comparing inaugural vs farewell addresses over time.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 16), sharex=True)
    
    # Calculate average score per speech
    speech_avg = df.groupby(['year', 'president', 'party', 'speech_type']).agg({
        'score_numeric': 'mean',
        'speech_id': 'first'
    }).reset_index()
    
    # Split by speech type
    inaugural_data = speech_avg[speech_avg['speech_type'] == 'inaugural'].sort_values('year')
    farewell_data = speech_avg[speech_avg['speech_type'] == 'farewell'].sort_values('year')
    
    # Top panel: Inaugural addresses
    for idx, row in inaugural_data.iterrows():
        party_color = COLORS.get(row['party'], COLORS['None'])
        ax1.bar(row['year'], row['score_numeric'], 
               width=0.8, color=party_color, alpha=0.75,
               edgecolor='white', linewidth=1.2)
    
    # Add trend line for inaugurals
    if len(inaugural_data) > 3:
        x_smooth = np.linspace(inaugural_data['year'].min(), inaugural_data['year'].max(), 300)
        spl = make_interp_spline(inaugural_data['year'], inaugural_data['score_numeric'], k=3)
        y_smooth = spl(x_smooth)
        ax1.plot(x_smooth, y_smooth, color='#1a1a1a', linewidth=3.5, alpha=0.8, label='Trend', zorder=100)
    
    ax1.set_ylabel('Average Hate/Violence Score', fontsize=16, fontweight='bold', labelpad=15)
    ax1.set_title('Inaugural Addresses (1789-2025)', fontsize=18, fontweight='bold', pad=15)
    ax1.set_ylim(0, max(speech_avg['score_numeric']) * 1.15)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax1.set_axisbelow(True)
    
    # Bottom panel: Farewell addresses
    for idx, row in farewell_data.iterrows():
        party_color = COLORS.get(row['party'], COLORS['None'])
        ax2.bar(row['year'], row['score_numeric'], 
               width=0.8, color=party_color, alpha=0.75,
               edgecolor='white', linewidth=1.2)
    
    # Add trend line for farewells
    if len(farewell_data) > 3:
        x_smooth = np.linspace(farewell_data['year'].min(), farewell_data['year'].max(), 300)
        spl = make_interp_spline(farewell_data['year'], farewell_data['score_numeric'], k=3)
        y_smooth = spl(x_smooth)
        ax2.plot(x_smooth, y_smooth, color='#1a1a1a', linewidth=3.5, alpha=0.8, label='Trend', zorder=100)
    
    ax2.set_xlabel('Year', fontsize=16, fontweight='bold', labelpad=15)
    ax2.set_ylabel('Average Hate/Violence Score', fontsize=16, fontweight='bold', labelpad=15)
    ax2.set_title('Farewell Addresses (1796-2025)', fontsize=18, fontweight='bold', pad=15)
    ax2.set_ylim(0, max(speech_avg['score_numeric']) * 1.15)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax2.set_axisbelow(True)
    
    # Create unified legend
    legend_patches = []
    for party in ['Democratic', 'Republican', 'Democratic-Republican', 'Federalist', 'Whig', 'None']:
        if party in speech_avg['party'].values:
            legend_patches.append(mpatches.Patch(color=COLORS[party], label=party, alpha=0.75))
    legend_patches.append(mpatches.Patch(color='#1a1a1a', label='Trend'))
    
    ax1.legend(handles=legend_patches, loc='upper left', fontsize=12, 
              frameon=True, shadow=True, fancybox=True, ncol=2)
    
    plt.suptitle('Presidential Rhetoric: Inaugural vs Farewell Addresses\nHate/Violence Assessment Over Time',
                 fontsize=22, fontweight='bold', y=0.995)
    
    # Add annotation
    fig.text(0.99, 0.01, 'Source: 3DL - Data Driven Decision Lab | datadrivendecisionlab.com',
             fontsize=11, verticalalignment='bottom',
             horizontalalignment='right', alpha=0.6, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved: {output_file}')
    plt.close()

def create_combined_timeline(df, output_file='outputs/inaugural_farewell_timeline.png'):
    """
    Create a single timeline showing both inaugural and farewell addresses.
    """
    fig, ax = plt.subplots(figsize=(24, 12))
    
    # Calculate average score per speech
    speech_avg = df.groupby(['year', 'president', 'party', 'speech_type']).agg({
        'score_numeric': 'mean',
        'speech_id': 'first'
    }).reset_index()
    
    # Plot inaugural addresses
    inaugural_data = speech_avg[speech_avg['speech_type'] == 'inaugural']
    for idx, row in inaugural_data.iterrows():
        party_color = COLORS.get(row['party'], COLORS['None'])
        ax.scatter(row['year'], row['score_numeric'], 
                  s=250, color=party_color, alpha=0.7,
                  marker='o', edgecolors='white', linewidth=2,
                  label='Inaugural' if idx == inaugural_data.index[0] else '')
    
    # Plot farewell addresses
    farewell_data = speech_avg[speech_avg['speech_type'] == 'farewell']
    for idx, row in farewell_data.iterrows():
        party_color = COLORS.get(row['party'], COLORS['None'])
        ax.scatter(row['year'], row['score_numeric'], 
                  s=250, color=party_color, alpha=0.7,
                  marker='s', edgecolors='white', linewidth=2,
                  label='Farewell' if idx == farewell_data.index[0] else '')
    
    # Add separate polynomial trend lines for smoother visualization
    for speech_type, marker, color in [('inaugural', 'o', COLORS['inaugural']), 
                                        ('farewell', 's', COLORS['farewell'])]:
        type_data = speech_avg[speech_avg['speech_type'] == speech_type].sort_values('year')
        if len(type_data) > 3:
            # Use polynomial fit for smoother trend line
            z = np.polyfit(type_data['year'], type_data['score_numeric'], 3)
            p = np.poly1d(z)
            x_smooth = np.linspace(type_data['year'].min(), type_data['year'].max(), 300)
            y_smooth = p(x_smooth)
            ax.plot(x_smooth, y_smooth, color=color, linewidth=3, alpha=0.6, 
                   linestyle='--', label=f'{speech_type.title()} Trend')
    
    # Styling
    ax.set_xlabel('Year', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel('Average Hate/Violence Score', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_title('Presidential Rhetoric Timeline: Inaugural vs Farewell Addresses (1789-2025)\n' +
                 'Circles = Inaugural | Squares = Farewell',
                 fontsize=22, fontweight='bold', pad=25)
    
    ax.set_ylim(0, max(speech_avg['score_numeric']) * 1.15)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.6, axis='y')
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.4, axis='x')
    ax.set_axisbelow(True)
    
    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=12, label='Inaugural Address', markeredgecolor='white', markeredgewidth=2),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                   markersize=12, label='Farewell Address', markeredgecolor='white', markeredgewidth=2),
        plt.Line2D([0], [0], color=COLORS['inaugural'], linewidth=3, 
                   linestyle='--', label='Inaugural Trend'),
        plt.Line2D([0], [0], color=COLORS['farewell'], linewidth=3, 
                   linestyle='--', label='Farewell Trend')
    ]
    
    # Add party colors to legend
    for party in ['Democratic', 'Republican', 'Democratic-Republican', 'Federalist', 'Whig', 'None']:
        if party in speech_avg['party'].values:
            legend_elements.append(mpatches.Patch(color=COLORS[party], label=party, alpha=0.7))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=13, 
             frameon=True, shadow=True, fancybox=True, ncol=2)
    
    # Add annotation
    ax.text(0.99, 0.01, 'Source: 3DL - Data Driven Decision Lab | datadrivendecisionlab.com',
            transform=ax.transAxes, fontsize=11, verticalalignment='bottom',
            horizontalalignment='right', alpha=0.6, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved: {output_file}')
    plt.close()

def create_distribution_comparison(df, output_file='outputs/inaugural_farewell_distribution.png'):
    """Create distribution charts comparing inaugural and farewell addresses."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    inaugural_scores = df[df['speech_type'] == 'inaugural']['score_numeric'].dropna()
    farewell_scores = df[df['speech_type'] == 'farewell']['score_numeric'].dropna()
    
    bins = np.arange(-0.5, 11.5, 1)
    
    # Top left: Overall distribution comparison
    ax1.hist([inaugural_scores, farewell_scores], bins=bins, 
             color=[COLORS['inaugural'], COLORS['farewell']], 
             alpha=0.7, label=['Inaugural', 'Farewell'],
             edgecolor='white', linewidth=1.5)
    ax1.set_xlabel('Hate/Violence Score', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Paragraphs', fontsize=14, fontweight='bold')
    ax1.set_title('Overall Score Distribution\nInaugural vs Farewell', 
                  fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Top right: Box plot comparison
    data_to_plot = [inaugural_scores, farewell_scores]
    bp = ax2.boxplot(data_to_plot, labels=['Inaugural', 'Farewell'],
                     patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], [COLORS['inaugural'], COLORS['farewell']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Hate/Violence Score', fontsize=14, fontweight='bold')
    ax2.set_title('Score Distribution (Box Plot)', fontsize=16, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    
    # Bottom left: Inaugural by party
    inaugural_df = df[df['speech_type'] == 'inaugural']
    parties = ['Democratic', 'Republican', 'Democratic-Republican']
    inaugural_by_party = [inaugural_df[inaugural_df['party'] == p]['score_numeric'].dropna() 
                          for p in parties if p in inaugural_df['party'].values]
    party_labels = [p for p in parties if p in inaugural_df['party'].values]
    
    if inaugural_by_party:
        ax3.hist(inaugural_by_party, bins=bins, 
                color=[COLORS[p] for p in party_labels], 
                alpha=0.7, label=party_labels,
                edgecolor='white', linewidth=1.5)
        ax3.set_xlabel('Hate/Violence Score', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Paragraphs', fontsize=14, fontweight='bold')
        ax3.set_title('Inaugural Addresses by Party', fontsize=16, fontweight='bold', pad=15)
        ax3.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_axisbelow(True)
    
    # Bottom right: Farewell by party
    farewell_df = df[df['speech_type'] == 'farewell']
    farewell_by_party = [farewell_df[farewell_df['party'] == p]['score_numeric'].dropna() 
                         for p in parties if p in farewell_df['party'].values]
    party_labels_farewell = [p for p in parties if p in farewell_df['party'].values]
    
    if farewell_by_party:
        ax4.hist(farewell_by_party, bins=bins, 
                color=[COLORS[p] for p in party_labels_farewell], 
                alpha=0.7, label=party_labels_farewell,
                edgecolor='white', linewidth=1.5)
        ax4.set_xlabel('Hate/Violence Score', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Number of Paragraphs', fontsize=14, fontweight='bold')
        ax4.set_title('Farewell Addresses by Party', fontsize=16, fontweight='bold', pad=15)
        ax4.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_axisbelow(True)
    
    plt.suptitle('Distribution Analysis: Inaugural vs Farewell Addresses',
                 fontsize=20, fontweight='bold', y=0.995)
    
    # Add annotation
    fig.text(0.99, 0.01, 'Source: 3DL - Data Driven Decision Lab | datadrivendecisionlab.com',
             fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', alpha=0.7, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved: {output_file}')
    plt.close()

def create_president_comparison(df, output_file='outputs/inaugural_farewell_by_president.png'):
    """
    Create a chart comparing presidents who gave both inaugural and farewell addresses.
    """
    # Find presidents with both types of speeches
    speech_counts = df.groupby(['president', 'speech_type']).size().unstack(fill_value=0)
    presidents_with_both = speech_counts[(speech_counts['inaugural'] > 0) & 
                                          (speech_counts['farewell'] > 0)].index
    
    # Calculate average scores for these presidents
    comparison_data = []
    for president in presidents_with_both:
        pres_data = df[df['president'] == president]
        inaugural_avg = pres_data[pres_data['speech_type'] == 'inaugural']['score_numeric'].mean()
        farewell_avg = pres_data[pres_data['speech_type'] == 'farewell']['score_numeric'].mean()
        year = pres_data['year'].max()
        party = pres_data['party'].iloc[0]
        
        if pd.notna(inaugural_avg) and pd.notna(farewell_avg):
            comparison_data.append({
                'president': president,
                'year': year,
                'party': party,
                'inaugural': inaugural_avg,
                'farewell': farewell_avg,
                'difference': farewell_avg - inaugural_avg
            })
    
    if not comparison_data:
        print("⚠ Not enough data for president comparison chart")
        return
    
    comparison_df = pd.DataFrame(comparison_data).sort_values('year')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14))
    
    # Top panel: Side-by-side comparison
    x = np.arange(len(comparison_df))
    width = 0.35
    
    for i, row in comparison_df.iterrows():
        idx = list(comparison_df.index).index(i)
        party_color = COLORS.get(row['party'], COLORS['None'])
        
        ax1.bar(idx - width/2, row['inaugural'], width, 
               color=party_color, alpha=0.6, edgecolor='white', linewidth=1.5)
        ax1.bar(idx + width/2, row['farewell'], width, 
               color=party_color, alpha=0.9, edgecolor='white', linewidth=1.5)
    
    # Add custom legend elements
    inaugural_bar = mpatches.Patch(facecolor='gray', alpha=0.6, label='Inaugural', edgecolor='white', linewidth=1.5)
    farewell_bar = mpatches.Patch(facecolor='gray', alpha=0.9, label='Farewell', edgecolor='white', linewidth=1.5)
    ax1.legend(handles=[inaugural_bar, farewell_bar], loc='upper left', fontsize=12, 
              frameon=True, shadow=True)
    
    ax1.set_ylabel('Average Hate/Violence Score', fontsize=14, fontweight='bold')
    ax1.set_title('Presidents with Both Inaugural and Farewell Addresses\nComparing Rhetoric Intensity',
                  fontsize=18, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{row['president']}\n{int(row['year'])}" 
                         for _, row in comparison_df.iterrows()], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Bottom panel: Difference (farewell - inaugural)
    colors = [COLORS.get(row['party'], COLORS['None']) for _, row in comparison_df.iterrows()]
    bars = ax2.bar(x, comparison_df['difference'], color=colors, alpha=0.8,
                   edgecolor='white', linewidth=1.5)
    
    # Add zero line
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    ax2.set_ylabel('Score Difference (Farewell - Inaugural)', fontsize=14, fontweight='bold')
    ax2.set_title('Rhetorical Shift from Inaugural to Farewell\nPositive = More Intense in Farewell',
                  fontsize=18, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{row['president']}\n{int(row['year'])}" 
                         for _, row in comparison_df.iterrows()], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    
    # Add annotation
    fig.text(0.99, 0.01, 'Source: 3DL - Data Driven Decision Lab | datadrivendecisionlab.com',
             fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', alpha=0.7, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved: {output_file}')
    plt.close()

def create_era_analysis(df, output_file='outputs/inaugural_farewell_by_era.png'):
    """
    Analyze rhetoric by historical era.
    """
    # Define eras
    df['era'] = pd.cut(df['year'], 
                       bins=[1789, 1860, 1920, 1980, 2025],
                       labels=['Founding to Civil War\n(1789-1860)', 
                              'Industrial Era\n(1861-1920)',
                              'Modern Era\n(1921-1980)', 
                              'Contemporary\n(1981-2025)'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left panel: Average by era and speech type
    era_speech_avg = df.groupby(['era', 'speech_type'])['score_numeric'].mean().unstack()
    
    x = np.arange(len(era_speech_avg))
    width = 0.35
    
    if 'inaugural' in era_speech_avg.columns:
        ax1.bar(x - width/2, era_speech_avg['inaugural'], width, 
               label='Inaugural', color=COLORS['inaugural'], alpha=0.8,
               edgecolor='white', linewidth=1.5)
    
    if 'farewell' in era_speech_avg.columns:
        ax1.bar(x + width/2, era_speech_avg['farewell'], width, 
               label='Farewell', color=COLORS['farewell'], alpha=0.8,
               edgecolor='white', linewidth=1.5)
    
    ax1.set_ylabel('Average Hate/Violence Score', fontsize=14, fontweight='bold')
    ax1.set_title('Rhetoric Intensity by Historical Era', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(era_speech_avg.index, fontsize=12)
    ax1.legend(fontsize=12, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Right panel: Box plots by era
    inaugural_by_era = [df[(df['era'] == era) & (df['speech_type'] == 'inaugural')]['score_numeric'].dropna()
                        for era in df['era'].cat.categories]
    farewell_by_era = [df[(df['era'] == era) & (df['speech_type'] == 'farewell')]['score_numeric'].dropna()
                       for era in df['era'].cat.categories]
    
    positions_inaugural = np.arange(len(df['era'].cat.categories)) * 2 - 0.3
    positions_farewell = np.arange(len(df['era'].cat.categories)) * 2 + 0.3
    
    bp1 = ax2.boxplot(inaugural_by_era, positions=positions_inaugural, widths=0.5,
                      patch_artist=True, labels=[''] * len(inaugural_by_era))
    bp2 = ax2.boxplot(farewell_by_era, positions=positions_farewell, widths=0.5,
                      patch_artist=True, labels=[''] * len(farewell_by_era))
    
    for patch in bp1['boxes']:
        patch.set_facecolor(COLORS['inaugural'])
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor(COLORS['farewell'])
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Hate/Violence Score', fontsize=14, fontweight='bold')
    ax2.set_title('Score Distribution by Era', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xticks(np.arange(len(df['era'].cat.categories)) * 2)
    ax2.set_xticklabels(df['era'].cat.categories, fontsize=11)
    ax2.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Inaugural', 'Farewell'],
              loc='upper right', fontsize=12, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    
    plt.suptitle('Historical Era Analysis: Inaugural vs Farewell Addresses',
                 fontsize=20, fontweight='bold', y=0.98)
    
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
    print('SUMMARY STATISTICS - INAUGURAL & FAREWELL ADDRESSES')
    print('='*80)
    
    valid_scores = df['score_numeric'].dropna()
    
    print(f'\nOverall Statistics:')
    print(f'  Total paragraphs: {len(df)}')
    print(f'  Valid scores: {len(valid_scores)}')
    print(f'  Mean score: {valid_scores.mean():.2f}')
    print(f'  Median score: {valid_scores.median():.2f}')
    print(f'  Std deviation: {valid_scores.std():.2f}')
    print(f'  Min score: {valid_scores.min():.0f}')
    print(f'  Max score: {valid_scores.max():.0f}')
    
    print(f'\nBy Speech Type:')
    for speech_type in ['inaugural', 'farewell']:
        type_scores = df[df['speech_type'] == speech_type]['score_numeric'].dropna()
        print(f'  {speech_type.title()}:')
        print(f'    Mean: {type_scores.mean():.2f}')
        print(f'    Median: {type_scores.median():.2f}')
        print(f'    Count: {len(type_scores)}')
    
    print(f'\nBy Party (Democratic & Republican only):')
    for party in ['Democratic', 'Republican']:
        party_scores = df[df['party'] == party]['score_numeric'].dropna()
        if len(party_scores) > 0:
            print(f'  {party}:')
            print(f'    Mean: {party_scores.mean():.2f}')
            print(f'    Median: {party_scores.median():.2f}')
            print(f'    Count: {len(party_scores)}')
    
    print(f'\nTop 5 Highest Scoring Speeches (by average):')
    speech_avg = df.groupby(['year', 'president', 'party', 'speech_type'])['score_numeric'].mean().reset_index()
    speech_avg = speech_avg.sort_values('score_numeric', ascending=False).head(5)
    for _, row in speech_avg.iterrows():
        print(f'  {row["year"]} - {row["president"]} ({row["party"]}) - {row["speech_type"].title()}: {row["score_numeric"]:.2f}')
    
    print(f'\nTop 5 Lowest Scoring Speeches (by average):')
    speech_avg = df.groupby(['year', 'president', 'party', 'speech_type'])['score_numeric'].mean().reset_index()
    speech_avg = speech_avg.sort_values('score_numeric', ascending=True).head(5)
    for _, row in speech_avg.iterrows():
        print(f'  {row["year"]} - {row["president"]} ({row["party"]}) - {row["speech_type"].title()}: {row["score_numeric"]:.2f}')

def main():
    """Generate all visualizations."""
    print('='*80)
    print('INAUGURAL & FAREWELL ADDRESSES - VISUALIZATION GENERATOR')
    print('3DL - Data Driven Decision Lab')
    print('='*80)
    
    # Load data
    print('\nLoading data...')
    df = load_data()
    print(f'✓ Loaded {len(df)} paragraphs from {len(df["speech_id"].unique())} speeches')
    print(f'  - Inaugural: {len(df[df["speech_type"] == "inaugural"])} paragraphs')
    print(f'  - Farewell: {len(df[df["speech_type"] == "farewell"])} paragraphs')
    
    # Generate visualizations
    print('\nGenerating visualizations...')
    create_speech_type_comparison(df)
    create_combined_timeline(df)
    create_distribution_comparison(df)
    create_president_comparison(df)
    create_era_analysis(df)
    
    # Generate statistics
    generate_summary_stats(df)
    
    print('\n' + '='*80)
    print('✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY')
    print('='*80)
    print('\nOutput files:')
    print('  📊 outputs/inaugural_farewell_comparison.png')
    print('  📊 outputs/inaugural_farewell_timeline.png')
    print('  📊 outputs/inaugural_farewell_distribution.png')
    print('  📊 outputs/inaugural_farewell_by_president.png')
    print('  📊 outputs/inaugural_farewell_by_era.png')
    print()

if __name__ == "__main__":
    main()
