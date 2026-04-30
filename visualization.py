"""
Combined Analysis & Validation Pipeline
========================================
Inputs:
  - Classified _terms.csv  (main dataset / predictions)
  - GT_file.xlsx                                          (ground-truth labels)

Outputs (all written to OUTPUT_DIR):
  Figure_1_Clinical_Landscape_and_Onset.pdf
  Figure_2_Evidence_Quality.pdf
  Figure_3_Tier_and_QoL_Overview.pdf
  Figure_4_QoL_Impact_Analysis.pdf
  Figure_5_Management_Drivers.pdf
  Figure_6_Distribution_and_Efficiency.pdf
  comprehensive_analysis_report_COMPLETE.txt
  comprehensive_dashboard.pdf
  comprehensive_validation_report.txt
"""

# =====================================================================
#  IMPORTS
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats
from scipy.stats import (
    spearmanr, pearsonr, chi2_contingency,
    kruskal, mannwhitneyu, kendalltau, linregress
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    cohen_kappa_score, mean_absolute_error,
    matthews_corrcoef, balanced_accuracy_score,
    f1_score, accuracy_score, precision_score, recall_score
)
import json
import warnings
import os
import tempfile
import shutil
from pathlib import Path

warnings.filterwarnings('ignore')


# =====================================================================
#  FILE PATHS  (auto-detects Code Ocean vs. local paths)
# =====================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir("/data") and os.path.isdir("/results"):
    DATA_DIR = "/data"
    RESULTS_DIR = "/results"
else:
    DATA_DIR = os.path.join(HERE, "data")
    RESULTS_DIR = os.path.join(HERE, "results")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

# Predicted file: prefer the no-space filename; fall back to the legacy
# filename with a space for backward compatibility.
_pred_clean = os.path.join(DATA_DIR, "Classified_terms.csv")
_pred_legacy = os.path.join(DATA_DIR, "Classified _terms.csv")
PREDICTED_FILE = _pred_clean if os.path.exists(_pred_clean) else _pred_legacy

GT_FILE = os.path.join(DATA_DIR, "GT_file.xlsx")

OUTPUT_DIR = os.path.join(RESULTS_DIR, "visualization")


# =====================================================================
#  SECTION 1 : SHARED STYLING
# =====================================================================
def setup_plot_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlepad'] = 12
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['ytick.minor.width'] = 0.5
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['ytick.minor.size'] = 2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    nature_colors = [
        '#0072B2', '#DF8F44', '#00A1D5',
        '#B24745', '#79AF97', '#6A6599', '#80796B',
    ]
    sns.set_palette(nature_colors)
    return nature_colors


# =====================================================================
#  SECTION 2 : STATISTICAL UTILITIES
# =====================================================================
def format_pvalue(p_val):
    if p_val < 0.001:
        return 'p < 0.001'
    return f'p = {p_val:.3f}'


def calculate_chisquare(contingency_table):
    try:
        chi2, p_val, dof, _ = chi2_contingency(contingency_table)
        return f"chi2({dof}) = {chi2:.2f}, {format_pvalue(p_val)}", chi2, p_val, dof
    except Exception as e:
        print(f"Warning: chi-square failed - {e}")
        return "", None, None, None


def calculate_kruskal(df, group_col, value_col):
    try:
        groups = [g[value_col].dropna() for _, g in df.groupby(group_col)]
        if len(groups) < 2:
            return "", None, None, None
        dof = len(groups) - 1
        h_stat, p_val = kruskal(*groups)
        return f"H({dof}) = {h_stat:.2f}, {format_pvalue(p_val)}", h_stat, p_val, dof
    except Exception as e:
        print(f"Warning: Kruskal-Wallis failed - {e}")
        return "", None, None, None


def calculate_regression_stats(x, y):
    try:
        slope, intercept, r_value, p_value, _ = linregress(x, y)
        return f"R2 = {r_value**2:.3f}, {format_pvalue(p_value)}", r_value, p_value
    except Exception as e:
        print(f"Warning: regression failed - {e}")
        return "", None, None


def calculate_correlation(x, y, method='pearson'):
    try:
        if method == 'pearson':
            corr, p_val = pearsonr(x, y)
            name = 'r'
        else:
            corr, p_val = spearmanr(x, y)
            name = 'rho'
        return f"{name} = {corr:.2f}, {format_pvalue(p_val)}", corr, p_val
    except Exception as e:
        print(f"Warning: correlation failed - {e}")
        return "", None, None


# =====================================================================
#  SECTION 3 : DATA LOADING & PREPROCESSING
# =====================================================================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        return df
    except FileNotFoundError:
        print(f"ERROR: file not found at {file_path}")
        return pd.DataFrame()


def reclassify_clinical_signs_as_nfc(df):
    df['original_classification_category'] = df['classification_category'].copy()
    df['original_classification_tier'] = df['classification_tier'].copy()
    mask = df['classification_category'] == 'Clinical Sign or Laboratory Abnormality'
    df.loc[mask, 'classification_category'] = 'NFC - Clinical Signs'
    df.loc[mask, 'classification_tier'] = 'NFC'
    print(f"Reclassified {mask.sum()} rows to 'NFC - Clinical Signs'")
    return df


def preprocess_data(df):
    print("Preprocessing data...")
    df.columns = df.columns.str.strip()

    if 'severity_assessment' in df.columns and 'QoL_assessment' not in df.columns:
        df.rename(columns={'severity_assessment': 'QoL_assessment'}, inplace=True)
    if 'QoL_assessment' in df.columns:
        df['QoL_assessment'] = df['QoL_assessment'].replace('NFC', 'Not Affected')
        df['QoL_assessment'] = df['QoL_assessment'].str.title()

    def extract_mgmt_categories(text):
        if pd.isna(text):
            return []
        try:
            data = json.loads(text)
            categories = data.get("management_category", [])
            clean_cats = set()
            for cat in categories:
                if 'Surgical' in cat:
                    clean_cats.add('Surgical')
                elif 'Pharmacological' in cat:
                    clean_cats.add('Pharmacological')
                elif 'Therapeutic Support' in cat:
                    clean_cats.add('Therapeutic Support')
                elif 'Dietary' in cat:
                    clean_cats.add('Dietary')
                elif 'Assistive' in cat or 'Devices' in cat:
                    clean_cats.add('Assistive Devices')
                elif 'Monitoring' in cat or 'Surveillance' in cat:
                    clean_cats.add('Monitoring')
                elif 'Genetic' in cat:
                    clean_cats.add('Genetic Counseling')
                elif 'Radiation' in cat:
                    clean_cats.add('Radiation Therapy')
                elif 'Conservative' in cat:
                    clean_cats.add('Conservative Management')
                elif 'Palliative' in cat:
                    clean_cats.add('Palliative')
                elif 'No Treatment' in cat:
                    clean_cats.add('No Treatment')
                elif 'Information not available' not in cat:
                    clean_cats.add(cat)
            return list(clean_cats)
        except (json.JSONDecodeError, TypeError):
            return []

    df['management_categories'] = df['management_profile'].apply(extract_mgmt_categories)
    df['num_interventions'] = df['management_categories'].apply(len)

    def parse_verification_log(log_string):
        if pd.isna(log_string):
            return {}
        try:
            log_data = json.loads(log_string)
            for item in log_data:
                if isinstance(item, dict) and 'total_claims' in item:
                    return item
        except (json.JSONDecodeError, TypeError):
            pass
        return {}

    verification_data = df['verification_log'].apply(parse_verification_log).apply(pd.Series)
    for col in ['total_claims', 'direct_support', 'valid_inferences',
                'weak_inferences', 'unsupported', 'weighted_score']:
        if col in verification_data.columns:
            df[f'verify_{col}'] = pd.to_numeric(
                verification_data[col], errors='coerce').fillna(0)
        else:
            df[f'verify_{col}'] = 0

    df['direct_evidence_ratio'] = np.where(
        df['verify_total_claims'] > 0,
        df['verify_direct_support'] / df['verify_total_claims'], np.nan)
    df['unsupported_ratio'] = np.where(
        df['verify_total_claims'] > 0,
        df['verify_unsupported'] / df['verify_total_claims'], np.nan)

    qol_domain_mapping = {
        'Physical':  'Physical Functioning',
        'Cognitive': 'Cognitive/School',
        'Symptoms':  'Physical Symptoms',
        'Emotional': 'Emotional Impact',
        'Social':    'Social Functioning',
    }

    def parse_acog_criteria(criteria_str):
        if pd.isna(criteria_str):
            return {'qol': [], 'early_onset': False}
        criteria = str(criteria_str).split(', ')
        result = {'qol': set(), 'early_onset': False}
        for c in criteria:
            if 'Early Onset' in c:
                result['early_onset'] = True
            elif 'Physical Functioning' in c:
                result['qol'].add(qol_domain_mapping['Physical'])
            elif 'Cognitive' in c or 'School' in c:
                result['qol'].add(qol_domain_mapping['Cognitive'])
            elif 'Physical Symptoms' in c:
                result['qol'].add(qol_domain_mapping['Symptoms'])
            elif 'Emotional' in c:
                result['qol'].add(qol_domain_mapping['Emotional'])
            elif 'Social' in c:
                result['qol'].add(qol_domain_mapping['Social'])
        return {k: list(v) if isinstance(v, set) else v for k, v in result.items()}

    acog_parsed = df['acog_criteria_met_names'].apply(parse_acog_criteria)
    df['qol_domains']     = acog_parsed.apply(lambda x: x['qol'])
    df['has_early_onset'] = acog_parsed.apply(lambda x: x['early_onset'])
    df['num_qol_domains'] = df['qol_domains'].apply(len)

    df['clinical_impact_score'] = (
        (df['QoL_assessment'] == 'Affected').astype(int) * 0.40
        + df['has_early_onset'].astype(int) * 0.25
        + np.clip(df['num_qol_domains'] / 5, 0, 1) * 0.20
        + np.clip(df['num_interventions'] / 5, 0, 1) * 0.15
    )

    df['review_status'] = df['review_status'].str.strip()
    print("Preprocessing complete.")
    return df


# =====================================================================
#  SECTION 4 : ANALYSIS FIGURES 1-6
# =====================================================================

def create_clinical_landscape_panel(df, colors, output_dir):
    fig = plt.figure(figsize=(18, 14), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.2], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    category_tier = pd.crosstab(df['classification_category'], df['classification_tier'])
    category_tier['total'] = category_tier.sum(axis=1)
    category_tier.sort_values('total', ascending=False, inplace=True)
    category_tier_pct = (category_tier.drop('total', axis=1)
                         .div(category_tier['total'], axis=0) * 100)
    category_tier_pct.plot(kind='barh', stacked=True, ax=ax1, colormap='viridis', width=0.8)
    for i, tick in enumerate(ax1.get_yticklabels()):
        total = category_tier.loc[tick.get_text(), 'total']
        ax1.text(102, i, f'N={total}', va='center', ha='left', fontsize=8)
    ax1.set_xlim(0, max(115, ax1.get_xlim()[1]))
    ax1.set_xlabel('Percentage of Phenotypes (%)')
    ax1.set_ylabel('Classification Category')
    ax1.set_title('A. Tier Distribution & Volume Within Each Category',
                  loc='left', fontsize=12)
    ax1.legend(title='Tier', frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')

    df_onset = df[~df['classification_category'].str.contains(
        'Out of Scope|NFC', na=False, case=False)]
    prevalence = df_onset.groupby('classification_category')['has_early_onset'].mean() * 100
    prevalence.sort_values(ascending=True).plot(kind='barh', ax=ax2, color=colors[4])
    ax2.set_xlabel('Phenotypes with Early Onset (%)')
    ax2.set_ylabel('')
    contingency = pd.crosstab(df_onset['classification_category'], df_onset['has_early_onset'])
    chi_str, *_ = calculate_chisquare(contingency)
    ax2.set_title(f'B. Early Onset Prevalence by Category ({chi_str})',
                  loc='left', fontsize=12)

    df_cat = df[~df['classification_category'].str.contains('NFC|Out of Scope', na=False)]
    df_exp_cat = df_cat.explode('management_categories')
    df_exp_cat = df_exp_cat[df_exp_cat['management_categories'] != 'Genetic Counseling']
    if not df_exp_cat['management_categories'].dropna().empty:
        cat_mgmt = pd.crosstab(df_exp_cat['classification_category'],
                               df_exp_cat['management_categories'])
        cat_mgmt_pct = cat_mgmt.div(cat_mgmt.sum(axis=1), axis=0) * 100
        cat_mgmt_pct.plot(kind='barh', stacked=True, ax=ax3, colormap='tab20', width=0.8)
    ax3.set_title('C. Management Profile Across Categories', loc='left', fontsize=12)
    ax3.set_xlabel('Proportion of Management Types (%)')
    ax3.set_ylabel('Classification Category')
    ax3.legend(title='Management Type', frameon=False,
               bbox_to_anchor=(1.02, 1), loc='upper left')

    df_tier = df[df['classification_tier'].astype(str) != 'NFC']
    df_exp_tier = df_tier.explode('management_categories')
    df_exp_tier = df_exp_tier[df_exp_tier['management_categories'] != 'Genetic Counseling']
    tier_mgmt = pd.crosstab(df_exp_tier['classification_tier'],
                            df_exp_tier['management_categories'])
    tier_mgmt_pct = tier_mgmt.div(tier_mgmt.sum(axis=1), axis=0) * 100
    tier_mgmt_pct.plot(kind='bar', stacked=True, ax=ax4, colormap='tab20', width=0.7)
    ax4.set_title('D. Management Profile Across Tiers', loc='left', fontsize=12)
    ax4.set_ylabel('Proportion of Management Types (%)')
    ax4.set_xlabel('Classification Tier')
    ax4.legend(title='Management Type', frameon=False,
               bbox_to_anchor=(1.01, 1), loc='upper left')
    ax4.tick_params(axis='x', rotation=0)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

    fig.savefig(os.path.join(output_dir, 'Figure_1_Clinical_Landscape_and_Onset.pdf'),
                dpi=600, bbox_inches='tight')
    plt.show()
    print("Figure 1 saved.")


def create_evidence_quality_panel(df, colors, output_dir):
    df_f = df[~df['classification_tier'].astype(str).str.contains('NFC', na=False)
              & ~df['classification_category'].str.contains('NFC', na=False)]

    fig = plt.figure(figsize=(16, 15), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ev_cols = ['verify_direct_support', 'verify_valid_inferences',
               'verify_weak_inferences', 'verify_unsupported']
    cmap = plt.get_cmap('cividis')

    ax1 = fig.add_subplot(gs[0, 0])
    ev_tier = df_f.groupby('classification_tier')[ev_cols].sum()
    ev_tier_pct = ev_tier.div(ev_tier.sum(axis=1), axis=0) * 100
    ev_tier_pct.plot(kind='barh', stacked=True, ax=ax1, colormap=cmap, legend=False)
    ax1.set_xlabel('Proportion of Evidence Claims (%)')
    ax1.set_ylabel('Classification Tier')
    ax1.set_title('A. Evidence Profile by Tier', loc='left', fontsize=12)
    for n, (index, row) in enumerate(ev_tier_pct.iterrows()):
        cum = 0
        for i, (col, w) in enumerate(row.items()):
            if w > 5:
                bar_color = cmap(i / len(row))
                lum = 0.299*bar_color[0] + 0.587*bar_color[1] + 0.114*bar_color[2]
                tc = 'white' if lum < 0.4 else 'black'
                ax1.text(cum + w/2, n, f'{w:.0f}%',
                         ha='center', va='center', color=tc, fontsize=8, weight='bold')
            cum += w

    ax2 = fig.add_subplot(gs[0, 1])
    ev_cat = df_f.groupby('classification_category')[ev_cols].sum()
    ev_cat_pct = ev_cat.div(ev_cat.sum(axis=1), axis=0) * 100
    ev_cat_pct.sort_values('verify_direct_support').plot(
        kind='barh', stacked=True, ax=ax2, colormap=cmap, legend=False)
    ax2.set_xlabel('Proportion of Evidence Claims (%)')
    ax2.set_ylabel('Classification Category')
    ax2.set_title('B. Evidence Profile by Category', loc='left', fontsize=12)

    handles, _ = ax1.get_legend_handles_labels()
    fig.legend(handles, ['Direct', 'Valid Infer', 'Weak Infer', 'Unsupported'],
               bbox_to_anchor=(0.5, 1.0), loc='lower center',
               ncol=4, frameon=False, title="Evidence Type", fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    sns.violinplot(x='classification_tier', y='statement_support_score',
                   data=df_f, ax=ax3, palette='viridis', inner='quartile',
                   order=sorted(df_f['classification_tier'].unique()), cut=0)
    ax3.set_xlabel('Classification Tier')
    ax3.set_ylabel('Statement Support Score')
    kw_str, *_ = calculate_kruskal(df_f, 'classification_tier', 'statement_support_score')
    ax3.set_title(f'C. Confidence Score Distribution by Tier ({kw_str})',
                  loc='left', fontsize=12)

    ax4 = fig.add_subplot(gs[1, 1])
    sns.violinplot(x='statement_support_score', y='classification_category',
                   data=df_f, ax=ax4, palette='viridis', inner='quartile',
                   orient='h', cut=0)
    ax4.set_ylabel('Classification Category')
    ax4.set_xlabel('Statement Support Score')
    kw_str, *_ = calculate_kruskal(df_f, 'classification_category', 'statement_support_score')
    ax4.set_title(f'D. Confidence Score Distribution by Category ({kw_str})',
                  loc='left', fontsize=12)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y' if ax in [ax3] else 'x', linestyle='--', alpha=0.4)

    fig.savefig(os.path.join(output_dir, 'Figure_2_Evidence_Quality.pdf'),
                dpi=600, bbox_inches='tight')
    plt.show()
    print("Figure 2 saved.")


def create_tier_and_qol_overview_panel(df, colors, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), layout='constrained')

    df_f = df[~df['classification_tier'].astype(str).str.contains('NFC', na=False)
              & ~df['classification_category'].str.contains('NFC', na=False)]
    qol_df = df_f.explode('qol_domains').dropna(subset=['qol_domains'])

    qol_tier = pd.crosstab(qol_df['classification_tier'], qol_df['qol_domains'])
    qol_tier_pct = qol_tier.div(qol_tier.sum(axis=1), axis=0) * 100
    qol_tier_pct.plot(kind='bar', stacked=True, ax=ax1, colormap='plasma')
    chi_str, *_ = calculate_chisquare(qol_tier)
    ax1.set_title(f'A. QoL Domain Distribution by Tier ({chi_str})',
                  loc='left', fontsize=12)
    ax1.set_xlabel('Classification Tier')
    ax1.set_ylabel('Percentage of QoL Domains (%)')
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend(title='QoL Domain', frameon=False,
               bbox_to_anchor=(1.02, 1), loc='upper left')

    qol_cat = pd.crosstab(qol_df['classification_category'], qol_df['qol_domains'])
    qol_cat_pct = qol_cat.div(qol_cat.sum(axis=1), axis=0) * 100
    qol_cat_pct.plot(kind='barh', stacked=True, ax=ax2, colormap='plasma')
    chi_str, *_ = calculate_chisquare(qol_cat)
    ax2.set_title(f'B. QoL Domain Distribution by Category ({chi_str})',
                  loc='left', fontsize=12)
    ax2.set_xlabel('Percentage of QoL Domains (%)')
    ax2.set_ylabel('Classification Category')
    ax2.get_legend().remove()

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y' if ax == ax1 else 'x', linestyle='--', alpha=0.4)

    fig.savefig(os.path.join(output_dir, 'Figure_3_Tier_and_QoL_Overview.pdf'),
                dpi=600, bbox_inches='tight')
    plt.show()
    print("Figure 3 saved.")


def create_qol_analysis_panel(df, colors, output_dir):
    df_f = df[~df['classification_category'].str.contains('NFC|Out of Scope', na=False)
              & ~df['classification_tier'].astype(str).str.contains('NFC', na=False)]

    fig = plt.figure(figsize=(16, 12), layout='constrained')
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    qol_cat = pd.crosstab(df_f['classification_category'], df_f['QoL_assessment'])
    qol_cat_pct = qol_cat.div(qol_cat.sum(axis=1), axis=0) * 100
    qol_cat_pct.sort_values('Affected', ascending=True).plot(
        kind='barh', stacked=True, ax=ax1, color=[colors[1], colors[2]])
    ax1.set_xlabel('Percentage of Phenotypes (%)')
    ax1.set_ylabel('Classification Category')
    chi_str, *_ = calculate_chisquare(qol_cat)
    ax1.set_title(f'A. QoL Status by Classification Category ({chi_str})',
                  loc='left', fontsize=12)
    ax1.legend(title='QoL Status', frameon=False,
               bbox_to_anchor=(1.02, 1), loc='upper left')

    ax2 = fig.add_subplot(gs[0, 1])
    qol_tier = pd.crosstab(df_f['classification_tier'], df_f['QoL_assessment'])
    qol_tier_pct = qol_tier.div(qol_tier.sum(axis=1), axis=0) * 100
    qol_tier_pct.plot(kind='bar', stacked=True, ax=ax2,
                      color=[colors[1], colors[2]], rot=0)
    ax2.set_xlabel('Classification Tier')
    ax2.set_ylabel('Percentage of Phenotypes (%)')
    chi_str, *_ = calculate_chisquare(qol_tier)
    ax2.set_title(f'B. QoL Status by Classification Tier ({chi_str})',
                  loc='left', fontsize=12)
    ax2.legend(title='QoL Status', frameon=False,
               bbox_to_anchor=(1.02, 1), loc='upper left')

    ax3 = fig.add_subplot(gs[1, :])
    df_exp = df.explode('management_categories')
    df_exp = df_exp[df_exp['management_categories'].notna()
                    & (df_exp['management_categories'] != '')]
    df_exp = df_exp[~df_exp['management_categories'].isin(
        ['Genetic Counseling', 'Radiation Therapy', 'Conservative Management'])]
    qol_mgmt = pd.crosstab(df_exp['QoL_assessment'], df_exp['management_categories'])
    qol_mgmt = qol_mgmt.loc[:, qol_mgmt.sum(axis=0) > 0]
    qol_mgmt_pct = qol_mgmt.div(qol_mgmt.sum(axis=1), axis=0) * 100
    qol_mgmt_pct.T.sort_values('Affected').plot(
        kind='barh', ax=ax3, color=[colors[1], colors[2]])
    ax3.set_ylabel('Management Category')
    ax3.set_xlabel('Proportion of Phenotypes (%)')
    chi_str, *_ = calculate_chisquare(qol_mgmt)
    ax3.set_title(f'C. Management Profile by QoL Status ({chi_str})',
                  loc='left', fontsize=12)
    ax3.legend(title='QoL Status', frameon=False,
               bbox_to_anchor=(1.02, 1), loc='upper left')

    for ax in fig.get_axes():
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(output_dir, 'Figure_4_QoL_Impact_Analysis.pdf'),
                dpi=600, bbox_inches='tight')
    plt.show()
    print("Figure 4 saved.")


def create_management_drivers_panel(df, colors, output_dir):
    fig = plt.figure(figsize=(16, 10), layout='constrained')
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    qol_mgmt = df.groupby('num_qol_domains')['num_interventions'].mean()
    sns.regplot(x=qol_mgmt.index, y=qol_mgmt.values, ax=ax1,
                color=colors[2], ci=95, scatter_kws={'s': 50, 'alpha': 0.8})
    ax1.set_xlabel('Number of QoL Domains Affected')
    ax1.set_ylabel('Average Number of Interventions')
    reg_str, *_ = calculate_regression_stats(qol_mgmt.index, qol_mgmt.values)
    ax1.set_title(f'A. QoL Impact on Management Complexity ({reg_str})',
                  loc='left', fontsize=12)

    ax2 = fig.add_subplot(gs[0, 1])
    qol_df = df.explode('qol_domains').dropna(subset=['qol_domains'])
    sns.countplot(data=qol_df, y='qol_domains', ax=ax2, color=colors[1],
                  order=qol_df['qol_domains'].value_counts().index)
    ax2.set_title('B. Frequency of ACOG QoL Domains', loc='left', fontsize=12)
    ax2.set_xlabel('Count')
    ax2.set_ylabel('QoL Domain')

    ax3 = fig.add_subplot(gs[1, 0])
    df_tier = df[df['classification_tier'].astype(str) != 'NFC']
    sns.pointplot(data=df_tier, x='classification_tier', y='num_interventions',
                  ax=ax3, color=colors[3], errorbar='sd',
                  capsize=0.1, markers='D', linestyles='--')
    ax3.set_xlabel('Classification Tier')
    ax3.set_ylabel('Average Number of Interventions')
    kw_str, *_ = calculate_kruskal(df_tier, 'classification_tier', 'num_interventions')
    ax3.set_title(f'C. Management Complexity by Tier ({kw_str})',
                  loc='left', fontsize=12)

    ax4 = fig.add_subplot(gs[1, 1])
    df_cat = df[~df['classification_category'].str.contains('NFC|Out of Scope', na=False)]
    cat_mgmt = (df_cat.groupby('classification_category')['num_interventions']
                .agg(['mean', 'sem']).dropna())
    cat_mgmt.sort_values('mean', ascending=True, inplace=True)
    ax4.barh(cat_mgmt.index, cat_mgmt['mean'], xerr=cat_mgmt['sem'],
             color=colors[5], capsize=4, error_kw={'elinewidth': 1.5})
    ax4.set_xlabel('Average Number of Interventions')
    ax4.set_ylabel('Classification Category')
    kw_str, *_ = calculate_kruskal(df_cat, 'classification_category', 'num_interventions')
    ax4.set_title(f'D. Management Complexity by Category ({kw_str})',
                  loc='left', fontsize=12)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y' if ax in [ax1, ax3] else 'x', alpha=0.4, linestyle='--')

    fig.savefig(os.path.join(output_dir, 'Figure_5_Management_Drivers.pdf'),
                dpi=600, bbox_inches='tight')
    plt.show()
    print("Figure 5 saved.")


def create_distribution_and_efficiency_panel(df, colors, output_dir):
    """
    Figure 6: two panels side by side.
    Panel A: histogram + KDE of statement support score.
    Panel B: scatter + regression + marginal histograms (cost vs complexity).

    Uses GridSpecFromSubplotSpec for the marginals so that the whole figure
    is self-contained and does NOT rely on canvas.tostring_rgb(), which was
    removed in matplotlib >= 3.8.
    """
    fig = plt.figure(figsize=(18, 7))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88,
                        bottom=0.14, wspace=0.38)

    outer_gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.3])

    # ---- Panel A ----
    ax_a = fig.add_subplot(outer_gs[0])
    sns.histplot(data=df, x='statement_support_score', kde=True,
                 ax=ax_a, color=colors[0], bins=30, line_kws={'lw': 1.8})
    mean_val = df['statement_support_score'].mean()
    ax_a.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                 label=f'Mean: {mean_val:.2f}')
    ax_a.legend(fontsize=10)
    ax_a.set_title('A. Distribution of Statement Support Score',
                   loc='left', fontsize=12, fontweight='bold', pad=8)
    ax_a.set_xlabel('Statement Support Score', fontweight='bold')
    ax_a.set_ylabel('Frequency', fontweight='bold')
    ax_a.spines[['top', 'right']].set_visible(False)
    ax_a.grid(axis='y', linestyle='--', alpha=0.4)

    # ---- Panel B: joint scatter + marginals ----
    inner_gs = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer_gs[1],
        width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.05, wspace=0.05,
    )
    ax_top    = fig.add_subplot(inner_gs[0, 0])
    ax_joint  = fig.add_subplot(inner_gs[1, 0])
    ax_right  = fig.add_subplot(inner_gs[1, 1])
    ax_corner = fig.add_subplot(inner_gs[0, 1])
    ax_corner.set_visible(False)

    plot_df = df[
        df['react_steps'].notna() &
        df['total_cost_usd'].notna() &
        (df['react_steps'] > 0)
    ].copy()

    if not plot_df.empty:
        x = plot_df['react_steps'].values
        y = plot_df['total_cost_usd'].values

        ax_joint.scatter(x, y, alpha=0.35, s=30,
                         color=colors[5], edgecolors='k', linewidths=0.2)

        slope, intercept, r_value, p_value, _ = linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax_joint.plot(x_line, slope * x_line + intercept,
                      color=colors[3], lw=2)

        p_txt = 'p < 0.001' if p_value < 0.001 else f'p = {p_value:.3f}'
        corr_str = f'r = {r_value:.2f}, {p_txt}'

        ax_top.hist(x, bins=25, color=colors[5], alpha=0.7, edgecolor='none')
        ax_top.set_xlim(ax_joint.get_xlim())
        ax_top.axis('off')

        ax_right.hist(y, bins=25, color=colors[5], alpha=0.7,
                      edgecolor='none', orientation='horizontal')
        ax_right.set_ylim(ax_joint.get_ylim())
        ax_right.axis('off')

        ax_joint.set_xlabel('Agent Complexity (React Steps)', fontweight='bold')
        ax_joint.set_ylabel('Total Cost (USD)', fontweight='bold')
        ax_joint.spines[['top', 'right']].set_visible(False)
        ax_joint.grid(linestyle='--', alpha=0.3)

        ax_top.set_title(f'B. Efficiency: Cost vs. Complexity ({corr_str})',
                         loc='left', fontsize=12, fontweight='bold', pad=8)
    else:
        ax_joint.text(0.5, 0.5, 'No data available',
                      ha='center', va='center', transform=ax_joint.transAxes)
        ax_top.set_visible(False)
        ax_right.set_visible(False)

    fig.savefig(os.path.join(output_dir, 'Figure_6_Distribution_and_Efficiency.pdf'),
                dpi=600, bbox_inches='tight')
    plt.show()
    print("Figure 6 saved.")


# =====================================================================
#  SECTION 5 : ANALYSIS TEXT REPORT
# =====================================================================
def generate_analysis_report(df, output_dir):
    report_path = os.path.join(output_dir, 'comprehensive_analysis_report_COMPLETE.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        def w(text):
            f.write(text + '\n')

        w('=' * 80)
        w('COMPREHENSIVE ANALYSIS REPORT - COMPLETE DATA')
        w('=' * 80)
        w(f'\nTotal Phenotypes Analyzed: {len(df):,}\n')

        w('\n1. OVERALL DATASET DISTRIBUTIONS')
        w('-' * 50)
        w('\nDistribution by Classification Tier:')
        w(df['classification_tier'].value_counts().to_string())
        w('\nDistribution by Classification Category:')
        w(df['classification_category'].value_counts().to_string())

        w('\n\n2. CLINICAL LANDSCAPE ANALYSIS')
        w('-' * 50)
        category_tier = pd.crosstab(df['classification_category'], df['classification_tier'])
        category_tier['total'] = category_tier.sum(axis=1)
        category_tier.sort_values('total', ascending=False, inplace=True)
        w('\n2A. TIER DISTRIBUTION BY CATEGORY (Figure 1A):')
        w(category_tier.to_string())
        pct = category_tier.drop('total', axis=1).div(category_tier['total'], axis=0) * 100
        for cat in pct.index:
            w(f'\n{cat}:')
            for tier in pct.columns:
                if pct.loc[cat, tier] > 0:
                    w(f'  Tier {tier}: {pct.loc[cat, tier]:.1f}%')
            w(f'  Total N: {category_tier.loc[cat, "total"]}')
        chi2_val, p_val, dof, _ = chi2_contingency(category_tier.drop('total', axis=1))
        w(f'\nChi-square (Category vs Tier): chi2={chi2_val:.2f}, df={dof}, '
          f'p={"<0.001" if p_val < 0.001 else f"{p_val:.4f}"}')

        w('\n\n2B. EARLY ONSET PREVALENCE BY CATEGORY (Figure 1B):')
        df_onset = df[~df['classification_category'].str.contains(
            'Out of Scope|NFC', na=False, case=False)]
        eo = df_onset.groupby('classification_category')['has_early_onset'].agg(
            ['sum', 'count', 'mean'])
        eo['percentage'] = eo['mean'] * 100
        eo = eo.sort_values('percentage', ascending=False)
        for cat, row in eo.iterrows():
            w(f'{cat}: {int(row["sum"])}/{int(row["count"])} ({row["percentage"]:.1f}%)')
        ct = pd.crosstab(df_onset['classification_category'], df_onset['has_early_onset'])
        chi2_val, p_val, dof, _ = chi2_contingency(ct)
        w(f'\nChi-square (Early Onset vs Category): chi2={chi2_val:.2f}, df={dof}, '
          f'p={"<0.001" if p_val < 0.001 else f"{p_val:.4f}"}')

        w('\n\n3. EVIDENCE QUALITY & VALIDATION')
        w('-' * 50)
        df_f = df[~df['classification_tier'].astype(str).str.contains('NFC', na=False)
                  & ~df['classification_category'].str.contains('NFC', na=False)]
        ev_cols = ['verify_direct_support', 'verify_valid_inferences',
                   'verify_weak_inferences', 'verify_unsupported']
        ev_tier = df_f.groupby('classification_tier')[ev_cols].sum()
        w('\n3A. EVIDENCE PROFILE BY TIER (Figure 2A):')
        w(ev_tier.to_string())
        ev_tier_pct = ev_tier.div(ev_tier.sum(axis=1), axis=0) * 100
        for tier in ev_tier_pct.index:
            w(f'\nTier {tier}:')
            w(f'  Direct Support: {ev_tier_pct.loc[tier, "verify_direct_support"]:.1f}%')
            w(f'  Valid Inferences: {ev_tier_pct.loc[tier, "verify_valid_inferences"]:.1f}%')
            w(f'  Weak Inferences: {ev_tier_pct.loc[tier, "verify_weak_inferences"]:.1f}%')
            w(f'  Unsupported: {ev_tier_pct.loc[tier, "verify_unsupported"]:.1f}%')
        groups = [g['statement_support_score'].dropna()
                  for _, g in df_f.groupby('classification_tier')]
        if len(groups) > 1:
            h, p = kruskal(*groups)
            w(f'\nKruskal-Wallis (Support Score across Tiers): H({len(groups)-1})={h:.2f}, '
              f'p={"<0.001" if p < 0.001 else f"{p:.4f}"}')

        w('\n\n4. QUALITY OF LIFE DOMAIN DISTRIBUTIONS')
        w('-' * 50)
        qol_df = df_f.explode('qol_domains').dropna(subset=['qol_domains'])
        qol_tier = pd.crosstab(qol_df['classification_tier'], qol_df['qol_domains'])
        w('\n4A. QOL DOMAIN COUNTS BY TIER (Figure 3A):')
        w(qol_tier.to_string())
        chi2_val, p_val, dof, _ = chi2_contingency(qol_tier)
        w(f'\nChi-square (QoL Domains vs Tier): chi2={chi2_val:.2f}, df={dof}, '
          f'p={"<0.001" if p_val < 0.001 else f"{p_val:.4f}"}')
        w('\nOverall QoL Domain Frequency:')
        w(qol_df['qol_domains'].value_counts().to_string())

        w('\n\n5. QUALITY OF LIFE IMPACT ANALYSIS')
        w('-' * 50)
        df_f2 = df[~df['classification_category'].str.contains('NFC|Out of Scope', na=False)
                   & ~df['classification_tier'].astype(str).str.contains('NFC', na=False)]
        qol_cat = pd.crosstab(df_f2['classification_category'], df_f2['QoL_assessment'])
        w('\n5A. QOL ASSESSMENT BY CATEGORY (Figure 4A):')
        w(qol_cat.to_string())
        chi2_val, p_val, dof, _ = chi2_contingency(qol_cat)
        w(f'\nChi-square: chi2={chi2_val:.2f}, df={dof}, '
          f'p={"<0.001" if p_val < 0.001 else f"{p_val:.4f}"}')

        w('\n\n6. MANAGEMENT STRATEGY DRIVERS')
        w('-' * 50)
        qol_mgmt_data = df.groupby('num_qol_domains')['num_interventions'].agg(
            ['mean', 'std', 'count'])
        w('\n6A. INTERVENTIONS BY QOL DOMAIN COUNT (Figure 5A):')
        for d, row in qol_mgmt_data.iterrows():
            w(f'  {int(d)} domains: {row["mean"]:.3f} +/- {row["std"]:.3f} '
              f'(n={int(row["count"])})')
        slope, intercept, r, p, _ = linregress(qol_mgmt_data.index, qol_mgmt_data['mean'])
        w(f'\nLinear Regression: R2={r**2:.3f}, slope={slope:.3f}, '
          f'p={"<0.001" if p < 0.001 else f"{p:.4f}"}')

        df_tier = df[df['classification_tier'].astype(str) != 'NFC']
        tier_stats = df_tier.groupby('classification_tier')['num_interventions'].describe()
        w('\n6C. INTERVENTIONS BY TIER (Figure 5C):')
        w(tier_stats.to_string())
        groups_t = [g['num_interventions'].dropna()
                    for _, g in df_tier.groupby('classification_tier')]
        if len(groups_t) > 1:
            h, p = kruskal(*groups_t)
            w(f'\nKruskal-Wallis: H({len(groups_t)-1})={h:.2f}, '
              f'p={"<0.001" if p < 0.001 else f"{p:.4f}"}')

        w('\n\n7. DISTRIBUTION AND EFFICIENCY ANALYSIS')
        w('-' * 50)
        w('\nStatement Support Score Statistics:')
        w(df['statement_support_score'].describe().to_string())
        hist_vals, bins = np.histogram(df['statement_support_score'].dropna(), bins=30)
        w('\nHistogram (30 bins):')
        for i in range(len(hist_vals)):
            w(f'  [{bins[i]:.3f}, {bins[i+1]:.3f}]: {hist_vals[i]}')

        w('\n\n8. EARLY ONSET IMPACT ANALYSIS')
        w('-' * 50)
        onset    = df[df['has_early_onset']]
        no_onset = df[~df['has_early_onset']]
        w(f'With Early Onset: {len(onset)} ({len(onset)/len(df)*100:.1f}%)')
        w(f'Without Early Onset: {len(no_onset)} ({len(no_onset)/len(df)*100:.1f}%)')
        w(f'Clinical Impact Score with Early Onset: '
          f'{onset["clinical_impact_score"].mean():.3f} +/- '
          f'{onset["clinical_impact_score"].std():.3f}')
        w(f'Clinical Impact Score without Early Onset: '
          f'{no_onset["clinical_impact_score"].mean():.3f} +/- '
          f'{no_onset["clinical_impact_score"].std():.3f}')

        w('\n\n9. KEY CORRELATIONS')
        w('-' * 50)
        pairs = [
            ('clinical_impact_score', 'verify_weighted_score', 'Clinical Impact vs Evidence Score'),
            ('clinical_impact_score', 'num_interventions',     'Clinical Impact vs Interventions'),
            ('total_cost_usd',        'react_steps',           'Cost vs React Steps'),
            ('num_qol_domains',       'num_interventions',     'QoL Domains vs Interventions'),
            ('statement_support_score', 'verify_weighted_score', 'Support Score vs Weighted Score'),
        ]
        for v1, v2, label in pairs:
            if v1 in df.columns and v2 in df.columns:
                clean = df[[v1, v2]].dropna()
                if len(clean) > 1:
                    r_p, p_p = pearsonr(clean[v1], clean[v2])
                    r_s, p_s = spearmanr(clean[v1], clean[v2])
                    w(f'\n{label}:')
                    w(f'  Pearson r={r_p:.3f}, p={p_p:.4f}')
                    w(f'  Spearman rho={r_s:.3f}, p={p_s:.4f}')
                    w(f'  N={len(clean)}')

        w('\n\n10. OPERATIONAL ANALYTICS')
        w('-' * 50)
        if 'total_cost_usd' in df.columns:
            w('\nTotal Cost (USD):')
            w(df['total_cost_usd'].describe(
                percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]
            ).round(6).to_string())
        if 'react_steps' in df.columns:
            w('\nReAct Steps:')
            w(df['react_steps'].describe(
                percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]
            ).round(2).to_string())
        ok_n = (df['review_status'] == 'OK').sum()
        w(f'\nReview Status OK: {ok_n}/{len(df)} ({ok_n/len(df)*100:.1f}%)')
        w('\nReview Status Breakdown:')
        for status, cnt in df['review_status'].value_counts().items():
            w(f'  {status}: {cnt} ({cnt/len(df)*100:.1f}%)')

        w('\n\n' + '=' * 80)
        w('LEGEND: *** p<0.001   ** p<0.01   * p<0.05   ns p>=0.05')
        w('=' * 80)
        w('END OF REPORT')
        w('=' * 80)

    print(f"Analysis report saved: {report_path}")


# =====================================================================
#  SECTION 6 : VALIDATION LABEL MAP & UTILITIES
# =====================================================================
LABEL_MAP = {
    'Clinical Sign or Laboratory Abnormality':    'Clinical Sign/Lab',
    'Dysmorphic feature':                         'Dysmorphic',
    'Immunodeficiency/cancer':                    'Immuno/Cancer',
    'Impaired mobility':                          'Impaired Mobility',
    'Infertility':                                'Infertility',
    'Intellectual disability':                    'Intellectual Dis.',
    'Internal physical malformations':            'Internal Malform.',
    'Mental illness':                             'Mental Illness',
    'NFC (Not Further Classifiable)':             'NFC',
    'Out of Scope/Medical/Environmental':         'Out of Scope',
    'Sensory Impairment - Hearing':               'Sens. - Hearing',
    'Sensory Impairment - Touch':                 'Sens. - Touch',
    'Sensory Impairment - Vision':                'Sens. - Vision',
    'Sensory impairment - touch, smell, taste':   'Sens. - Touch/Smell',
    'Shortened life span: adulthood':             'Short Life: Adult',
    'Shortened life span: childhood/adolescence': 'Short Life: Child',
    'Shortened life span: infancy':               'Short Life: Infancy',
}


def short_label(label):
    clean = str(label).strip()
    if clean in LABEL_MAP:
        return LABEL_MAP[clean]
    return clean if len(clean) <= 20 else clean[:18] + '...'


def calculate_psi(expected, actual):
    exp_dist = expected.value_counts(normalize=True).rename('expected')
    act_dist = actual.value_counts(normalize=True).rename('actual')
    psi_df = (pd.merge(exp_dist, act_dist, left_index=True, right_index=True, how='outer')
              .replace(0, 0.0001).fillna(0.0001))
    psi_df['psi'] = ((psi_df['actual'] - psi_df['expected'])
                     * np.log(psi_df['actual'] / psi_df['expected']))
    return psi_df['psi'].sum()


def interpret_psi(psi):
    if psi < 0.1:
        return f'No significant shift (PSI = {psi:.4f})'
    elif psi < 0.25:
        return f'Minor shift (PSI = {psi:.4f})'
    return f'Major shift (PSI = {psi:.4f})'


def calculate_overall_metrics(y_true_cat, y_pred_cat):
    return {
        'accuracy':           accuracy_score(y_true_cat, y_pred_cat),
        'precision_weighted': precision_score(y_true_cat, y_pred_cat,
                                              average='weighted', zero_division=0),
        'recall_weighted':    recall_score(y_true_cat, y_pred_cat,
                                           average='weighted', zero_division=0),
        'f1_weighted':        f1_score(y_true_cat, y_pred_cat,
                                       average='weighted', zero_division=0),
        'precision_macro':    precision_score(y_true_cat, y_pred_cat,
                                              average='macro', zero_division=0),
        'recall_macro':       recall_score(y_true_cat, y_pred_cat,
                                           average='macro', zero_division=0),
        'f1_macro':           f1_score(y_true_cat, y_pred_cat,
                                       average='macro', zero_division=0),
        'mcc':                matthews_corrcoef(y_true_cat, y_pred_cat),
    }


def get_top_misclassifications(y_true_cat, y_pred_cat, cat_labels, n=10):
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=cat_labels)
    np.fill_diagonal(cm, 0)
    errors = []
    for i, tl in enumerate(cat_labels):
        for j, pl in enumerate(cat_labels):
            if cm[i, j] > 0:
                errors.append({'True': tl, 'Predicted': pl, 'Count': cm[i, j]})
    top = sorted(errors, key=lambda x: x['Count'], reverse=True)[:n]
    lines = ['\n' + '='*80, 'TOP MISCLASSIFICATION PAIRS', '='*80]
    for e in top:
        lines.append(f"  True '{e['True']}' -> Predicted '{e['Predicted']}': "
                     f"{e['Count']} times")
    return '\n'.join(lines)


def generate_validation_report(y_true_cat, y_pred_cat, y_true_tier, y_pred_tier,
                                cat_labels, psi_value, comparison_df):
    parts = []
    parts.append('=' * 80)
    parts.append('SAMPLE REPRESENTATIVENESS ANALYSIS')
    parts.append('=' * 80)
    parts.append(interpret_psi(psi_value))
    parts.append('\nDistribution Comparison:')
    parts.append(comparison_df.to_string())
    parts.append('\n' + '-'*40 + '\n')

    parts.append('=' * 80)
    parts.append('OVERALL MODEL PERFORMANCE')
    parts.append('=' * 80)
    m = calculate_overall_metrics(y_true_cat, y_pred_cat)
    for k, v in m.items():
        parts.append(f'  {k}: {v:.4f}')
    parts.append('\n' + '-'*40 + '\n')

    parts.append('=' * 80)
    parts.append('MULTI-CLASS CATEGORICAL PERFORMANCE')
    parts.append('=' * 80)
    parts.append(f'  MCC: {matthews_corrcoef(y_true_cat, y_pred_cat):.4f}')
    parts.append(
        f'  Balanced Accuracy: {balanced_accuracy_score(y_true_cat, y_pred_cat):.4f}')
    parts.append(
        f'  Macro F1: {f1_score(y_true_cat, y_pred_cat, average="macro", zero_division=0):.4f}')
    parts.append(f"  Cohen's Kappa: {cohen_kappa_score(y_true_cat, y_pred_cat):.4f}")
    parts.append('\n' + '-'*40 + '\n')

    parts.append('DETAILED PER-CLASS REPORT\n')
    parts.append(classification_report(y_true_cat, y_pred_cat,
                                       labels=cat_labels, zero_division=0))
    parts.append(get_top_misclassifications(y_true_cat, y_pred_cat, cat_labels))
    parts.append('\n' + '-'*40 + '\n')

    parts.append('=' * 80)
    parts.append('ORDINAL SEVERITY TIER PERFORMANCE')
    parts.append('=' * 80)
    mae_s   = mean_absolute_error(y_true_tier, y_pred_tier)
    kappa_q = cohen_kappa_score(y_true_tier, y_pred_tier, weights='quadratic')
    tol_acc = np.mean(np.abs(y_true_tier - y_pred_tier) <= 1) * 100
    sp_corr, _ = spearmanr(y_true_tier, y_pred_tier)
    kt_corr, _ = kendalltau(y_true_tier, y_pred_tier)
    parts.append(f'  Quadratic-Weighted Kappa: {kappa_q:.4f}')
    parts.append(f'  MAE: {mae_s:.4f}')
    parts.append(f'  Tolerance Accuracy (+/-1): {tol_acc:.2f}%')
    parts.append(f'  Spearman: {sp_corr:.4f}')
    parts.append(f'  Kendall Tau: {kt_corr:.4f}')

    f1_mac   = f1_score(y_true_cat, y_pred_cat, average='macro', zero_division=0)
    max_diff = max(y_true_tier.max() - y_true_tier.min(), 1)
    joint    = 0.3 * f1_mac + 0.7 * (1 - mae_s / max_diff)
    parts.append(f'\n  Joint Score (0.3*F1 + 0.7*Severity): {joint:.4f}')

    tmp_df = pd.DataFrame({'ytc': y_true_cat, 'ypc': y_pred_cat, 'ytt': y_true_tier})
    tmp_df['correct'] = tmp_df['ytc'] == tmp_df['ypc']
    parts.append('\nCategory Accuracy by Tier:')
    for tier, acc in sorted(tmp_df.groupby('ytt')['correct'].mean().items()):
        parts.append(f'  Tier {tier}: {acc*100:.2f}%')

    return '\n'.join(parts)


# =====================================================================
#  SECTION 7 : VALIDATION DASHBOARD PANELS  (A-F)
# =====================================================================

def _val_style():
    plt.rcParams.update({
        'font.family':      'DejaVu Sans',
        'axes.titleweight': 'bold',
        'figure.dpi':       150,
        'axes.labelsize':   15,
        'xtick.labelsize':  13,
        'ytick.labelsize':  13,
        'legend.fontsize':  12,
    })


def fig_distribution_comparison(comparison_df, psi_interpretation, path):
    _val_style()
    plot_df = comparison_df[['Population Proportion (%)']].copy()
    plot_df['Sample Proportion (%)'] = (
        comparison_df['Current Count'] / comparison_df['Current Count'].sum() * 100)
    plot_df.index = [short_label(x) for x in plot_df.index]
    plot_df = plot_df.sort_values('Population Proportion (%)', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(plot_df))
    h = 0.35
    ax.barh(y - h/2, plot_df['Population Proportion (%)'], h,
            label='Population', color='#2c7bb6', edgecolor='white')
    ax.barh(y + h/2, plot_df['Sample Proportion (%)'], h,
            label='Sample (GT)', color='#abd9e9', edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.index, fontsize=14)
    ax.set_xlabel('Proportion (%)', fontsize=16, weight='bold')
    ax.set_title('A   Distribution Comparison\n(Sample vs. Population)',
                 fontsize=18, weight='bold', loc='left')
    ax.legend(fontsize=12, loc='upper right',
              bbox_to_anchor=(1.0, -0.1), ncol=2, frameon=True)
    ax.grid(axis='x', ls='--', alpha=0.4)
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def fig_overall_performance(y_true_cat, y_pred_cat, path):
    _val_style()
    m = calculate_overall_metrics(y_true_cat, y_pred_cat)
    names = ['Accuracy', 'Prec.\n(Wt)', 'Rec.\n(Wt)', 'F1\n(Wt)',
             'Prec.\n(Mac)', 'Rec.\n(Mac)', 'F1\n(Mac)', 'MCC']
    vals  = [m['accuracy'], m['precision_weighted'], m['recall_weighted'],
             m['f1_weighted'], m['precision_macro'], m['recall_macro'],
             m['f1_macro'], m['mcc']]
    bar_colors = ['#2E7D32', '#1565C0', '#7B1FA2', '#C62828',
                  '#F57C00', '#616161', '#00897B', '#D81B60']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(names, vals, color=bar_colors, edgecolor='black', lw=1.2, width=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{v:.3f}', ha='center', va='bottom', fontsize=13, weight='bold')
    ax.axhline(0.8, color='grey', ls='--', lw=1, alpha=0.6, label='0.80 reference')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=16, weight='bold')
    ax.set_title('B   Overall Model Performance', fontsize=18, weight='bold', loc='left')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', ls='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def fig_confusion_matrix(y_true_cat, y_pred_cat, labels, path):
    _val_style()
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    short_labels = [short_label(l) for l in labels]
    n    = len(labels)
    size = max(14, n * 1.0)
    fig, ax = plt.subplots(figsize=(size + 2, size))
    sns.heatmap(cm_norm * 100, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=short_labels, yticklabels=short_labels, ax=ax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Recall Rate (%)', 'shrink': 0.8},
                annot_kws={'size': 13}, square=True)
    ax.set_title('C   Normalized Confusion Matrix',
                 fontsize=18, weight='bold', loc='left', pad=14)
    ax.set_ylabel('True Category', fontsize=16, weight='bold')
    ax.set_xlabel('Predicted Category', fontsize=16, weight='bold')
    ax.tick_params(axis='x', labelsize=16, rotation=45)
    ax.tick_params(axis='y', labelsize=16, rotation=0)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def fig_metrics_by_category(class_report_dict, path):
    _val_style()
    df = pd.DataFrame(class_report_dict).T
    df = df.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    df = df[['precision', 'recall', 'f1-score']].sort_values('f1-score', ascending=True)
    df.index = [short_label(x) for x in df.index]
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.55)))
    y = np.arange(len(df))
    h = 0.25
    ax.barh(y - h, df['precision'], h, label='Precision', color='#a6cee3', edgecolor='white')
    ax.barh(y,     df['recall'],    h, label='Recall',    color='#1f78b4', edgecolor='white')
    ax.barh(y + h, df['f1-score'],  h, label='F1-Score',  color='#08306b', edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=14)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Score', fontsize=16, weight='bold')
    ax.set_title('D   Performance Metrics by Category',
                 fontsize=18, weight='bold', loc='left')
    ax.legend(fontsize=12, loc='upper right',
              bbox_to_anchor=(1.0, -0.12), ncol=3, frameon=True)
    ax.grid(axis='x', ls='--', alpha=0.4)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def fig_tier_scatter(y_true_tier, y_pred_tier, path):
    _val_style()
    fig, ax = plt.subplots(figsize=(7, 7))
    jitter = 0.18
    xt = y_true_tier + np.random.uniform(-jitter, jitter, size=len(y_true_tier))
    xp = y_pred_tier + np.random.uniform(-jitter, jitter, size=len(y_pred_tier))
    ax.scatter(xt, xp, alpha=0.35, s=60, edgecolors='k', linewidths=0.3,
               c=y_true_tier, cmap='RdYlGn_r')
    lo = min(y_true_tier.min(), y_pred_tier.min()) - 0.3
    hi = max(y_true_tier.max(), y_pred_tier.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('True Severity Tier', fontsize=16, weight='bold')
    ax.set_ylabel('Predicted Severity Tier', fontsize=16, weight='bold')
    ax.set_title('E   Jitter Scatter Plot of Tiers',
                 fontsize=18, weight='bold', loc='left')
    ax.legend(fontsize=13)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, ls='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def fig_metrics_by_tier(tier_report_dict, path):
    _val_style()
    df = pd.DataFrame(tier_report_dict).T
    df = df.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    df = df[['precision', 'recall', 'f1-score']]
    df.index = ['Tier ' + str(x) for x in df.index]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df['precision'], w, label='Precision', color='#a6cee3', edgecolor='white')
    ax.bar(x,     df['recall'],    w, label='Recall',    color='#1f78b4', edgecolor='white')
    ax.bar(x + w, df['f1-score'],  w, label='F1-Score',  color='#08306b', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=16, weight='bold')
    ax.set_title('F   Performance Metrics by Tier',
                 fontsize=18, weight='bold', loc='left')
    ax.legend(fontsize=12, loc='lower center',
              bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=True)
    ax.grid(axis='y', ls='--', alpha=0.3)
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def create_composite_dashboard(panel_paths, output_path):
    from matplotlib.image import imread
    fig = plt.figure(figsize=(42, 26))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.02, wspace=0.02)
    for idx, ppath in enumerate(panel_paths):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(imread(ppath))
        ax.axis('off')
    fig.savefig(output_path, format='pdf', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =====================================================================
#  SECTION 8 : MAIN
# =====================================================================
def main():
    print('\n' + '='*80)
    print('COMBINED ANALYSIS & VALIDATION PIPELINE')
    print('='*80 + '\n')

    # Guard: required input files
    if not os.path.exists(PREDICTED_FILE):
        print('=' * 70)
        print('  Predictions file not found:')
        print(f'     {PREDICTED_FILE}')
        print('\nClassified_terms.csv is the IP-protected output of main.py.')
        print('You can either:')
        print('  1) Run `python main.py` to generate it from your HPO term list, or')
        print('  2) Request the file from the corresponding author under a')
        print('     data use agreement (see README §Data availability).')
        print('=' * 70)
        return

    if not os.path.exists(GT_FILE):
        print(f'⚠️  Ground-truth file not found at {GT_FILE}.')
        print('Place GT_file.xlsx in data/ and re-run.')
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Outputs will be saved to: {OUTPUT_DIR}\n")

    colors = setup_plot_style()

    # ==========================================
    # PART A : ANALYSIS PIPELINE  (Figures 1-6)
    # ==========================================
    print('\n' + '-'*60)
    print('PART A : ANALYSIS PIPELINE')
    print('-'*60 + '\n')

    df_raw = load_data(PREDICTED_FILE)
    if df_raw.empty:
        print('ERROR: could not load predicted file. Aborting.')
        return

    df_raw = reclassify_clinical_signs_as_nfc(df_raw)
    df     = preprocess_data(df_raw.copy())

    create_clinical_landscape_panel(df, colors, OUTPUT_DIR)
    create_evidence_quality_panel(df, colors, OUTPUT_DIR)
    create_tier_and_qol_overview_panel(df, colors, OUTPUT_DIR)
    create_qol_analysis_panel(df, colors, OUTPUT_DIR)
    create_management_drivers_panel(df, colors, OUTPUT_DIR)
    create_distribution_and_efficiency_panel(df, colors, OUTPUT_DIR)
    generate_analysis_report(df, OUTPUT_DIR)

    # ==========================================
    # PART B : VALIDATION PIPELINE  (Dashboard)
    # ==========================================
    print('\n' + '-'*60)
    print('PART B : VALIDATION PIPELINE')
    print('-'*60 + '\n')

    try:
        print('Loading ground-truth file...')
        gt_df   = pd.read_excel(GT_FILE)
        pred_df = pd.read_csv(PREDICTED_FILE)

        pred_df = pred_df.rename(columns={
            'classification_tier':     'classification_tier_pred',
            'classification_category': 'classification_category_pred',
        })

        def clean_categories(s):
            return (s
                    .str.replace('\u00e2\u20ac\u201c', ' - ', regex=False)
                    .str.replace('\u2013',             ' - ', regex=False)
                    .str.replace('Sensory impairment: touch, others',
                                 'Sensory Impairment - Touch', regex=False)
                    .str.replace('NFC (Not Further Classified)', 'NFC', regex=False)
                    .str.replace('NFC (Not Further Classifiable)', 'NFC', regex=False))

        gt_df['Final_classification_category']  = clean_categories(
            gt_df['Final_classification_category'])
        pred_df['classification_category_pred'] = clean_categories(
            pred_df['classification_category_pred'])

        cons_df = pred_df.copy()

        print('Merging...')
        eval_df = pd.merge(gt_df, pred_df, on='hpo_id', how='inner')
        eval_df['Final classification_tier'] = pd.to_numeric(
            eval_df['Final classification_tier'], errors='coerce')
        eval_df['classification_tier_pred']  = pd.to_numeric(
            eval_df['classification_tier_pred'], errors='coerce')

        orig = len(eval_df)
        eval_df.dropna(subset=[
            'Final_classification_category', 'classification_category_pred',
            'Final classification_tier',     'classification_tier_pred',
        ], inplace=True)
        dropped = orig - len(eval_df)
        if dropped:
            print(f'  Dropped {dropped} rows with missing values.')

        eval_df['Final classification_tier'] = eval_df['Final classification_tier'].astype(int)
        eval_df['classification_tier_pred']  = eval_df['classification_tier_pred'].astype(int)
        print(f'  {len(eval_df)} matched and clean terms.')

        y_true_cat  = eval_df['Final_classification_category']
        y_pred_cat  = eval_df['classification_category_pred']
        y_true_tier = eval_df['Final classification_tier']
        y_pred_tier = eval_df['classification_tier_pred']
        cat_labels  = sorted(y_true_cat.unique())

        print('Computing PSI...')
        gt_psi   = gt_df.rename(columns={'Final_classification_category': 'category'}) \
                   if 'Final_classification_category' in gt_df.columns else gt_df.copy()
        cons_psi = cons_df.rename(columns={'classification_category_pred': 'category'}) \
                   if 'classification_category_pred' in cons_df.columns else cons_df.copy()

        pop_dist     = cons_psi['category'].value_counts(normalize=True)
        cur_counts   = gt_psi['category'].value_counts()
        ideal_counts = (pop_dist * len(gt_psi)).round().astype(int)
        psi_value    = calculate_psi(expected=cons_psi['category'],
                                     actual=gt_psi['category'])
        psi_interp   = interpret_psi(psi_value)

        comparison_df = pd.DataFrame({
            'Population Proportion (%)': (pop_dist * 100).round(2),
            'Current Count':  cur_counts,
            'Ideal Count':    ideal_counts,
        }).fillna(0).astype({'Current Count': int, 'Ideal Count': int})
        comparison_df['Action Required (Difference)'] = (
            comparison_df['Ideal Count'] - comparison_df['Current Count'])
        comparison_df.sort_index(inplace=True)

        report_path = os.path.join(OUTPUT_DIR, 'comprehensive_validation_report.txt')
        print(f'Writing validation report -> {report_path}')
        report = generate_validation_report(
            y_true_cat, y_pred_cat, y_true_tier, y_pred_tier,
            cat_labels, psi_value, comparison_df)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        class_report_dict = classification_report(
            y_true_cat, y_pred_cat, labels=cat_labels,
            zero_division=0, output_dict=True)
        tier_labels_int  = sorted(y_true_tier.unique())
        tier_labels_str  = [str(t) for t in tier_labels_int]
        tier_report_dict = classification_report(
            y_true_tier, y_pred_tier, labels=tier_labels_int,
            target_names=tier_labels_str, zero_division=0, output_dict=True)

        print('Generating validation figures...')
        tmp_dir = tempfile.mkdtemp()
        panel_paths = [os.path.join(tmp_dir, f'panel_{c}.png')
                       for c in ['A', 'B', 'C', 'D', 'E', 'F']]

        fig_distribution_comparison(comparison_df, psi_interp, panel_paths[0])
        fig_overall_performance(y_true_cat, y_pred_cat,        panel_paths[1])
        fig_confusion_matrix(y_true_cat, y_pred_cat,
                             cat_labels,                        panel_paths[2])
        fig_metrics_by_category(class_report_dict,             panel_paths[3])
        fig_tier_scatter(y_true_tier, y_pred_tier,             panel_paths[4])
        fig_metrics_by_tier(tier_report_dict,                  panel_paths[5])

        print('Assembling composite dashboard...')
        dashboard_path = os.path.join(OUTPUT_DIR, 'comprehensive_dashboard.pdf')
        create_composite_dashboard(panel_paths, dashboard_path)
        shutil.rmtree(tmp_dir)

        m = calculate_overall_metrics(y_true_cat, y_pred_cat)
        print('\n' + '='*55)
        print(f'PSI:         {psi_interp}')
        print(f'Accuracy:    {m["accuracy"]:.4f}')
        print(f'F1 (Wt):     {m["f1_weighted"]:.4f}')
        print(f'F1 (Macro):  {m["f1_macro"]:.4f}')
        print(f'MCC:         {m["mcc"]:.4f}')
        print('='*55)

    except FileNotFoundError as e:
        print(f'\nERROR: file not found. {e}')
    except KeyError as e:
        print(f'\nERROR: missing column. {e}')
    except Exception as e:
        import traceback
        print(f'\nERROR: {e}')
        traceback.print_exc()

    print('\n' + '='*80)
    print('PIPELINE COMPLETE')
    print('='*80)
    print(f'\nAll outputs saved to: {OUTPUT_DIR}')
    print('\nAnalysis outputs:')
    print('  Figure_1_Clinical_Landscape_and_Onset.pdf')
    print('  Figure_2_Evidence_Quality.pdf')
    print('  Figure_3_Tier_and_QoL_Overview.pdf')
    print('  Figure_4_QoL_Impact_Analysis.pdf')
    print('  Figure_5_Management_Drivers.pdf')
    print('  Figure_6_Distribution_and_Efficiency.pdf')
    print('  comprehensive_analysis_report_COMPLETE.txt')
    print('\nValidation outputs:')
    print('  comprehensive_dashboard.pdf')
    print('  comprehensive_validation_report.txt')
    print('=' * 80 + '\n')


if __name__ == '__main__':
    main()
