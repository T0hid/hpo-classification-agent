import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
from datetime import datetime

# =============================================================================
# SECTION 1: HELPER FUNCTIONS
# =============================================================================

# ---- Frequency Filter Settings ----
KEEP_FREQ      = ['HP:0040280', 'HP:0040281', 'HP:0040282']
REMOVE_FREQ    = ['HP:0040283', 'HP:0040284', 'HP:0040285']
FREQ_THRESHOLD = 0.30   # 30% cutoff for fractional frequencies

FREQ_LABEL_MAP = {
    'HP:0040280': 'Obligate (100%)',
    'HP:0040281': 'Very frequent (80-99%)',
    'HP:0040282': 'Frequent (30-79%)',
    'HP:0040283': 'Occasional (5-29%)   ← REMOVED',
    'HP:0040284': 'Very rare (1-4%)     ← REMOVED',
    'HP:0040285': 'Excluded (0%)        ← REMOVED',
}


def parse_fraction(val):
    """
    Convert frequency values to float in [0, 1].
    """
    if pd.isna(val):
        return np.nan

    # --- pandas/Python datetime objects (Excel converted to real date) ---
    if isinstance(val, (pd.Timestamp, datetime)):
        n, d = val.month, val.day
        return n / d if d > 0 else np.nan

    val_str = str(val).strip()

    # Skip HPO term IDs and empty strings
    if not val_str or val_str.lower() == 'nan' or val_str.startswith('HP:'):
        return np.nan

    # Pattern: day-monthAbbr  (e.g. '8-Mar' from fraction '3/8')
    m = re.match(r'^(\d{1,2})[-\s/]([A-Za-z]{3,4})\.?$', val_str)
    if m:
        day = int(m.group(1))
        mon = m.group(2).lower().rstrip('.')
        if mon in MONTH_MAP:
            month = MONTH_MAP[mon]
            return month / day if day > 0 else np.nan

    # Pattern: monthAbbr-day  (e.g. 'Mar-8')
    m = re.match(r'^([A-Za-z]{3,4})\.?[-\s/](\d{1,2})$', val_str)
    if m:
        mon = m.group(1).lower().rstrip('.')
        day = int(m.group(2))
        if mon in MONTH_MAP:
            month = MONTH_MAP[mon]
            return month / day if day > 0 else np.nan

    # --- Excel full-date format: '3/8/2024', '3/8/24' ---
    m = re.match(r'^(\d{1,2})/(\d{1,2})/\d{2,4}$', val_str)
    if m:
        try:
            n, d = int(m.group(1)), int(m.group(2))
            return n / d if d > 0 else np.nan
        except ValueError:
            pass

    # --- Plain fraction: 'n/d' ---
    if '/' in val_str and val_str.count('/') == 1:
        parts = val_str.split('/')
        try:
            n, d = float(parts[0]), float(parts[1])
            return n / d if d > 0 else np.nan
        except ValueError:
            pass

    # --- Percentage: '75%' ---
    if val_str.endswith('%'):
        try:
            return float(val_str.replace('%', '').strip()) / 100
        except ValueError:
            pass

    # --- Bare decimal already in [0, 1] (rare but possible) ---
    try:
        f = float(val_str)
        if 0 <= f <= 1:
            return f
        if 0 < f <= 100:   # treat as percentage
            return f / 100
    except ValueError:
        pass

    return np.nan


def calculate_severity(row):
    """Severity classification rules based on trait counts per tier."""
    count_tier_1 = row.get('1', 0)
    count_tier_2 = row.get('2', 0)
    count_tier_3 = row.get('3', 0)
    if count_tier_1 > 1:  return 'Profound'
    if count_tier_1 == 1: return 'Severe'
    if count_tier_2 >= 1:
        return 'Severe' if (count_tier_2 + count_tier_3) >= 4 else 'Moderate'
    if count_tier_3 >= 1: return 'Moderate'
    return 'Mild'


# =============================================================================
# SECTION 2: LOAD MERGED FILE + APPLY FREQUENCY FILTER
# =============================================================================

def load_and_filter_merged_file(merged_path):
    """
    Loads the single merged file (CSV or pickle) which must already contain
    the 'frequency' and 'qualifier' columns. Use add_frequency_columns.py
    once to bake those columns in if they aren't there yet.

    Then applies the frequency filter:
      KEEP   → Obligate / Very frequent / Frequent  (HP:0040280-282)
      KEEP   → Fractions >= 30%
      KEEP   → NaN (no frequency annotation)
      REMOVE → Occasional / Very rare / Excluded  (HP:0040283-285)
      REMOVE → Fractions < 30%
      REMOVE → qualifier = NOT

    Also drops the pre-existing Severity column so it can be recomputed
    cleanly downstream.
    """
    try:
        print("=" * 80)
        print("LOADING MERGED FILE & APPLYING FREQUENCY FILTER")
        print("=" * 80)


        ext = os.path.splitext(merged_path)[1].lower()
        if ext == '.pkl':
            df = pd.read_pickle(merged_path)
        else:

            df = pd.read_csv(
                merged_path,
                low_memory=False,
                dtype={'frequency': str, 'qualifier': str},
                keep_default_na=True,
            )
        print(f"  ✓ Loaded {len(df):,} rows from {os.path.basename(merged_path)}")
        print(f"  Columns: {list(df.columns)}")

        # ---- Drop the pre-existing Severity column (will be recomputed) ----
        if 'Severity' in df.columns:
            df = df.drop(columns=['Severity'])
            print("  ✓ Dropped pre-existing 'Severity' column (will be recomputed)")

        # ---- Sanity check: required columns ----
        required = ['HGNC', 'Gene Name', 'MONDO_ID', 'Disease', 'HPO_ID',
                    'classification_tier', 'Parent Term', 'frequency', 'qualifier']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f" Missing required columns: {missing}")
            return None

        # ---- Parse frequency, with Excel-date un-corruption built in ----
        print("\n  Parsing frequency values (handles fractions, %, and Excel date corruption)...")
        df['freq_numeric'] = df['frequency'].apply(parse_fraction)

        # Audit: how many rows fell into each parse bucket?
        freq_str = df['frequency'].astype(str)
        n_hpo  = freq_str.str.startswith('HP:').sum()
        n_frac = df['freq_numeric'].notna().sum()
        n_nan  = df['frequency'].isna().sum()

        # Detect any leftover Excel-corrupted values that we successfully recovered
        date_pattern = re.compile(r'^\d{1,2}[-\s/][A-Za-z]{3,4}\.?$|^[A-Za-z]{3,4}\.?[-\s/]\d{1,2}$|^\d{1,2}/\d{1,2}/\d{2,4}$')
        n_recovered = freq_str.apply(lambda s: bool(date_pattern.match(str(s).strip()))).sum()

        print(f"  HPO frequency terms (HP:004028x)        : {n_hpo:>7,}")
        print(f"  Numeric frequencies parsed (n/d, %, etc): {n_frac:>7,}")
        print(f"    └─ of which Excel-date-corrupted, recovered: {n_recovered:>5,}")
        print(f"  NaN / unannotated (kept by default)     : {n_nan:>7,}")

        # ---- Apply the filter ----
        print("\n  Applying frequency filter...")
        is_hpo_remove  = df['frequency'].isin(REMOVE_FREQ)
        is_frac_remove = df['freq_numeric'].notna() & (df['freq_numeric'] < FREQ_THRESHOLD)
        is_not_qual    = df['qualifier'].astype(str).str.upper().str.strip() == 'NOT'
        mask_remove    = is_hpo_remove | is_frac_remove | is_not_qual

        df_filtered = df[~mask_remove].copy()

        print(f"  Rows BEFORE filter                         : {len(df):>7,}")
        print(f"  Removed — HPO low-freq terms (Occ/VR/Exc)  : {is_hpo_remove.sum():>7,}")
        print(f"  Removed — Fractions < 30%                  : {is_frac_remove.sum():>7,}")
        print(f"  Removed — qualifier = NOT                  : {(~is_hpo_remove & ~is_frac_remove & is_not_qual).sum():>7,}")
        print(f"  Rows AFTER filter                          : {len(df_filtered):>7,}")
        print(f"  Net removed                                : {len(df) - len(df_filtered):>7,}")

        # Human-readable category column
        def freq_category(row):
            freq = row['frequency']
            frac = row['freq_numeric']
            if pd.isna(freq):
                return 'Not annotated (kept)'
            freq_str = str(freq)
            if freq_str in FREQ_LABEL_MAP:
                return FREQ_LABEL_MAP[freq_str]
            if pd.notna(frac):
                pct = frac * 100
                if frac >= 0.80:   return f'Fraction ≥80% — Very frequent ({pct:.0f}%)'
                elif frac >= 0.30: return f'Fraction 30-79% — Frequent ({pct:.0f}%)'
                else:              return f'Fraction <30% — Removed ({pct:.0f}%)'
            return freq_str

        df_filtered['frequency_category'] = df_filtered.apply(freq_category, axis=1)
        print("\n  Frequency breakdown in KEPT data:")
        for cat, cnt in df_filtered['frequency_category'].value_counts().items():
            print(f"    {cnt:>7,}  {cat}")

        print("\n Frequency filter applied successfully.")
        return df_filtered

    except FileNotFoundError:
        print(f" Merged file not found: {merged_path}")
        return None
    except Exception as e:
        print(f" Unexpected error in load/filter step: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# SECTION 3: CORE ANALYSIS FUNCTIONS
# =============================================================================

def run_severity_classification(df_input, output_dir):
    """
    PART A — Recompute severity from tiers.
    Generates 'gene_severity_classification_excluding_nfc.csv'.
    """
    try:
        print("=" * 80 + "\nPART A: OVERALL SEVERITY CLASSIFICATION\n" + "=" * 80)
        df = df_input.copy()
        df['classification_tier'] = df['classification_tier'].astype(str)
        print(f"✓ Working with {len(df):,} frequency-filtered rows")

        grouping_cols = ['HGNC', 'Gene Name', 'MONDO_ID', 'Disease']

        # Identify gene-disease pairs whose tiers are ONLY {'NFC'} → exclude
        all_tiers = df.groupby(grouping_cols)['classification_tier'].apply(set).reset_index(name='tier_set')
        nfc_only  = all_tiers[all_tiers['tier_set'] == {'NFC'}]
        exclude_set = set(nfc_only[grouping_cols].apply(tuple, axis=1))

        df['gene_disease_key'] = df[grouping_cols].apply(tuple, axis=1)
        df_valid = df[~df['gene_disease_key'].isin(exclude_set)]
        valid_pairs = df_valid[grouping_cols].drop_duplicates()

        df_tiered = df_valid[df_valid['classification_tier'].isin(['1', '2', '3'])]

        if len(df_tiered) > 0:
            tier_counts = (df_tiered.groupby(grouping_cols)['classification_tier']
                                    .value_counts().unstack(fill_value=0))
        else:
            tier_counts = pd.DataFrame()

        tier_counts_complete = (valid_pairs.set_index(grouping_cols)
                                           .join(tier_counts, how='left').fillna(0))
        severity_df = tier_counts_complete.apply(calculate_severity, axis=1) \
                                          .reset_index(name='Severity')

        tier_out = tier_counts_complete.reset_index().rename(
            columns={'1': 'Tier1_Count', '2': 'Tier2_Count', '3': 'Tier3_Count'}
        )
        for col in ['Tier1_Count', 'Tier2_Count', 'Tier3_Count']:
            if col not in tier_out.columns:
                tier_out[col] = 0

        final_df = pd.merge(severity_df, tier_out, on=grouping_cols)
        out_path = os.path.join(output_dir, 'gene_severity_classification_excluding_nfc.csv')
        final_df.to_csv(out_path, index=False)
        print(f"\n Part A results saved to: {out_path}")

        if len(nfc_only) > 0:
            ex_path = os.path.join(output_dir, 'excluded_nfc_only_cases.csv')
            nfc_only[grouping_cols].to_csv(ex_path, index=False)
            print(f" Excluded NFC-only cases saved to: {ex_path}")

        return final_df, len(nfc_only)

    except Exception as e:
        print(f" Unexpected error in Part A: {e}")
        traceback.print_exc()
        return None, 0


def prepare_parent_term_data(severity_df, source_df):
    """PART B — Severity distribution by Parent Term (body system)."""
    try:
        print("\n" + "=" * 80 + "\nPART B: PARENT TERM (BODY SYSTEM) DATA\n" + "=" * 80)
        grouping_cols = ['HGNC', 'Gene Name', 'MONDO_ID', 'Disease']

        parent_map = source_df.groupby(grouping_cols)['Parent Term'].unique().reset_index()
        merged = pd.merge(severity_df, parent_map, on=grouping_cols, how='left')
        df = merged.dropna(subset=['Parent Term']).copy()

        rows = []
        for _, r in df.iterrows():
            for system in r['Parent Term']:
                if system:
                    rows.append({'Severity': r['Severity'], 'Parent Term': system.strip()})
        expanded = pd.DataFrame(rows)

        severity_order = ['Profound', 'Severe', 'Moderate', 'Mild']
        expanded['Severity'] = pd.Categorical(expanded['Severity'], categories=severity_order, ordered=True)
        crosstab = pd.crosstab(expanded['Parent Term'], expanded['Severity'], dropna=False)

        if 'Not Classified' in crosstab.index:
            crosstab.drop('Not Classified', inplace=True)
            print("✓ Removed 'Not Classified' category from Parent Term analysis.")

        total = crosstab.sum().sum()
        crosstab_pct = (crosstab / total * 100) if total > 0 else crosstab
        print("✓ Parent Term data prepared.")
        return crosstab_pct, crosstab

    except Exception as e:
        print(f" Unexpected error in Part B: {e}")
        traceback.print_exc()
        return None, None


def prepare_moi_data(source_df, severity_df):
    """
    PART C — Severity by Mode of Inheritance.
    Now reads MOI directly from the merged DataFrame and joins on the
    recomputed Severity (instead of reading a separate file).
    """
    try:
        print("\n" + "=" * 80 + "\nPART C: MODE OF INHERITANCE (MOI) DATA\n" + "=" * 80)
        grouping_cols = ['HGNC', 'Gene Name', 'MONDO_ID', 'Disease']

        # One MOI per gene-disease pair (take the first non-null)
        moi_map = (source_df.dropna(subset=['moi_title'])
                            .groupby(grouping_cols)['moi_title']
                            .first()
                            .reset_index())

        df = pd.merge(severity_df, moi_map, on=grouping_cols, how='inner')
        df = df.drop_duplicates(subset=['HGNC', 'MONDO_ID', 'Disease'])

        main_moi = ['Autosomal recessive', 'Autosomal dominant', 'X-linked']
        df['moi_grouped'] = df['moi_title'].apply(lambda x: x if x in main_moi else 'Others')

        sev_order = ['Mild', 'Moderate', 'Severe', 'Profound']
        moi_order = ['Autosomal dominant', 'Autosomal recessive', 'X-linked', 'Others']
        cont = pd.crosstab(df['moi_grouped'], df['Severity'])
        cont = cont.reindex(index=moi_order, columns=sev_order, fill_value=0)
        cont_pct = cont.div(cont.sum(axis=1), axis=0).mul(100).fillna(0)

        print("✓ MOI data prepared.")
        return cont_pct, cont

    except Exception as e:
        print(f" Unexpected error in Part C: {e}")
        traceback.print_exc()
        return None, None


def prepare_comparison_data(severity_df, mm_file, sf_file, cs_file, output_dir):
    """PART D — Comparison across MM / SF / CS datasets."""
    try:
        print("\n" + "=" * 80 + "\nPART D: DATASET COMPARISON\n" + "=" * 80)
        all_dists = []

        # MM
        if mm_file and os.path.exists(mm_file):
            df_mm = pd.read_csv(mm_file)
            if 'Severity' in df_mm.columns:
                d = df_mm['Severity'].value_counts(normalize=True).mul(100).reset_index()
                d.columns = ['Severity', 'Percentage']
                d['File'] = "Mackenzie's Mission"
                all_dists.append(d)
                print("✓ MM dataset loaded")
        else:
            print("  MM file not found — skipping")

        # SF
        if sf_file and os.path.exists(sf_file):
            df_sf = pd.read_excel(sf_file)
            sf_merged = pd.merge(df_sf, severity_df, on='MONDO_ID', how='left')
            sf_final = sf_merged[['gene_symbol', 'MONDO_ID', 'Disease_x', 'Severity']].rename(
                columns={'Disease_x': 'Disease'}
            )
            sf_final.to_csv(os.path.join(output_dir, 'Sf_with_Severity.csv'), index=False)
            d = sf_final['Severity'].value_counts(normalize=True).mul(100).reset_index()
            d.columns = ['Severity', 'Percentage']
            d['File'] = 'Secondary finding'
            all_dists.append(d)
            print("✓ SF dataset processed")
        else:
            print("  SF file not found — skipping")

        # CS
        if cs_file and os.path.exists(cs_file):
            df_cs = pd.read_excel(cs_file)
            cs_merged = pd.merge(df_cs, severity_df, on='MONDO_ID', how='left')
            cs_final = cs_merged[['gene_symbol', 'MONDO_ID', 'Disease_x', 'Severity']].rename(
                columns={'Disease_x': 'Disease'}
            )
            cs_final.to_csv(os.path.join(output_dir, 'CS_with_Severity.csv'), index=False)
            d = cs_final['Severity'].value_counts(normalize=True).mul(100).reset_index()
            d.columns = ['Severity', 'Percentage']
            d['File'] = 'ACMG carrier screening'
            all_dists.append(d)
            print("✓ CS dataset processed")
        else:
            print("  CS file not found — skipping")

        if not all_dists:
            print("  No comparison datasets — skipping Plot D")
            return None

        return pd.concat(all_dists, ignore_index=True)

    except Exception as e:
        print(f" Unexpected error in Part D: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# SECTION 4: REPORTS
# =============================================================================

def generate_comprehensive_report(output_dir, severity_data, parent_term_counts, nfc_excluded_count):
    try:
        print("\n" + "=" * 80 + "\n COMPREHENSIVE REPORT (A & B)\n" + "=" * 80)
        report_pct = parent_term_counts.div(parent_term_counts.sum(axis=1), axis=0) * 100
        rc  = f"COMPREHENSIVE SEVERITY ANALYSIS REPORT (WITH FREQUENCY FILTER)\n"
        rc += f"================================================================\n\n"
        rc += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        rc += f"Frequency filter: Keep Obligate / Very frequent / Frequent (>= 30%)\n"
        rc += f"                  Remove Occasional / Very rare / Excluded (< 30%)\n"
        rc += f"                  Remove qualifier = NOT\n"
        rc += f"                  Excel-corrupted dates ('4-Feb', '3/8/2024') auto-recovered\n\n"
        rc += f"---------------------------------------\nPART A: OVERALL SEVERITY DISTRIBUTION\n---------------------------------------\n\n"
        rc += f"A total of {len(severity_data)} gene-disease pairs were analyzed after excluding {nfc_excluded_count} NFC-only pairs.\n\nOverall Distribution:\n"
        sc = severity_data['Severity'].value_counts()
        for level in ['Profound', 'Severe', 'Moderate', 'Mild']:
            cnt = sc.get(level, 0); pct = (cnt / len(severity_data)) * 100
            rc += f"- {level:<10}: {cnt:>5} pairs ({pct:>4.1f}%)\n"

        rc += f"\n\n------------------------------------------------------\nPART B: SEVERITY DISTRIBUTION WITHIN EACH PARENT TERM\n------------------------------------------------------\n\n"
        rc += f"The table details severity distribution within each of the {len(report_pct)} parent terms.\nEach row sums to 100%. Sorted by % Profound.\n\n"
        full = pd.concat([parent_term_counts.sum(axis=1).rename('Total Assocs'), report_pct], axis=1) \
                 .sort_values(by='Profound', ascending=False)
        rc += "-" * 80 + "\n"
        rc += f"{'Parent Term (Body System)':<45} {'Total Assocs':>12} {'%P':>6} {'%S':>6} {'%M':>6} {'%L':>6}\n"
        rc += "-" * 80 + "\n"
        for sys_, row in full.iterrows():
            p = row.get('Profound', 0); s = row.get('Severe', 0)
            m = row.get('Moderate', 0); l = row.get('Mild', 0)
            rc += f"{sys_:<45} {int(row['Total Assocs']):>12} {p:>5.1f} {s:>6.1f} {m:>6.1f} {l:>6.1f}\n"

        path = os.path.join(output_dir, 'comprehensive_report.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(rc)
        print(f"✓ Saved: {path}")
    except Exception as e:
        print(f" Could not generate comprehensive report: {e}")
        traceback.print_exc()


def generate_moi_report(output_dir, contingency_pct, contingency):
    try:
        print("\n" + "=" * 80 + "\n MOI REPORT (C)\n" + "=" * 80)
        lines = ["=" * 80,
                 "SEVERITY DISTRIBUTION SUMMARY BY MODE OF INHERITANCE (GROUPED)",
                 "(WITH FREQUENCY FILTER APPLIED)",
                 "=" * 80,
                 "\n Key Insights (% Profound):",
                 "-" * 50]
        for moi in contingency_pct.index:
            lines.append(f"  • {moi:<25}: {contingency_pct.loc[moi, 'Profound']:.1f}% are Profound")
        lines.append("\nRAW COUNTS\n" + "=" * 80)
        lines.append(contingency.to_string())
        lines.append("\n\nPERCENTAGE DATA (%)\n" + "=" * 80)
        lines.append(contingency_pct.round(1).to_string())
        path = os.path.join(output_dir, "severity_distribution_summary.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"✓ Saved: {path}")
    except Exception as e:
        print(f" Could not generate MOI report: {e}")
        traceback.print_exc()


# =============================================================================
# SECTION 5: MAIN ORCHESTRATION + PLOTTING
# =============================================================================

if __name__ == "__main__":

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT  = SCRIPT_DIR  # adjust if the script lives in a sub-folder

    merged_input_file = os.path.join(REPO_ROOT, 'data', 'final_merged_output_with_freq.pkl')
    output_directory  = os.path.join(REPO_ROOT, 'output')

    # Comparison files (Part D) — put them in data/ in the repo too
    mm_file = os.path.join(REPO_ROOT, 'data', 'MM_with_Severity.csv')
    sf_file = os.path.join(REPO_ROOT, 'data', 'SF.xlsx')
    cs_file = os.path.join(REPO_ROOT, 'data', 'CS.xlsx')

    os.makedirs(output_directory, exist_ok=True)
    print(f" Output directory: {output_directory}")
    print(f" Input file:       {merged_input_file}\n")

    # --- Step 1: Load merged file + apply frequency filter -------------------
    df_freq_filtered = load_and_filter_merged_file(merged_input_file)
    if df_freq_filtered is None:
        print("\n Cannot proceed without merged file. Exiting.")
        raise SystemExit(1)
    # --- Step 2: Part A — recompute severity from tiers ---------------------
    severity_data, nfc_count = run_severity_classification(df_freq_filtered, output_directory)

    if severity_data is None:
        print("\n Severity classification failed. Exiting.")
        raise SystemExit(1)

    # --- Step 3: Part B — Parent Term analysis ------------------------------
    parent_term_pct, parent_term_count = prepare_parent_term_data(severity_data, df_freq_filtered)

    # --- Step 4: Part C — MOI analysis ----------------
    moi_pct, moi_count = prepare_moi_data(df_freq_filtered, severity_data)

    # --- Step 5: Part D — Dataset comparison --------------------------------
    comparison_data = prepare_comparison_data(severity_data, mm_file, sf_file, cs_file, output_directory)

    # --- Step 6: Combined panel plot ----------------------------------------
    print("\n" + "=" * 80 + "\n CREATING COMBINED VISUALIZATION PANEL\n" + "=" * 80)
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle('Comprehensive Disease Severity Analysis (Frequency-Filtered)',
                 fontsize=26, weight='bold')

    severity_order = ['Profound', 'Severe', 'Moderate', 'Mild']
    palette = {'Profound': '#d73027', 'Severe': '#fc8d59',
               'Moderate': '#fee08b', 'Mild': '#91cf60'}

    # Plot A
    ax1 = axes[0, 0]
    sc = severity_data['Severity'].value_counts().reindex(severity_order).fillna(0)
    sns.barplot(x=sc.index, y=sc.values, hue=sc.index, palette=palette,
                ax=ax1, order=severity_order, legend=False)
    ax1.set_title('A: Overall Gene-Disease Severity Distribution', fontsize=18, weight='bold')
    ax1.set_xlabel('Severity Level', fontsize=14)
    ax1.set_ylabel('Number of Gene-Disease Pairs', fontsize=14)
    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                     ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=14)
    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.1)
    ax1.tick_params(axis='x', rotation=15, labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)

    # Plot B
    ax2 = axes[0, 1]
    if parent_term_count is not None:
        plot_b = parent_term_count.div(parent_term_count.sum(axis=1), axis=0).mul(100)
        plot_b = plot_b[severity_order].sort_values(by='Profound', ascending=True)
        plot_b.plot(kind='barh', stacked=True,
                    color=[palette[c] for c in severity_order], width=0.8, ax=ax2)
        for ctr in ax2.containers:
            labels = [f'{w:.1f}%' if w > 4 else '' for w in ctr.datavalues]
            ax2.bar_label(ctr, labels=labels, label_type='center', color='black', weight='bold', fontsize=11)
        ax2.set_title('B: Distribution of Severity by Body System', fontsize=18, weight='bold')
        ax2.set_xlabel('Percentage of Associations within System (%)', fontsize=14)
        ax2.set_ylabel('Parent Term (Body System)', fontsize=14)
        ax2.legend(title='Severity', bbox_to_anchor=(1.02, 1), loc='upper left',
                   title_fontsize='14', fontsize='12')
        ax2.set_xlim(0, 100)
        ax2.tick_params(axis='x', labelsize=12); ax2.tick_params(axis='y', labelsize=12)
    else:
        ax2.text(0.5, 0.5, 'Parent Term data not available', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=14)
        ax2.set_title('B: Distribution of Severity by Body System', fontsize=18, weight='bold')

    # Plot C
    ax3 = axes[1, 0]
    if moi_pct is not None and moi_count is not None:
        moi_order = ['Autosomal dominant', 'Autosomal recessive', 'X-linked', 'Others']
        sev_plot_order = ['Mild', 'Moderate', 'Severe', 'Profound']
        moi_pct.loc[moi_order, sev_plot_order].plot(
            kind='bar', stacked=True, ax=ax3,
            color=[palette[s] for s in sev_plot_order],
            edgecolor='black', linewidth=0.8, legend=False
        )
        for i, ctr in enumerate(ax3.containers):
            sev_lvl = sev_plot_order[i]; lbls = []
            for j, p in enumerate(ctr.patches):
                pct = p.get_height(); m_ = moi_order[j]; cnt = moi_count.loc[m_, sev_lvl]
                lbls.append(f'{pct:.0f}%\n(n={cnt})' if pct > 4 else '')
            ax3.bar_label(ctr, labels=lbls, label_type='center', color='black', weight='bold', fontsize=12)
        ax3.set_title('C: Severity Distribution by Mode of Inheritance', fontsize=18, weight='bold')
        ax3.set_xlabel('Mode of Inheritance', fontsize=14, labelpad=40)
        ax3.set_ylabel('Percentage of Gene-Disease Associations (%)', fontsize=14)
        ax3.set_xticklabels(moi_order, size=12)
        totals = moi_count.sum(axis=1)
        for i, m_ in enumerate(moi_order):
            ax3.text(i, -12, f'Total n={totals[m_]:,}', ha='center', va='top',
                     fontsize=13, style='italic', color='dimgray')
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis='x', rotation=15, labelsize=12)
        ax3.tick_params(axis='y', labelsize=12)
        h, l = ax3.get_legend_handles_labels()
        ax3.legend(reversed(h), reversed(l), title='Severity',
                   bbox_to_anchor=(1.02, 1), loc='upper left',
                   title_fontsize='14', fontsize='12')
    else:
        ax3.text(0.5, 0.5, 'MOI data not available', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=14)
        ax3.set_title('C: Severity Distribution by Mode of Inheritance', fontsize=18, weight='bold')

    # Plot D
    ax4 = axes[1, 1]
    if comparison_data is not None and len(comparison_data) > 0:
        cmp_f = comparison_data[comparison_data['Severity'].isin(severity_order)]
        sns.barplot(data=cmp_f, x='Severity', y='Percentage', hue='File', order=severity_order, ax=ax4)
        for p in ax4.patches:
            if p.get_height() > 0:
                ax4.annotate(f'{p.get_height():.1f}%',
                             (p.get_x() + p.get_width()/2., p.get_height()),
                             ha='center', va='center', xytext=(0, 9),
                             textcoords='offset points', fontsize=12)
        ax4.set_title('D: Comparison of Disease Severity Levels Across Datasets', fontsize=18, weight='bold')
        ax4.set_xlabel('Severity Level', fontsize=14); ax4.set_ylabel('Percentage (%)', fontsize=14)
        ax4.legend(title='Dataset', title_fontsize='14', fontsize='12')
        ax4.tick_params(axis='x', rotation=15, labelsize=12); ax4.tick_params(axis='y', labelsize=12)
    else:
        ax4.text(0.5, 0.5, 'Comparison data not available', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=14)
        ax4.set_title('D: Comparison of Disease Severity Levels Across Datasets', fontsize=18, weight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    panel_path = os.path.join(output_directory, 'combined_analysis_panel.pdf')
    plt.savefig(panel_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined panel saved: {panel_path}")
    plt.show()
    plt.close(fig)

    # --- Step 7: Reports -----------------------------------------------------
    if parent_term_count is not None:
        generate_comprehensive_report(output_directory, severity_data, parent_term_count, nfc_count)
    if moi_pct is not None and moi_count is not None:
        generate_moi_report(output_directory, moi_pct, moi_count)

    print("\n" + "=" * 80)
    print(f" ALL DONE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Outputs saved to: {output_directory}")
    print("=" * 80)
