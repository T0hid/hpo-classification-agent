import pandas as pd
import os
import sys
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
    Convert fraction strings like '3/8', '4/4', '1/7' to float (0.0-1.0).
    Also handles percentage strings like '75%'.
    Returns NaN if not a parseable fraction (e.g. HPO IDs, plain NaN).
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str.startswith('HP:'):
        return np.nan
    if '/' in val_str:
        parts = val_str.split('/')
        if len(parts) == 2:
            try:
                n, d = float(parts[0]), float(parts[1])
                return n / d if d > 0 else np.nan
            except ValueError:
                pass
    if val_str.endswith('%'):
        try:
            return float(val_str.replace('%', '')) / 100
        except ValueError:
            pass
    return np.nan


def calculate_severity(row):
    """Applies severity classification rules based on trait counts per tier."""
    count_tier_1 = row.get('1', 0)
    count_tier_2 = row.get('2', 0)
    count_tier_3 = row.get('3', 0)
    if count_tier_1 > 1: return 'Profound'
    if count_tier_1 == 1: return 'Severe'
    if count_tier_2 >= 1:
        return 'Severe' if (count_tier_2 + count_tier_3) >= 4 else 'Moderate'
    if count_tier_3 >= 1: return 'Moderate'
    return 'Mild'


# =============================================================================
# SECTION 2: FREQUENCY FILTER (NEW — integrated from frequency-filter version)
# =============================================================================

def apply_frequency_filter(merged_path, hpoa_path):
    """
    Joins phenotype.hpoa frequency data onto the merged phenotype file and filters
    to keep only Frequent / Very frequent / Obligate phenotypes (>= 30%).

    Frequency filter logic:
      KEEP   → Obligate (HP:0040280), Very frequent (HP:0040281), Frequent (HP:0040282)
      KEEP   → Fractions >= 30%  (e.g. 3/8=37.5% → keep, 4/4=100% → keep)
      KEEP   → NaN (no frequency annotation — valid unannotated OMIM phenotypes)
      REMOVE → Occasional (HP:0040283), Very rare (HP:0040284), Excluded (HP:0040285)
      REMOVE → Fractions <  30%  (e.g. 1/7=14% → remove, 1/4=25% → remove)
      REMOVE → qualifier = NOT   (negated phenotypes)

    Returns the filtered DataFrame.
    """
    try:
        print("=" * 80)
        print("FREQUENCY FILTER: Loading input files")
        print("=" * 80)
        df_merged = pd.read_csv(merged_path, low_memory=False)
        df_hpoa   = pd.read_csv(hpoa_path, sep='\t', comment='#', low_memory=False)
        print(f"  ✓ merged_gene_disease_phenotype.csv : {len(df_merged):>7,} rows")
        print(f"  ✓ phenotype.hpoa                    : {len(df_hpoa):>7,} rows")

        # ------------------------------------------------------------------
        # Parse frequency values from HPOA
        # ------------------------------------------------------------------
        print("\n  Parsing frequency values (HPO terms + fractions + %)...")
        df_freq = df_hpoa[['database_id', 'hpo_id', 'frequency', 'qualifier']].copy()
        df_freq = df_freq.drop_duplicates(subset=['database_id', 'hpo_id'])
        df_freq['freq_numeric'] = df_freq['frequency'].apply(parse_fraction)

        hpo_rows  = df_freq['frequency'].astype(str).str.startswith('HP:').sum()
        frac_rows = df_freq['freq_numeric'].notna().sum()
        nan_rows  = df_freq['frequency'].isna().sum()
        print(f"  HPO term frequency rows (HP:004028x) : {hpo_rows:>7,}")
        print(f"  Fractional frequency rows (n/d or %) : {frac_rows:>7,}")
        print(f"  No frequency annotation (NaN)        : {nan_rows:>7,}")

        # ------------------------------------------------------------------
        # Join frequency onto merged file
        # ------------------------------------------------------------------
        print("\n  Joining frequency onto merged file...")
        df = df_merged.merge(
            df_freq,
            left_on=['disease_original_curie', 'HPO_ID'],
            right_on=['database_id', 'hpo_id'],
            how='left'
        )
        matched     = df['frequency'].notna().sum()
        not_matched = df['frequency'].isna().sum()
        print(f"  ✓ Rows after join      : {len(df):>7,}")
        print(f"  Frequency matched      : {matched:>7,}")
        print(f"  No frequency (NaN-kept): {not_matched:>7,}")

        # ------------------------------------------------------------------
        # Apply filter
        # ------------------------------------------------------------------
        print("\n  Applying frequency filter...")
        is_hpo_remove  = df['frequency'].isin(REMOVE_FREQ)
        is_frac_remove = df['freq_numeric'].notna() & (df['freq_numeric'] < FREQ_THRESHOLD)
        is_not_qual    = df['qualifier'].astype(str).str.upper().str.strip() == 'NOT'

        mask_remove = is_hpo_remove | is_frac_remove | is_not_qual
        df_filtered = df[~mask_remove].copy()

        print(f"  Rows BEFORE filter                         : {len(df):>7,}")
        print(f"  Removed — HPO low-freq terms (Occ/VR/Exc)  : {is_hpo_remove.sum():>7,}")
        print(f"  Removed — Fractions < 30%                  : {is_frac_remove.sum():>7,}")
        print(f"  Removed — qualifier = NOT                   : {(~is_hpo_remove & ~is_frac_remove & is_not_qual).sum():>7,}")
        print(f"  Rows AFTER filter                          : {len(df_filtered):>7,}")
        print(f"  Net removed                                : {len(df) - len(df_filtered):>7,}")

        # Human-readable frequency category for the kept data
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

        print(f"\n✅ Frequency filter applied successfully.")
        return df_filtered

    except Exception as e:
        print(f"❌ Unexpected error in frequency filter: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# SECTION 3: CORE ANALYSIS FUNCTIONS
# =============================================================================

def run_severity_classification(df_input, output_dir, from_dataframe=True, file_path=None):
    """
    (Original Script 1 - Part A)
    Performs the main severity classification based on phenotype tiers.
    Generates 'gene_severity_classification_excluding_nfc.csv'.

    If from_dataframe=True, uses df_input as the already-filtered DataFrame.
    If from_dataframe=False, reads from file_path instead (original behaviour).
    """
    try:
        print("=" * 80 + "\nPART A: RUNNING OVERALL SEVERITY CLASSIFICATION\n" + "=" * 80)

        if from_dataframe and df_input is not None:
            df = df_input.copy()
            print(f"✓ Using frequency-filtered DataFrame with {len(df)} rows")
        else:
            df = pd.read_csv(file_path)
            print(f"✓ Loaded {len(df)} rows from {os.path.basename(file_path)}")

        df['classification_tier'] = df['classification_tier'].astype(str)
        grouping_cols = ['HGNC', 'Gene Name', 'MONDO_ID', 'Disease']
        all_tiers_per_gene = df.groupby(grouping_cols)['classification_tier'].apply(set).reset_index(name='tier_set')
        nfc_only = all_tiers_per_gene[all_tiers_per_gene['tier_set'] == {'NFC'}]
        exclude_set = set(nfc_only[grouping_cols].apply(tuple, axis=1))
        df['gene_disease_key'] = df[grouping_cols].apply(tuple, axis=1)
        df_valid = df[~df['gene_disease_key'].isin(exclude_set)]
        valid_gene_disease = df_valid[grouping_cols].drop_duplicates()
        df_filtered = df_valid[df_valid['classification_tier'].isin(['1', '2', '3'])]

        tier_counts = (
            df_filtered.groupby(grouping_cols)['classification_tier']
            .value_counts()
            .unstack(fill_value=0)
        ) if len(df_filtered) > 0 else pd.DataFrame()

        tier_counts_complete = valid_gene_disease.set_index(grouping_cols).join(tier_counts, how='left').fillna(0)
        severity_results = tier_counts_complete.apply(calculate_severity, axis=1)
        severity_df = severity_results.reset_index(name='Severity')
        tier_counts_for_output = tier_counts_complete.reset_index().rename(
            columns={'1': 'Tier1_Count', '2': 'Tier2_Count', '3': 'Tier3_Count'}
        )
        for col in ['Tier1_Count', 'Tier2_Count', 'Tier3_Count']:
            if col not in tier_counts_for_output.columns:
                tier_counts_for_output[col] = 0

        final_df = pd.merge(severity_df, tier_counts_for_output, on=grouping_cols)
        output_path = os.path.join(output_dir, 'gene_severity_classification_excluding_nfc.csv')
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ SUCCESS! Part A results saved to: {output_path}")

        if len(nfc_only) > 0:
            excluded_path = os.path.join(output_dir, 'excluded_nfc_only_cases.csv')
            nfc_only[grouping_cols].to_csv(excluded_path, index=False)
            print(f"📝 List of excluded NFC-only cases saved to: {excluded_path}")

        # Also save the frequency-filtered phenotype file for reference
        if from_dataframe and df_input is not None:
            filt_path = os.path.join(output_dir, 'merged_gene_disease_phenotype_freq_filtered.csv')
            df_input.to_csv(filt_path, index=False)
            print(f"📝 Frequency-filtered phenotype file saved to: {filt_path}")

        return final_df, len(nfc_only)

    except Exception as e:
        print(f"❌ Unexpected error in Part A: {e}")
        traceback.print_exc()
        return None, 0


def prepare_parent_term_data(severity_results_path, parent_term_source_path,
                             parent_term_source_df=None):
    """
    (Original Script 1 - Part B)
    Prepares data for the Parent Term (Body System) analysis.

    If parent_term_source_df is provided, uses it instead of reading from
    parent_term_source_path (allows using the frequency-filtered DataFrame).
    """
    try:
        print("\n" + "=" * 80 + "\nPART B: PREPARING PARENT TERM (BODY SYSTEM) DATA\n" + "=" * 80)
        severity_df = pd.read_csv(severity_results_path)

        if parent_term_source_df is not None:
            source_df = parent_term_source_df.copy()
            print(f"✓ Using frequency-filtered DataFrame as parent term source ({len(source_df)} rows)")
        else:
            source_df = pd.read_csv(parent_term_source_path)

        grouping_cols = ['HGNC', 'Gene Name', 'MONDO_ID', 'Disease']
        parent_term_map = source_df.groupby(grouping_cols)['Parent Term'].unique().reset_index()
        merged_df = pd.merge(severity_df, parent_term_map, on=grouping_cols, how='left')
        df = merged_df.dropna(subset=['Parent Term']).copy()

        expanded_data = []
        for _, row in df.iterrows():
            systems = row['Parent Term']
            for system in systems:
                if system:
                    expanded_data.append({'Severity': row['Severity'], 'Parent Term': system.strip()})
        expanded_df = pd.DataFrame(expanded_data)
        severity_order = ['Profound', 'Severe', 'Moderate', 'Mild']
        expanded_df['Severity'] = pd.Categorical(expanded_df['Severity'], categories=severity_order, ordered=True)
        crosstab_counts = pd.crosstab(expanded_df['Parent Term'], expanded_df['Severity'], dropna=False)

        if 'Not Classified' in crosstab_counts.index:
            crosstab_counts.drop('Not Classified', inplace=True)
            print("✓ Removed 'Not Classified' category from Parent Term analysis.")

        grand_total_associations = crosstab_counts.sum().sum()
        crosstab_percent_of_total = (
            (crosstab_counts / grand_total_associations) * 100
            if grand_total_associations > 0 else crosstab_counts
        )

        print("✓ Parent Term (Body System) data prepared.")
        return crosstab_percent_of_total, crosstab_counts

    except Exception as e:
        print(f"❌ Unexpected error in Part B: {e}")
        traceback.print_exc()
        return None, None


def prepare_moi_data(moi_input_file):
    """
    (Original Script 3)
    Prepares data for Severity by Mode of Inheritance (MOI) analysis.
    """
    try:
        print("\n" + "=" * 80 + "\nPART C: PREPARING MODE OF INHERITANCE (MOI) DATA\n" + "=" * 80)
        df = pd.read_csv(moi_input_file)
        df_clean = df.dropna(subset=['Severity', 'moi_title'])
        df_unique = df_clean.drop_duplicates(subset=['HGNC', 'MONDO_ID', 'Disease'])
        main_moi = ['Autosomal recessive', 'Autosomal dominant', 'X-linked']
        df_unique = df_unique.copy()
        df_unique['moi_grouped'] = df_unique['moi_title'].apply(
            lambda x: x if x in main_moi else 'Others'
        )
        severity_order = ['Mild', 'Moderate', 'Severe', 'Profound']
        moi_order = ['Autosomal dominant', 'Autosomal recessive', 'X-linked', 'Others']
        contingency = pd.crosstab(df_unique['moi_grouped'], df_unique['Severity'])
        contingency = contingency.reindex(index=moi_order, columns=severity_order, fill_value=0)
        contingency_pct = contingency.div(contingency.sum(axis=1), axis=0).mul(100).fillna(0)
        print("✓ Mode of Inheritance data prepared.")
        return contingency_pct, contingency

    except FileNotFoundError:
        print(f"❌ File not found for MOI analysis: {moi_input_file}. Skipping Part C.")
        return None, None
    except Exception as e:
        print(f"❌ Unexpected error in Part C: {e}")
        traceback.print_exc()
        return None, None


def prepare_comparison_data(severity_file, mm_file, sf_file, cs_file, output_dir):
    """
    (Original Scripts 2 & 4)
    Prepares data for comparing severity across different datasets (MM, SF, CS).
    sf_file and cs_file can be None if not available.
    """
    try:
        print("\n" + "=" * 80 + "\nPART D: PREPARING DATASET COMPARISON DATA\n" + "=" * 80)
        df_severity = pd.read_csv(severity_file)
        all_dists = []

        # MM dataset
        if mm_file and os.path.exists(mm_file):
            df_mm_severity = pd.read_csv(mm_file)
            if 'Severity' in df_mm_severity.columns:
                mm_dist = df_mm_severity['Severity'].value_counts(normalize=True).mul(100).reset_index()
                mm_dist.columns = ['Severity', 'Percentage']
                mm_dist['File'] = "Mackenzie's Mission"
                all_dists.append(mm_dist)
                print("✓ MM dataset loaded")
        else:
            print("⚠️  MM file not found — skipping")

        # SF dataset
        if sf_file and os.path.exists(sf_file):
            print("Processing SF.xlsx...")
            df_sf = pd.read_excel(sf_file)
            sf_merged = pd.merge(df_sf, df_severity, on='MONDO_ID', how='left')
            sf_merged_final = sf_merged[['gene_symbol', 'MONDO_ID', 'Disease_x', 'Severity']].rename(
                columns={'Disease_x': 'Disease'}
            )
            sf_output_file = os.path.join(output_dir, 'Sf_with_Severity.csv')
            sf_merged_final.to_csv(sf_output_file, index=False)
            sf_dist = sf_merged_final['Severity'].value_counts(normalize=True).mul(100).reset_index()
            sf_dist.columns = ['Severity', 'Percentage']
            sf_dist['File'] = 'Secondary finding'
            all_dists.append(sf_dist)
            print(f"✓ Saved Sf data with severity to: {sf_output_file}")
        else:
            print("⚠️  SF file not provided or not found — skipping")

        # CS dataset
        if cs_file and os.path.exists(cs_file):
            print("Processing CS.xlsx...")
            df_cs = pd.read_excel(cs_file)
            cs_merged = pd.merge(df_cs, df_severity, on='MONDO_ID', how='left')
            cs_merged_final = cs_merged[['gene_symbol', 'MONDO_ID', 'Disease_x', 'Severity']].rename(
                columns={'Disease_x': 'Disease'}
            )
            cs_output_file = os.path.join(output_dir, 'CS_with_Severity.csv')
            cs_merged_final.to_csv(cs_output_file, index=False)
            cs_dist = cs_merged_final['Severity'].value_counts(normalize=True).mul(100).reset_index()
            cs_dist.columns = ['Severity', 'Percentage']
            cs_dist['File'] = 'ACMG carrier screening'
            all_dists.append(cs_dist)
            print(f"✓ Saved CS data with severity to: {cs_output_file}")
        else:
            print("⚠️  CS file not provided or not found — skipping")

        if len(all_dists) == 0:
            print("⚠️  No comparison datasets available — skipping Plot D")
            return None

        plot_data_combined = pd.concat(all_dists, ignore_index=True)
        print("✓ Dataset comparison data prepared.")
        return plot_data_combined

    except Exception as e:
        print(f"❌ Unexpected error in Part D: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# SECTION 4: REPORT GENERATION
# =============================================================================

def generate_comprehensive_report(output_dir, severity_data, parent_term_counts, nfc_excluded_count):
    """Generates a text report for Parts A and B."""
    try:
        print("\n" + "=" * 80 + "\n📝 GENERATING COMPREHENSIVE SEVERITY REPORT (A & B)\n" + "=" * 80)
        report_percent_data = parent_term_counts.div(parent_term_counts.sum(axis=1), axis=0) * 100
        report_content = f"COMPREHENSIVE SEVERITY ANALYSIS REPORT (WITH FREQUENCY FILTER)\n"
        report_content += f"================================================================\n\n"
        report_content += f"Report Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += f"Frequency filter: Keep Obligate / Very frequent / Frequent (>= 30%)\n"
        report_content += f"                  Remove Occasional / Very rare / Excluded (< 30%)\n"
        report_content += f"                  Remove qualifier = NOT (negated phenotypes)\n\n"
        report_content += f"---------------------------------------\nPART A: OVERALL SEVERITY DISTRIBUTION\n---------------------------------------\n\n"
        report_content += f"A total of {len(severity_data)} gene-disease pairs were analyzed after excluding {nfc_excluded_count} pairs that only had 'Not Fulfilling Criteria' (NFC) phenotypes.\n\nOverall Distribution:\n"
        severity_counts = severity_data['Severity'].value_counts()
        for level in ['Profound', 'Severe', 'Moderate', 'Mild']:
            count = severity_counts.get(level, 0)
            percentage = (count / len(severity_data)) * 100
            report_content += f"- {level:<10}: {count:>5} pairs ({percentage:>4.1f}%)\n"

        report_content += f"\n\n------------------------------------------------------\nPART B: SEVERITY DISTRIBUTION WITHIN EACH PARENT TERM\n------------------------------------------------------\n\n"
        report_content += f"The following table details the severity distribution *within* each of the {len(report_percent_data)} analyzed parent terms.\nEach row in the percentage columns sums to 100%. The table is sorted by the percentage of 'Profound' cases.\n\n"
        full_parent_term_data = pd.concat(
            [parent_term_counts.sum(axis=1).rename('Total Assocs'), report_percent_data], axis=1
        ).sort_values(by='Profound', ascending=False)
        report_content += "-" * 80 + "\n"
        report_content += f"{'Parent Term (Body System)':<45} {'Total Assocs':>12} {'%P':>6} {'%S':>6} {'%M':>6} {'%L':>6}\n"
        report_content += "-" * 80 + "\n"
        for system, row in full_parent_term_data.iterrows():
            p = row.get('Profound', 0)
            s = row.get('Severe', 0)
            m = row.get('Moderate', 0)
            l = row.get('Mild', 0)
            report_content += f"{system:<45} {int(row['Total Assocs']):>12} {p:>5.1f} {s:>6.1f} {m:>6.1f} {l:>6.1f}\n"

        report_path = os.path.join(output_dir, 'comprehensive_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_content)
        print(f"✓ Comprehensive report saved successfully to: {report_path}")

    except Exception as e:
        print(f"❌ Could not generate comprehensive report: {e}")
        traceback.print_exc()


def generate_moi_report(output_dir, contingency_pct, contingency):
    """Generates a text report for Part C."""
    try:
        print("\n" + "=" * 80 + "\n📝 GENERATING MODE OF INHERITANCE REPORT (C)\n" + "=" * 80)
        report_lines = [
            "=" * 80,
            "SEVERITY DISTRIBUTION SUMMARY BY MODE OF INHERITANCE (GROUPED)",
            "(WITH FREQUENCY FILTER APPLIED)",
            "=" * 80,
        ]
        report_lines.append("\n📈 Key Insights (Percentages of Profound Severity):")
        report_lines.append("-" * 50)
        for moi in contingency_pct.index:
            profound_pct = contingency_pct.loc[moi, 'Profound']
            report_lines.append(f"  • {moi:<25}: {profound_pct:.1f}% are Profound severity")
        report_lines.append("\nRAW COUNTS (CONTINGENCY TABLE)\n" + "=" * 80)
        report_lines.append(contingency.to_string())
        report_lines.append("\n\nPERCENTAGE DATA (%)\n" + "=" * 80)
        report_lines.append(contingency_pct.round(1).to_string())
        report_path = os.path.join(output_dir, "severity_distribution_summary.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"✓ MOI report saved successfully to: {report_path}")

    except Exception as e:
        print(f"❌ Could not generate MOI report: {e}")
        traceback.print_exc()


# =============================================================================
# SECTION 5: MAIN ORCHESTRATION AND PLOTTING
# =============================================================================

if __name__ == "__main__":
    # --- Define File Paths (auto-detects Code Ocean vs. local) ---
    HERE = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir("/data") and os.path.isdir("/results"):
        base_path = "/data"
        output_directory = "/results"
    else:
        base_path = os.path.join(HERE, "data")
        output_directory = os.path.join(HERE, "results")
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)

    # Input files (all in base_path, matching Code 2)
    severity_input_file = os.path.join(base_path, 'merged_gene_disease_phenotype.csv')
    hpoa_file           = os.path.join(base_path, 'phenotype.hpoa')
    moi_input_file      = os.path.join(base_path, 'final_output_with_severity.csv')
    mm_file             = os.path.join(base_path, 'MM_with_Severity.csv')

    # For comparison plots
    data_dir_comparison = base_path
    cs_file = os.path.join(data_dir_comparison, 'CS.xlsx')
    sf_file = os.path.join(data_dir_comparison, 'SF.xlsx')

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    print(f"📂 All output will be saved to: {output_directory}")

    # =========================================================================
    # GUARD: Check for required input files. The IP-protected datasets
    # (Classified_terms-derived files) are not redistributed; see README
    # for the data availability statement.
    # =========================================================================
    _required = {
        "merged_gene_disease_phenotype.csv": severity_input_file,
        "phenotype.hpoa": hpoa_file,
        "final_output_with_severity.csv": moi_input_file,
    }
    _missing = [name for name, path in _required.items() if not os.path.exists(path)]
    if _missing:
        print("\n" + "=" * 70)
        print("⚠️  Required input files are missing from the data/ folder:")
        for name in _missing:
            print(f"     - {name}")
        print("\nPublic data:")
        print("  • phenotype.hpoa — run `python scripts/download_public_data.py`")
        print("\nIP-protected data (not redistributed):")
        print("  • merged_gene_disease_phenotype.csv")
        print("  • final_output_with_severity.csv")
        print("  Available from the corresponding author under a")
        print("  data use agreement. See README §Data availability.")
        print("=" * 70)
        sys.exit(1)

    # =========================================================================
    # NEW STEP: Apply HPO frequency filter BEFORE severity classification
    # =========================================================================
    df_freq_filtered = apply_frequency_filter(severity_input_file, hpoa_file)

    if df_freq_filtered is not None:
        # Save the filtered phenotype file
        filtered_pheno_path = os.path.join(output_directory, 'merged_gene_disease_phenotype_freq_filtered.csv')
        df_freq_filtered.to_csv(filtered_pheno_path, index=False)
        print(f"📝 Frequency-filtered phenotype file saved to: {filtered_pheno_path}")

        # --- Run severity classification on the FILTERED data ---
        severity_data, nfc_count = run_severity_classification(
            df_input=df_freq_filtered,
            output_dir=output_directory,
            from_dataframe=True
        )
    else:
        print("\n⚠️  Frequency filter failed or HPOA file not found.")
        print("    Falling back to original behaviour (no frequency filter).\n")
        severity_data, nfc_count = run_severity_classification(
            df_input=None,
            output_dir=output_directory,
            from_dataframe=False,
            file_path=severity_input_file
        )
        df_freq_filtered = None  # Signal that no filtered DF is available

    severity_output_file_path = os.path.join(output_directory, 'gene_severity_classification_excluding_nfc.csv')

    # --- Part B: Parent Term analysis (uses filtered data if available) ---
    parent_term_percent_data, parent_term_count_data = prepare_parent_term_data(
        severity_output_file_path,
        severity_input_file,
        parent_term_source_df=df_freq_filtered  # Pass filtered DF if available
    )

    # --- Part C: MOI analysis ---
    moi_percent_data, moi_count_data = prepare_moi_data(moi_input_file)

    # --- Part D: Dataset comparison ---
    comparison_data = prepare_comparison_data(
        severity_output_file_path, mm_file, sf_file, cs_file, output_directory
    )

    # --- Generate Panel Visualization ---
    has_any_data = severity_data is not None
    if has_any_data:
        print("\n" + "=" * 80 + "\n🎨 CREATING COMBINED VISUALIZATION PANEL\n" + "=" * 80)
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(24, 18))
        fig.suptitle('Comprehensive Disease Severity Analysis (Frequency-Filtered)', fontsize=26, weight='bold')

        severity_order = ['Profound', 'Severe', 'Moderate', 'Mild']
        traffic_light_palette = {
            'Profound': '#d73027', 'Severe': '#fc8d59',
            'Moderate': '#fee08b', 'Mild': '#91cf60'
        }

        # --- Plot A: Overall Severity Distribution ---
        ax1 = axes[0, 0]
        severity_counts_to_plot = severity_data['Severity'].value_counts().reindex(severity_order).fillna(0)
        sev_df_plot = pd.DataFrame({'Severity': severity_counts_to_plot.index, 'Count': severity_counts_to_plot.values})
        sns.barplot(data=sev_df_plot, x='Severity', y='Count', hue='Severity',
                    palette=traffic_light_palette, ax=ax1, order=severity_order, legend=False)
        ax1.set_title('A: Overall Gene-Disease Severity Distribution', fontsize=18, weight='bold')
        ax1.set_xlabel('Severity Level', fontsize=14)
        ax1.set_ylabel('Number of Gene-Disease Pairs', fontsize=14)
        for p in ax1.patches:
            ax1.annotate(f'{int(p.get_height())}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 9),
                         textcoords='offset points', fontsize=14)
        ax1.set_ylim(top=ax1.get_ylim()[1] * 1.1)
        ax1.tick_params(axis='x', rotation=15, labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)

        # --- Plot B: Severity by Parent Term ---
        ax2 = axes[0, 1]
        if parent_term_count_data is not None:
            plot_data_b_internal_dist = parent_term_count_data.div(
                parent_term_count_data.sum(axis=1), axis=0
            ).mul(100)
            plot_data_b = plot_data_b_internal_dist[severity_order].sort_values(by='Profound', ascending=True)
            plot_data_b.plot(kind='barh', stacked=True,
                             color=[traffic_light_palette.get(col) for col in severity_order],
                             width=0.8, ax=ax2)
            for container in ax2.containers:
                labels = [f'{w:.1f}%' if w > 4 else '' for w in container.datavalues]
                ax2.bar_label(container, labels=labels, label_type='center',
                              color='black', weight='bold', fontsize=11)
            ax2.set_title('B: Distribution of Severity by Body System', fontsize=18, weight='bold')
            ax2.set_xlabel('Percentage of Associations within System (%)', fontsize=14)
            ax2.set_ylabel('Parent Term (Body System)', fontsize=14)
            ax2.legend(title='Severity', bbox_to_anchor=(1.02, 1), loc='upper left',
                       title_fontsize='14', fontsize='12')
            ax2.set_xlim(0, 100)
            ax2.tick_params(axis='x', labelsize=12)
            ax2.tick_params(axis='y', labelsize=12)
        else:
            ax2.text(0.5, 0.5, 'Parent Term data not available',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('B: Distribution of Severity by Body System', fontsize=18, weight='bold')

        # --- Plot C: Severity by Mode of Inheritance ---
        ax3 = axes[1, 0]
        if moi_percent_data is not None and moi_count_data is not None:
            moi_order_plot = ['Autosomal dominant', 'Autosomal recessive', 'X-linked', 'Others']
            severity_plot_order = ['Mild', 'Moderate', 'Severe', 'Profound']

            moi_percent_data.loc[moi_order_plot, severity_plot_order].plot(
                kind='bar', stacked=True, ax=ax3,
                color=[traffic_light_palette[s] for s in severity_plot_order],
                edgecolor='black', linewidth=0.8, legend=False
            )
            for i, container in enumerate(ax3.containers):
                severity_level = severity_plot_order[i]
                labels = []
                for j, p in enumerate(container.patches):
                    percentage = p.get_height()
                    moi = moi_order_plot[j]
                    count = moi_count_data.loc[moi, severity_level]
                    if percentage > 4:
                        labels.append(f'{percentage:.0f}%\n(n={count})')
                    else:
                        labels.append('')
                ax3.bar_label(container, labels=labels, label_type='center',
                              color='black', weight='bold', fontsize=12)

            ax3.set_title('C: Severity Distribution by Mode of Inheritance', fontsize=18, weight='bold')
            ax3.set_xlabel('Mode of Inheritance', fontsize=14, labelpad=40)
            ax3.set_ylabel('Percentage of Gene-Disease Associations (%)', fontsize=14)
            ax3.set_xticklabels(moi_order_plot, size=12)
            total_counts = moi_count_data.sum(axis=1)
            for i, moi in enumerate(moi_order_plot):
                ax3.text(i, -12, f'Total n={total_counts[moi]:,}', ha='center', va='top',
                         fontsize=13, style='italic', color='dimgray')
            ax3.set_ylim(0, 100)
            ax3.tick_params(axis='x', rotation=15, labelsize=12)
            ax3.tick_params(axis='y', labelsize=12)

            handles, labels = ax3.get_legend_handles_labels()
            ax3.legend(reversed(handles), reversed(labels), title='Severity',
                       bbox_to_anchor=(1.02, 1), loc='upper left',
                       title_fontsize='14', fontsize='12')
        else:
            ax3.text(0.5, 0.5, 'MOI data not available',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=14)
            ax3.set_title('C: Severity Distribution by Mode of Inheritance', fontsize=18, weight='bold')

        # --- Plot D: Comparison of Severity Levels ---
        ax4 = axes[1, 1]
        if comparison_data is not None and len(comparison_data) > 0:
            plot_data_filtered = comparison_data[comparison_data['Severity'].isin(severity_order)]
            sns.barplot(data=plot_data_filtered, x='Severity', y='Percentage',
                        hue='File', order=severity_order, ax=ax4)
            for p in ax4.patches:
                if p.get_height() > 0:
                    ax4.annotate(f'{p.get_height():.1f}%',
                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha='center', va='center', xytext=(0, 9),
                                 textcoords='offset points', fontsize=12)
            ax4.set_title('D: Comparison of Disease Severity Levels Across Datasets', fontsize=18, weight='bold')
            ax4.set_xlabel('Severity Level', fontsize=14)
            ax4.set_ylabel('Percentage (%)', fontsize=14)
            ax4.legend(title='Dataset', title_fontsize='14', fontsize='12')
            ax4.tick_params(axis='x', rotation=15, labelsize=12)
            ax4.tick_params(axis='y', labelsize=12)
        else:
            ax4.text(0.5, 0.5, 'Comparison data not available',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('D: Comparison of Disease Severity Levels Across Datasets', fontsize=18, weight='bold')

        # --- Finalize and Save Plot ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        combined_plot_path = os.path.join(output_directory, 'combined_analysis_panel.pdf')
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Combined panel plot saved successfully to: {combined_plot_path}")
        plt.show()
        plt.close(fig)

        # --- Generate Reports ---
        if parent_term_count_data is not None:
            generate_comprehensive_report(output_directory, severity_data, parent_term_count_data, nfc_count)
        if moi_percent_data is not None and moi_count_data is not None:
            generate_moi_report(output_directory, moi_percent_data, moi_count_data)

    else:
        print("\n❌ Could not generate the final plot because severity classification failed.")

    print("\n" + "=" * 80)
    print(f"✅ ALL DONE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All outputs saved to: {output_directory}")
    print("=" * 80)
