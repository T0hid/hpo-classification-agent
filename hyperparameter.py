import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# ==============================================================================
# SCRIPT CONFIGURATION (auto-detects Code Ocean vs. local paths)
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir("/data") and os.path.isdir("/results"):
    DATA_DIR = "/data"
    RESULTS_DIR = "/results"
else:
    DATA_DIR = os.path.join(HERE, "data")
    RESULTS_DIR = os.path.join(HERE, "results")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_FILE = os.path.join(DATA_DIR, "hyperparameter_results.csv")

# *** Global Font Size and Style Enhancements ***
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['figure.titlesize'] = 26
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.title_fontsize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 20
SAVE_DPI = 300

# ==============================================================================
# STEP 1: SETUP OUTPUT DIRECTORY
# ==============================================================================
print("*** Step 1: Setting up output directory ***")
PLOTS_OUTPUT_DIR = RESULTS_DIR
print(f"✅ Plots will be saved in: {PLOTS_OUTPUT_DIR}")

# ==============================================================================
# STEP 2: LOAD DATA
# ==============================================================================
print("\n*** Step 2: Loading results file ***")
if not os.path.exists(TARGET_FILE):
    print(f"❌ ERROR: File not found at {TARGET_FILE}")
    exit()

print(f"✅ Loading data from: {TARGET_FILE}")
df = pd.read_csv(TARGET_FILE)

# Clean up model names for prettier labels
df['model_short_name'] = df['model'].apply(lambda x: str(x).split('/')[-1])

# *** FILTER OUT THE SPECIFIC MODEL ***
df = df[df['model_short_name'] != 'sonoma-sky-alpha']

# ==============================================================================
# STEP 3: GENERATE COMBINED PERFORMANCE HEATMAP
# ==============================================================================
print("\n*** Step 3: Generating heatmap ***")

accuracy_pivot = df.pivot_table(
    index='model_short_name', columns='temperature', values='overall_accuracy'
)
time_pivot = df.pivot_table(
    index='model_short_name', columns='temperature', values='avg_response_time'
)

annot_labels = np.full_like(accuracy_pivot, "", dtype=object)
for i, model in enumerate(accuracy_pivot.index):
    for j, temp in enumerate(accuracy_pivot.columns):
        acc = accuracy_pivot.iat[i, j]
        time = time_pivot.iat[i, j]
        if pd.notna(acc):
            annot_labels[i, j] = f"{acc:.1%}\n({time:.1f}s)"
        else:
            annot_labels[i, j] = "N/A"

plt.figure(figsize=(20, 12))
sns.heatmap(
    accuracy_pivot,
    annot=annot_labels,
    fmt="",
    linewidths=.5,
    cmap='cividis',
    annot_kws={"size": 14, "fontweight": "bold", "va": "center"}
)
plt.title('Combined Performance: Accuracy (Color) & Response Time (Text)', fontsize=26, pad=20)
plt.xlabel('Temperature Setting')
plt.ylabel('Model')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

heatmap_plot_path = os.path.join(PLOTS_OUTPUT_DIR, '3_combined_performance_heatmap.pdf')
plt.savefig(heatmap_plot_path, dpi=SAVE_DPI, bbox_inches='tight')
print(f"✅ Saved Combined Performance Heatmap: {os.path.basename(heatmap_plot_path)}")
plt.close()

print("\n🎉 Done. Heatmap saved in the /results directory.")
