import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as path_effects
from datetime import datetime

# ==============================================================================
# SCRIPT CONFIGURATION FOR CODE OCEAN
# ==============================================================================
# Code Ocean standard paths
DATA_DIR = "/data"
RESULTS_DIR = "/results"

# Directly target the specific file shown in your file tree
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
# In Code Ocean, we write directly to the /results folder
PLOTS_OUTPUT_DIR = RESULTS_DIR 
print(f"✅ Plots will be saved in: {PLOTS_OUTPUT_DIR}")

# ==============================================================================
# STEP 2: AUTOMATICALLY LOAD THE SPECIFIC RESULTS FILE
# ==============================================================================
print("\n*** Step 2: Loading results file ***")

# Check if the file exists in the /data folder
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
# STEP 3: PERFORM DATA ANALYSIS
# ==============================================================================
print("\n*** Step 3: Analyzing data ***")
# Find the trial with the best accuracy for each model
best_per_model_accuracy = df.loc[df.groupby('model_short_name')['overall_accuracy'].idxmax()]

# Calculate an efficiency score (accuracy per second of response time)
df['efficiency_score'] = df['overall_accuracy'] / df['avg_response_time']
best_efficient_per_model = df.loc[df.groupby('model_short_name')['efficiency_score'].idxmax()].sort_values('efficiency_score', ascending=False)
print("✅ Data analysis complete.")

# ==============================================================================
# STEP 4: GENERATE ENHANCED VISUALIZATIONS
# ==============================================================================
print("\n*** Step 4: Generating visualizations ***")
# Define a consistent color palette for models
model_palette = sns.color_palette("viridis", n_colors=df['model_short_name'].nunique())
model_color_map = dict(zip(df['model_short_name'].unique(), model_palette))

# *** Visualization 1: Best Accuracy per Model (Bar Chart) ***
plt.figure(figsize=(16, 9))
bar_plot = sns.barplot(
    x='overall_accuracy',
    y='model_short_name',
    data=best_per_model_accuracy.sort_values('overall_accuracy', ascending=False),
    palette=model_palette,
    hue='model_short_name',
    dodge=False
)
plt.title('Peak Overall Accuracy by Model')
plt.xlabel('Best Overall Accuracy')
plt.ylabel('Model')
plt.xlim(0, 1)
plt.legend([],[], frameon=False) 

# Add clear labels to each bar
for i, (index, row) in enumerate(best_per_model_accuracy.sort_values('overall_accuracy', ascending=False).iterrows()):
    text = plt.text(row.overall_accuracy + 0.01, i, f'{row.overall_accuracy:.2%}',
                    va='center', ha='left', fontsize=18, fontweight='bold', color='black')
    text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

plt.tight_layout()
accuracy_plot_path = os.path.join(PLOTS_OUTPUT_DIR, '1_best_accuracy_per_model.png')
plt.savefig(accuracy_plot_path, dpi=SAVE_DPI, bbox_inches='tight')
print(f"✅ Saved Best Accuracy plot: {os.path.basename(accuracy_plot_path)}")
plt.close()


# *** Visualization 2: Consolidated Performance View (Scatter Plot) ***
plt.figure(figsize=(20, 12))
scatter_plot = sns.scatterplot(
    data=df,
    x='avg_response_time',
    y='overall_accuracy',
    hue='temperature',
    style='model_short_name',
    size='top_p',
    sizes=(50, 500),
    palette='magma',
    s=200
)

# Annotate the best point for each model to reduce clutter
for idx, row in best_per_model_accuracy.iterrows():
    text = plt.text(row['avg_response_time'], row['overall_accuracy'] + 0.01,
                    f"Best {row['model_short_name']}\n{row['overall_accuracy']:.2%}",
                    fontdict={'ha': 'center', 'size': 16, 'weight': 'bold'})
    text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

plt.title('Consolidated Model Performance: Accuracy vs. Time', fontsize=28, pad=20)
plt.xlabel('Average Response Time (seconds)')
plt.ylabel('Overall Accuracy')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.85, 1])

tradeoff_plot_path = os.path.join(PLOTS_OUTPUT_DIR, '2_consolidated_performance_view.png')
plt.savefig(tradeoff_plot_path, dpi=SAVE_DPI, bbox_inches='tight')
print(f"✅ Saved Consolidated Performance plot: {os.path.basename(tradeoff_plot_path)}")
plt.close()


# *** Visualization 3: Combined Performance Heatmap (Accuracy & Time) ***
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


# *** Visualization 4: Model Efficiency Score (Bar Chart) ***
plt.figure(figsize=(16, 9))
sns.barplot(
    x='efficiency_score',
    y='model_short_name',
    data=best_efficient_per_model,
    palette=model_palette,
    hue='model_short_name',
    dodge=False
)
plt.title('Model Efficiency (Accuracy per Second)')
plt.xlabel('Efficiency Score (Higher is Better)')
plt.ylabel('Model')
plt.legend([],[], frameon=False) 

for i, (index, row) in enumerate(best_efficient_per_model.iterrows()):
    text = plt.text(row.efficiency_score, i, f' {row.efficiency_score:.3f} ',
                    va='center', ha='left', fontsize=18, fontweight='bold', color='black')
    text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

plt.tight_layout()
efficiency_plot_path = os.path.join(PLOTS_OUTPUT_DIR, '4_model_efficiency_score.pdf')
plt.savefig(efficiency_plot_path, dpi=SAVE_DPI, bbox_inches='tight')
print(f"✅ Saved Efficiency Score plot: {os.path.basename(efficiency_plot_path)}")
plt.close()

print("\n🎉 All tasks complete. Visualizations are saved in the /results directory.")