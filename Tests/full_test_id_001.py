import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg') # حالت بدون نمایشگر (مناسب سرور)
import matplotlib.pyplot as plt

# ==========================================
# 1. تنظیمات اصلی (Configuration)
# ==========================================
RESULTS_DIR = '../results'

# --- لیست مدل‌ها برای MNIST ---
MODELS_MAPPING = {
    'TEST_ID001': 'Base Model',   # Base
    'TEST_ID003': 'TRL Model',    # TRL
    'TEST_ID0012': 'Our Method'    # Our Method
}

NUM_SEEDS = 16
TOTAL_EPOCHS = 15
UNFREEZE_EPOCH = 5

# --- مسیر ذخیره سازی: MNIST ---
BASE_OUTPUT_ROOT = os.path.join(RESULTS_DIR, 'images', 'mnist')

# --- تنظیمات متریک‌ها (تیترها به MNIST تغییر کرد) ---
METRICS_CONFIG = {
    'Train_Accuracy': {
        'pattern': r"Train epoch \d+:.*?top1=([\d\.]+)",
        'color': 'blue',
        'ylabel': 'Train Accuracy (%)',
        'title': 'Train Accuracy (MNIST)'
    },
    'Train_Loss': {
        'pattern': r"Train epoch \d+:.*?loss=([\d\.]+)",
        'color': 'blue',
        'ylabel': 'Train Loss',
        'title': 'Train Loss (MNIST)'
    },
    'Test_Accuracy': {
        'pattern': r"Test epoch \d+:.*?top1=([\d\.]+)",
        'color': 'darkorange',
        'ylabel': 'Test Accuracy (%)',
        'title': 'Test Accuracy (MNIST)'
    },
    'Test_Loss': {
        'pattern': r"Test epoch \d+:.*?loss=([\d\.]+)",
        'color': 'darkorange',
        'ylabel': 'Test Loss',
        'title': 'Test Loss (MNIST)'
    }
}

# ==========================================
# 2. توابع کمکی
# ==========================================
def get_data_for_metric(metric_name, seed_prefix):
    config = METRICS_CONFIG[metric_name]
    pattern_str = config['pattern']
    all_seeds_data = []
    
    for seed in range(NUM_SEEDS):
        folder_name = f'{seed_prefix}_SEED_{seed}'
        file_path = os.path.join(RESULTS_DIR, folder_name, 'accuracy_stats', 'report.txt')
        
        if not os.path.exists(file_path):
            continue 
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                matches = re.findall(pattern_str, content)
                values = [float(m) for m in matches]
                if len(values) >= TOTAL_EPOCHS:
                    all_seeds_data.append(values[:TOTAL_EPOCHS])
        except Exception:
            pass

    if not all_seeds_data:
        return None
    return np.array(all_seeds_data)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ==========================================
# 3. تابع رسم نمودار
# ==========================================
def plot_and_save(data_np, metric_key, range_type, shadow_type, model_name, model_folder_name):
    config = METRICS_CONFIG[metric_key]
    
    # الف) برش داده‌ها
    if range_type == 'Full_Range':
        idx_start = 0
        idx_end = 15
    else: # Unfrozen_Only
        idx_start = 5
        idx_end = 15
        
    sliced_data = data_np[:, idx_start:idx_end]
    
    # ب) تنظیم محور افقی (شروع از 1)
    num_points = sliced_data.shape[1] 
    x_epochs = list(range(1, num_points + 1))
    
    # ج) محاسبات آماری
    y_mean = np.mean(sliced_data, axis=0)
    
    if shadow_type == 'Min_Max':
        y_lower = np.min(sliced_data, axis=0)
        y_upper = np.max(sliced_data, axis=0)
        annotation_values = y_upper - y_lower
        shadow_label = 'Range (Min-Max)'
        text_color = 'darkred'
    else: # Std_Dev
        y_std = np.std(sliced_data, axis=0)
        y_lower = y_mean - y_std
        y_upper = y_mean + y_std
        annotation_values = y_std
        shadow_label = 'Range (Mean ± Std)'
        text_color = 'darkgreen'

    # د) رسم نمودار
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.grid(True, linestyle='--', alpha=0.5)
    
    col = config['color']
    
    # 1. سایه
    ax.fill_between(x_epochs, y_lower, y_upper, color=col, alpha=0.15, label=shadow_label)
    
    # 2. خط میانگین
    ax.plot(x_epochs, y_mean, color=col, linewidth=2, marker='o', markersize=5, label='Mean')
    
    # 3. نوشتن اعداد
    y_span = np.max(y_upper) - np.min(y_lower)
    if y_span == 0: y_span = 1.0
    offset = y_span * 0.06 
    
    for x, y, val in zip(x_epochs, y_mean, annotation_values):
        label = f"{val:.1e}"
        ax.text(x, y + offset, label, fontsize=8, ha='center', va='bottom', color=text_color, rotation=45, fontweight='bold')

    # ه) تنظیمات ظاهری
    clean_shadow_name = shadow_type.replace('_', ' ')
    
    # Title: Metric | Model Name | Shadow Type
    plot_title = f"{config['title']} | {model_name} | {clean_shadow_name}"
    
    ax.set_title(plot_title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    ax.legend(loc='best')
    ax.set_xticks(x_epochs) 
    
    ylim_top = np.max(y_upper) + (offset * 6)
    ylim_bottom = np.min(y_lower) - (offset * 3)
    if 'Loss' in metric_key and ylim_bottom < 0: ylim_bottom = 0
    ax.set_ylim(ylim_bottom, ylim_top)

    plt.tight_layout()
    
    # و) ذخیره سازی
    # مسیر: ../results/images/mnist/{Model_Name}/{Range_Type}/{Shadow_Type}/
    
    save_dir = os.path.join(BASE_OUTPUT_ROOT, model_folder_name, range_type, shadow_type)
    ensure_dir(save_dir)
    
    filename = f"{metric_key}_{range_type}_{shadow_type}.png"
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" Saved: {save_path}")

# ==========================================
# 4. بدنه اصلی
# ==========================================
if __name__ == "__main__":
    print(f"--- Generating Plots for All Models (MNIST) ---")
    print(f"Target Root Directory: {BASE_OUTPUT_ROOT}\n")
    
    for seed_prefix, model_name in MODELS_MAPPING.items():
        print(f"\n==========================================")
        print(f" Processing: {model_name} ({seed_prefix})")
        print(f"==========================================")
        
        # تبدیل نام مدل به فرمت پوشه
        model_folder_name = model_name.replace(" ", "_")
        
        for metric_name in METRICS_CONFIG.keys():
            print(f"  > Metric: {metric_name}...")
            
            data = get_data_for_metric(metric_name, seed_prefix)
            
            if data is None:
                print(f"    [!] Skipping {metric_name} (No Data Found for {seed_prefix})")
                continue
            
            for r_type in ['Full_Range', 'Unfrozen_Only']:
                for s_type in ['Min_Max', 'Std_Dev']:
                    plot_and_save(data, metric_name, r_type, s_type, model_name, model_folder_name)

    print("\n\n--- All MNIST Models Processed Successfully! ---")