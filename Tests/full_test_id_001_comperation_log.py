import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg') # حالت بدون نمایشگر
import matplotlib.pyplot as plt

# ==========================================
# 1. تنظیمات اصلی
# ==========================================
RESULTS_DIR = '../results'

# --- تنظیمات مدل‌های MNIST ---
MODELS_CONFIG = {
    'TEST_ID001':  {'label': 'Base Model', 'color': 'gray',      'style': '--'}, 
    'TEST_ID003':  {'label': 'TRL Model',  'color': 'blue',      'style': '-.'}, 
    'TEST_ID0012': {'label': 'Our Method', 'color': 'red',       'style': '-'}   
}

# --- تغییرات MNIST ---
NUM_SEEDS = 30       # تعداد سیدها
TOTAL_EPOCHS = 35    # تعداد کل ایپاک‌ها
UNFREEZE_EPOCH = 30  # نقطه شروع آنفریز

# مسیر ذخیره سازی: MNIST -> Comparison
BASE_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'images', 'mnist', 'Comparison')

# تنظیمات متریک‌ها
METRICS_CONFIG = {
    'Train_Accuracy': {
        'pattern': r"Train epoch \d+:.*?top1=([\d\.]+)",
        'ylabel': 'Train Accuracy (%)',
        'title': 'Train Accuracy Comparison (MNIST)'
    },
    'Train_Loss': {
        'pattern': r"Train epoch \d+:.*?loss=([\d\.]+)",
        'ylabel': 'Train Loss',
        'title': 'Train Loss Comparison (MNIST)'
    },
    'Test_Accuracy': {
        'pattern': r"Test epoch \d+:.*?top1=([\d\.]+)",
        'ylabel': 'Test Accuracy (%)',
        'title': 'Test Accuracy Comparison (MNIST)'
    },
    'Test_Loss': {
        'pattern': r"Test epoch \d+:.*?loss=([\d\.]+)",
        'ylabel': 'Test Loss',
        'title': 'Test Loss Comparison (MNIST)'
    }
}

# ==========================================
# 2. تابع خواندن دیتا (فقط میانگین)
# ==========================================
def get_model_mean_data(metric_key, seed_prefix):
    """
    میانگین دیتای یک مدل خاص را برای تمام سیدها برمی‌گرداند.
    """
    config = METRICS_CONFIG[metric_key]
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
        
    return np.mean(np.array(all_seeds_data), axis=0)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ==========================================
# 3. تابع رسم نمودار مقایسه‌ای
# ==========================================
def plot_comparison(metric_key, range_type, scale_type):
    config = METRICS_CONFIG[metric_key]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # گرید بندی (برای Log مهم است که both باشد)
    ax.grid(True, linestyle='--', alpha=0.4, which='both')
    
    # تعیین بازه
    if range_type == 'Full_Range':
        idx_start = 0
        idx_end = TOTAL_EPOCHS
    else: # Unfrozen_Only
        idx_start = UNFREEZE_EPOCH
        idx_end = TOTAL_EPOCHS
        
    plotted_any = False
    
    # --- حلقه روی مدل‌ها ---
    for seed_prefix, model_info in MODELS_CONFIG.items():
        
        # 1. گرفتن میانگین دیتا
        mean_data = get_model_mean_data(metric_key, seed_prefix)
        
        if mean_data is None:
            print(f"  [Warning] No data for {model_info['label']} ({seed_prefix})")
            continue
            
        # 2. برش دیتا
        sliced_data = mean_data[idx_start:idx_end]
        
        # 3. محور X
        x_epochs = list(range(1, len(sliced_data) + 1))
        
        # 4. رسم خط (بدون سایه)
        ax.plot(x_epochs, sliced_data, 
                label=model_info['label'], 
                color=model_info['color'], 
                linestyle=model_info['style'], 
                linewidth=2.5 if 'Our' in model_info['label'] else 2, 
                marker='o', markersize=5)
        
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    # --- تنظیمات Log Scale ---
    if scale_type == 'Log_Scale':
        ax.set_yscale('log')

    # --- تنظیمات ظاهری ---
    clean_scale_name = "(Log Scale)" if scale_type == 'Log_Scale' else ""
    plot_title = f"{config['title']} {clean_scale_name} | {range_type.replace('_', ' ')}"
    
    ax.set_title(plot_title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    
    if len(x_epochs) > 0:
        ax.set_xticks(x_epochs)
    
    ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    
    plt.tight_layout()
    
    # --- ذخیره سازی ---
    # ساختار: .../mnist/Comparison/{Scale_Type}/{Range_Type}/filename.png
    save_dir = os.path.join(BASE_OUTPUT_DIR, scale_type, range_type)
    ensure_dir(save_dir)
    
    filename = f"Compare_{metric_key}_{range_type}.png"
    if scale_type == 'Log_Scale':
         filename = f"Compare_{metric_key}_{range_type}_LOG.png"

    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    # print(f" Saved: {save_path}")

# ==========================================
# 4. اجرا
# ==========================================
if __name__ == "__main__":
    print(f"--- Generating Comparison Plots (MNIST) ---")
    print(f"Output Root: {BASE_OUTPUT_DIR}")
    print(f"Config: Seeds={NUM_SEEDS}, Epochs={TOTAL_EPOCHS}, Unfreeze={UNFREEZE_EPOCH}\n")
    
    for metric_name in METRICS_CONFIG.keys():
        print(f"Processing Metric: {metric_name}...")
        
        # حلقه روی بازه‌ها
        for r_type in ['Full_Range', 'Unfrozen_Only']:
            # حلقه روی نوع اسکیل (خطی و لگاریتمی)
            for scale in ['Linear_Scale', 'Log_Scale']:
                plot_comparison(metric_name, r_type, scale)
            
    print("\n--- All MNIST Comparison Plots (Linear & Log) Created! ---")