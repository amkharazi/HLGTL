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
    'TEST_ID001': 'Base Model',    # Base
    'TEST_ID003': 'TRL Model',     # TRL
    'TEST_ID0012': 'Our Method'    # Our Method
}

# --- تغییرات خاص MNIST ---
NUM_SEEDS = 30       # تعداد سیدها: 30
TOTAL_EPOCHS = 35    # کل ایپاک‌ها: 35
UNFREEZE_EPOCH = 30  # 30 ایپاک اول فریز هستند (شروع آنفریز از 31)

# --- مسیر ذخیره سازی: MNIST ---
BASE_OUTPUT_ROOT = os.path.join(RESULTS_DIR, 'images', 'mnist')

# --- تنظیمات متریک‌ها ---
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
def plot_and_save(data_np, metric_key, range_type, shadow_type, scale_type, model_name, model_folder_name):
    config = METRICS_CONFIG[metric_key]
    
    # --- الف) برش داده‌ها (Slicing) ---
    if range_type == 'Full_Range':
        # کل 35 ایپاک
        idx_start = 0
        idx_end = TOTAL_EPOCHS
    else: # Unfrozen_Only
        # فقط 5 تای آخر (از ایپاک 31 تا 35)
        # ایندکس 30 متناظر با ایپاک 31 است
        idx_start = UNFREEZE_EPOCH 
        idx_end = TOTAL_EPOCHS
        
    sliced_data = data_np[:, idx_start:idx_end]
    
    # --- ب) تنظیم محور افقی (شروع از 1) ---
    num_points = sliced_data.shape[1] 
    x_epochs = list(range(1, num_points + 1))
    
    # --- ج) محاسبات آماری ---
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

    # --- د) رسم نمودار ---
    fig, ax = plt.subplots(figsize=(11, 7))
    # گرید بندی (برای Log Scale گزینه both مهم است تا خطوط ریز هم بیفتند)
    ax.grid(True, linestyle='--', alpha=0.5, which='both')
    
    col = config['color']
    
    # 1. سایه
    ax.fill_between(x_epochs, y_lower, y_upper, color=col, alpha=0.15, label=shadow_label)
    
    # 2. خط میانگین
    ax.plot(x_epochs, y_mean, color=col, linewidth=2, marker='o', markersize=5, label='Mean')
    
    # --- اعمال اسکیل (Linear یا Log) ---
    if scale_type == 'Log_Scale':
        ax.set_yscale('log')
    
    # 3. نوشتن اعداد روی نقاط
    y_span = np.max(y_upper) - np.min(y_lower)
    if y_span == 0: y_span = 1.0
    
    # آفست برای حالت خطی
    offset = y_span * 0.06 
    
    if scale_type == 'Log_Scale':
        # در حالت لگاریتمی آفست ضربی بهتر جواب می‌دهد تا جمعی
        for x, y, val in zip(x_epochs, y_mean, annotation_values):
            label = f"{val:.1e}"
            # متن کمی بالاتر از نقطه (1.05 برابر)
            ax.text(x, y * 1.05, label, fontsize=8, ha='center', va='bottom', color=text_color, rotation=45, fontweight='bold')
    else:
        # حالت خطی (Linear)
        for x, y, val in zip(x_epochs, y_mean, annotation_values):
            label = f"{val:.1e}"
            ax.text(x, y + offset, label, fontsize=8, ha='center', va='bottom', color=text_color, rotation=45, fontweight='bold')

    # --- ه) تنظیمات ظاهری و تیتر ---
    clean_shadow_name = shadow_type.replace('_', ' ')
    clean_scale_name = " (Log Scale)" if scale_type == 'Log_Scale' else ""
    
    # Title: Metric | Model Name | Shadow | (Log?)
    plot_title = f"{config['title']}{clean_scale_name} | {model_name} | {clean_shadow_name}"
    
    ax.set_title(plot_title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    ax.legend(loc='best')
    ax.set_xticks(x_epochs) 
    
    # تنظیم محدوده Y فقط برای حالت خطی (لگاریتمی خودکار بهتر است)
    if scale_type == 'Linear_Scale':
        ylim_top = np.max(y_upper) + (offset * 6)
        ylim_bottom = np.min(y_lower) - (offset * 3)
        if 'Loss' in metric_key and ylim_bottom < 0: ylim_bottom = 0
        ax.set_ylim(ylim_bottom, ylim_top)

    plt.tight_layout()
    
    # --- و) ذخیره سازی ---
    # ساختار پوشه: .../images/mnist/{Model}/{Scale}/{Range}/{Shadow}/filename.png
    save_dir = os.path.join(BASE_OUTPUT_ROOT, model_folder_name, scale_type, range_type, shadow_type)
    ensure_dir(save_dir)
    
    filename = f"{metric_key}_{range_type}_{shadow_type}.png"
    if scale_type == 'Log_Scale':
        filename = f"{metric_key}_{range_type}_{shadow_type}_LOG.png"
        
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    # print(f" Saved: {save_path}")

# ==========================================
# 4. بدنه اصلی
# ==========================================
if __name__ == "__main__":
    print(f"--- Generating Plots for All Models (MNIST) ---")
    print(f"Target Root Directory: {BASE_OUTPUT_ROOT}")
    print(f"Settings: Seeds={NUM_SEEDS}, Total Epochs={TOTAL_EPOCHS}, Unfreeze at={UNFREEZE_EPOCH}\n")
    
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
                    # اضافه شدن حلقه برای Linear و Log
                    for scale in ['Linear_Scale', 'Log_Scale']:
                        plot_and_save(data, metric_name, r_type, s_type, scale, model_name, model_folder_name)

    print("\n\n--- All MNIST Models (Linear & Log) Processed Successfully! ---")