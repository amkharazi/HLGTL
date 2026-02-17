import os
import re
import numpy as np
import matplotlib
<<<<<<< HEAD
matplotlib.use('Agg') # حالت بدون نمایشگر (مناسب سرور)
=======
matplotlib.use('Agg') # جلوگیری از ارور گرافیکی (Headless Mode)
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
import matplotlib.pyplot as plt

# ==========================================
# 1. تنظیمات اصلی (Configuration)
# ==========================================
RESULTS_DIR = '../results'
<<<<<<< HEAD

# لیست مدل‌ها و نام‌های نمایشی آن‌ها
MODELS_MAPPING = {
    'TEST_ID0025': 'Base Model',
    'TEST_ID0028': 'TRL Model',
    'TEST_ID0032': 'Our Method'
}

NUM_SEEDS = 16
TOTAL_EPOCHS = 15
UNFREEZE_EPOCH = 5

# مسیر ذخیره سازی پایه
BASE_OUTPUT_ROOT = os.path.join(RESULTS_DIR, 'images', 'tiny_image_net')

# تنظیمات متریک‌ها
=======
SEED_PREFIX = 'TEST_ID0025'
NUM_SEEDS = 16
TOTAL_EPOCHS = 15
UNFREEZE_EPOCH = 5  # نقطه تغییر فاز

# --- تغییر مسیر ذخیره‌سازی ---
# مسیر: ../results/images/tiny_image_net
BASE_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'images', 'tiny_image_net')

# --- تغییر تیترها (Tiny ImageNet) ---
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
METRICS_CONFIG = {
    'Train_Accuracy': {
        'pattern': r"Train epoch \d+:.*?top1=([\d\.]+)",
        'color': 'blue',
        'ylabel': 'Train Accuracy (%)',
<<<<<<< HEAD
        'title': 'Train Accuracy (Tiny ImageNet)'
=======
        'title': 'Train Accuracy (Tiny ImageNet)' 
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
    },
    'Train_Loss': {
        'pattern': r"Train epoch \d+:.*?loss=([\d\.]+)",
        'color': 'blue',
        'ylabel': 'Train Loss',
        'title': 'Train Loss (Tiny ImageNet)'
    },
    'Test_Accuracy': {
        'pattern': r"Test epoch \d+:.*?top1=([\d\.]+)",
        'color': 'darkorange',
        'ylabel': 'Test Accuracy (%)',
        'title': 'Test Accuracy (Tiny ImageNet)'
    },
    'Test_Loss': {
        'pattern': r"Test epoch \d+:.*?loss=([\d\.]+)",
        'color': 'darkorange',
        'ylabel': 'Test Loss',
        'title': 'Test Loss (Tiny ImageNet)'
    }
}

# ==========================================
# 2. توابع کمکی
# ==========================================
<<<<<<< HEAD
def get_data_for_metric(metric_name, seed_prefix):
    """
    داده‌ها را برای یک متریک خاص و یک مدل خاص (seed_prefix) می‌خواند.
    """
=======
def get_data_for_metric(metric_name):
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
    config = METRICS_CONFIG[metric_name]
    pattern_str = config['pattern']
    all_seeds_data = []
    
    for seed in range(NUM_SEEDS):
<<<<<<< HEAD
        folder_name = f'{seed_prefix}_SEED_{seed}'
=======
        folder_name = f'{SEED_PREFIX}_SEED_{seed}'
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
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
<<<<<<< HEAD
def plot_and_save(data_np, metric_key, range_type, shadow_type, model_name, model_folder_name):
    config = METRICS_CONFIG[metric_key]
    
    # --- الف) برش داده‌ها (Slicing) ---
    if range_type == 'Full_Range':
        idx_start = 0
        idx_end = 15
    else: # Unfrozen_Only
=======
def plot_and_save(data_np, metric_key, range_type, shadow_type):
    config = METRICS_CONFIG[metric_key]
    
    # --- الف) تعیین بازه زمانی ---
    if range_type == 'Full_Range':
        start_epoch = 1
        end_epoch = 15
        idx_start = 0
        idx_end = 15
    else: # Unfrozen_Only
        start_epoch = 6
        end_epoch = 15
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
        idx_start = 5
        idx_end = 15
        
    sliced_data = data_np[:, idx_start:idx_end]
<<<<<<< HEAD
    
    # --- ب) تنظیم محور افقی (شروع از 1) ---
    num_points = sliced_data.shape[1] 
    x_epochs = list(range(1, num_points + 1))
    
    # --- ج) محاسبات آماری ---
=======
    x_epochs = list(range(start_epoch, end_epoch + 1))
    
    # --- ب) محاسبات آماری ---
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
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

<<<<<<< HEAD
    # --- د) رسم نمودار ---
=======
    # --- ج) رسم نمودار ---
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.grid(True, linestyle='--', alpha=0.5)
    
    col = config['color']
    
    # 1. سایه
    ax.fill_between(x_epochs, y_lower, y_upper, color=col, alpha=0.15, label=shadow_label)
    
    # 2. خط میانگین
    ax.plot(x_epochs, y_mean, color=col, linewidth=2, marker='o', markersize=5, label='Mean')
    
<<<<<<< HEAD
    # 3. نوشتن اعداد روی نقاط
=======
    # 3. خط جداکننده (فقط برای Full Range)
    if range_type == 'Full_Range':
        ax.axvline(x=UNFREEZE_EPOCH + 0.5, color='gray', linestyle='--', linewidth=1.5)
        y_lim_text = np.max(y_upper)
        ax.text(UNFREEZE_EPOCH - 1, y_lim_text, 'Frozen', color='gray', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(UNFREEZE_EPOCH + 2, y_lim_text, 'Unfrozen', color='gray', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. نوشتن اعداد
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
    y_span = np.max(y_upper) - np.min(y_lower)
    if y_span == 0: y_span = 1.0
    offset = y_span * 0.06 
    
    for x, y, val in zip(x_epochs, y_mean, annotation_values):
        label = f"{val:.1e}"
        ax.text(x, y + offset, label, fontsize=8, ha='center', va='bottom', color=text_color, rotation=45, fontweight='bold')

<<<<<<< HEAD
    # --- ه) تنظیمات ظاهری و تیتر ---
    
    clean_shadow_name = shadow_type.replace('_', ' ')
    
    # تیتر شامل: نام متریک | نام مدل | نوع سایه
    plot_title = f"{config['title']} | {model_name} | {clean_shadow_name}"
    
    ax.set_title(plot_title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12) # همیشه Epochs
    ax.set_ylabel(config['ylabel'], fontsize=12)
    ax.legend(loc='best')
    ax.set_xticks(x_epochs) 
    
    # تنظیم محدوده Y
=======
    # --- د) تنظیمات ظاهری ---
    # ترکیب تیتر متریک + نوع بازه + نوع سایه
    plot_title = f"{config['title']} | {range_type.replace('_', ' ')} | {shadow_type.replace('_', ' ')}"
    
    ax.set_title(plot_title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    ax.legend(loc='best')
    ax.set_xticks(x_epochs)
    
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
    ylim_top = np.max(y_upper) + (offset * 6)
    ylim_bottom = np.min(y_lower) - (offset * 3)
    if 'Loss' in metric_key and ylim_bottom < 0: ylim_bottom = 0
    ax.set_ylim(ylim_bottom, ylim_top)

    plt.tight_layout()
    
<<<<<<< HEAD
    # --- و) ذخیره سازی ---
    # ساختار: ../results/images/tiny_image_net/{Model_Name}/Full_Range/Min_Max/filename.png
    
    save_dir = os.path.join(BASE_OUTPUT_ROOT, model_folder_name, range_type, shadow_type)
=======
    # --- ه) ذخیره سازی ---
    # ساخت مسیر: ../results/images/tiny_image_net/Full_Range/Min_Max/filename.png
    save_dir = os.path.join(BASE_OUTPUT_DIR, range_type, shadow_type)
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
    ensure_dir(save_dir)
    
    filename = f"{metric_key}_{range_type}_{shadow_type}.png"
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" Saved: {save_path}")

# ==========================================
<<<<<<< HEAD
# 4. بدنه اصلی (Main Loop)
# ==========================================
if __name__ == "__main__":
    print(f"--- Generating Plots for All Models ---")
    print(f"Target Root Directory: {BASE_OUTPUT_ROOT}\n")
    
    # ==========================================
    # حلقه روی مدل‌ها (Test IDs)
    # ==========================================
    for seed_prefix, model_name in MODELS_MAPPING.items():
        print(f"\n==========================================")
        print(f" Processing: {model_name} ({seed_prefix})")
        print(f"==========================================")
        
        # تبدیل نام مدل به فرمت مناسب پوشه (جایگزینی فاصله با آندرلاین)
        model_folder_name = model_name.replace(" ", "_")
        
        # حلقه روی متریک‌ها
        for metric_name in METRICS_CONFIG.keys():
            print(f"  > Metric: {metric_name}...")
            
            # دریافت دیتا با استفاده از seed_prefix متغیر
            data = get_data_for_metric(metric_name, seed_prefix)
            
            if data is None:
                print(f"    [!] Skipping {metric_name} (No Data Found for {seed_prefix})")
                continue
                
            # حلقه روی بازه‌ها
            for r_type in ['Full_Range', 'Unfrozen_Only']:
                # حلقه روی سایه‌ها
                for s_type in ['Min_Max', 'Std_Dev']:
                    plot_and_save(data, metric_name, r_type, s_type, model_name, model_folder_name)

    print("\n\n--- All Models Processed Successfully! ---")
=======
# 4. بدنه اصلی
# ==========================================
if __name__ == "__main__":
    print(f"--- Generating 16 Plots for {SEED_PREFIX} ---")
    print(f"Target Directory: {BASE_OUTPUT_DIR}\n")
    
    # 1. Loop Metrics
    for metric_name in METRICS_CONFIG.keys():
        print(f"Processing Metric: {metric_name}...")
        
        data = get_data_for_metric(metric_name)
        if data is None:
            print(f"  -> Skipping {metric_name} (No Data)")
            continue
            
        # 2. Loop Range (Full vs Unfrozen)
        for r_type in ['Full_Range', 'Unfrozen_Only']:
            
            # 3. Loop Shadow (MinMax vs Std)
            for s_type in ['Min_Max', 'Std_Dev']:
                
                plot_and_save(data, metric_name, r_type, s_type)

    print("\n--- All Done! ---")
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
