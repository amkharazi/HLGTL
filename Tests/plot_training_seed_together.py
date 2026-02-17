import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# تنظیمات اصلی (مسیرها و شناسه‌ها را چک کنید)
# ==========================================
RESULTS_DIR = '../results'   # مسیر پوشه نتایج
NUM_SEEDS = 30               # تعداد Seed ها
NUM_EPOCHS = 35              # کل اپوک‌ها (۳۰ فریز + ۵ آنفریز)

# شناسه‌های پوشه‌های مدل‌های شما
ID_BASE = 'TEST_ID001'       # Test 1 (Base Model)
ID_TCL  = 'TEST_ID003'       # Test 3 (TCL/TRL Model)
ID_OURS = 'TEST_ID0012'       # Test 12 (Our Methods) - فرض بر این است که نام پوشه 012 دارد

# ==========================================
# توابع کمکی اصلاح شده
# ==========================================

def parse_accuracy_reports(base_dir, folder_prefix, num_seeds, epochs, mode='test'):
    """
    اطلاعات دقت را استخراج می‌کند.
    mode: می‌تواند 'train' یا 'test' باشد.
    folder_prefix: ابتدای نام پوشه (مثلا TEST_ID001)
    """
    # انتخاب الگوی Regex بر اساس مود (Train یا Test)
    if mode == 'train':
        # دنبال الگویی مثل: Train epoch 1: ... top1=95.5%
        pattern = r"Train epoch \d+:.*?top1=([\d\.]+)"
        print(f"--- Extracting TRAINING data for {folder_prefix} ---")
    elif mode == 'test':
        # دنبال الگویی مثل: Test epoch 1: ... top1=94.2%
        pattern = r"Test epoch \d+:.*?top1=([\d\.]+)"
        print(f"--- Extracting TESTING data for {folder_prefix} ---")
    else:
        raise ValueError("Mode parameter must be 'train' or 'test'")

    all_accuracies = np.zeros((num_seeds, epochs))
    # با NaN پر می‌کنیم تا اگر سیدی ناقص بود، میانگین خراب نشود
    all_accuracies[:] = np.nan 
    
    found_count = 0
    for seed in range(num_seeds):
        # ساخت نام پوشه به صورت داینامیک
        folder_name = f'{folder_prefix}_SEED_{seed}'
        file_path = os.path.join(base_dir, folder_name, 'accuracy_stats', 'report.txt')
        
        if not os.path.exists(file_path):
            # print(f"Warning: File not found: {file_path}") # برای خلوت شدن خروجی کامنت شد
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                matches = re.findall(pattern, content)
                
                # تبدیل رشته به عدد
                accs = [float(m) for m in matches]
                
                # اگر اعداد در فایل لاگ شما بین 0 و 1 هستند (مثل 0.95)، خط زیر را فعال کنید:
                # accs = [a * 100 for a in accs]
                # اما طبق کدی که فرستادید، اعداد شما قبلاً درصد هستند (مثل 95.5)، پس نیازی نیست.
                
                length = len(accs)
                if length > 0:
                    found_count += 1
                    if length >= epochs:
                        all_accuracies[seed, :] = accs[:epochs]
                    else:
                        all_accuracies[seed, :length] = accs
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if found_count == 0:
         print(f"CRITICAL WARNING: No valid {mode} data found for prefix: {folder_prefix}!")

    return all_accuracies

def plot_mean_std(data_dict, title, save_name, ylabel='Accuracy (%)'):
    """
    رسم نمودار مقایسه‌ای با نوار انحراف معیار
    """
    plt.figure(figsize=(10, 7))
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.minorticks_on()
    
    # لیست رنگ‌ها برای تمایز مدل‌ها
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    
    for i, (label, data) in enumerate(data_dict.items()):
        # محاسبه میانگین و انحراف معیار (با نادیده گرفتن NaN ها)
        mean_acc = np.nanmean(data, axis=0)
        std_acc = np.nanstd(data, axis=0)
        
        if np.isnan(mean_acc).all():
             print(f"Skipping plot for {label} due to missing data.")
             continue
             
        epochs = range(1, len(mean_acc) + 1)
        color = colors[i % len(colors)]
        
        # رسم خط میانگین
        plt.plot(epochs, mean_acc, label=label, color=color, linewidth=2.5)
        
        # رسم سایه (Std Dev)
        plt.fill_between(epochs, 
                         mean_acc - std_acc, 
                         mean_acc + std_acc, 
                         color=color, alpha=0.2)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.tight_layout()
    
    plt.savefig(save_name, dpi=300)
    print(f"\nPlot saved successfully as: {save_name}")
    # plt.show() # اگر روی سرور هستید این خط را کامنت کنید

# ==========================================
# بخش اصلی اجرا (Main)
# ==========================================
if __name__ == "__main__":
    print(">>> Starting Process for TRAINING Data <<< \n")

    # 1. استخراج داده‌های TRAINING برای هر سه مدل
    # نکته مهم: پارامتر mode='train' ارسال شده است
    
    train_data_base = parse_accuracy_reports(RESULTS_DIR, ID_BASE, NUM_SEEDS, NUM_EPOCHS, mode='train')
    train_data_tcl  = parse_accuracy_reports(RESULTS_DIR, ID_TCL, NUM_SEEDS, NUM_EPOCHS, mode='train')
    train_data_ours = parse_accuracy_reports(RESULTS_DIR, ID_OURS, NUM_SEEDS, NUM_EPOCHS, mode='train')
    
    # 2. آماده‌سازی دیکشنری برای رسم
    # کلیدها (Keys) نام‌هایی هستند که در راهنمای نمودار (Legend) نمایش داده می‌شوند.
    plot_data_train = {
        'Base Model (ResNet50)': train_data_base,
        'TCL/TRL Model': train_data_tcl,
        'Our Proposed Method': train_data_ours
    }
    
    # 3. رسم نمودار و ذخیره
    plot_mean_std(
        plot_data_train, 
        title='Training Accuracy Comparison (30 Random Seeds)', 
        save_name='training_comparison.png', # نام فایل خروجی
        ylabel='Training Accuracy (%)'
    )
    
    print("\nProcess Complete.")
