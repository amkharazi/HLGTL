import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# تنظیمات اصلی (اینجا را چک کنید)
# ==========================================
RESULTS_DIR = '../results'   # مسیر پوشه نتایج
NUM_SEEDS = 30               # تعداد Seed ها
NUM_EPOCHS = 35              # کل اپوک‌ها (مثلا ۳۰ تا فریز + ۵ تا آنفریز)

# نام پوشه‌های تست‌ها (بر اساس الگوی نام‌گذاری کدهای قبلی شما)
# اگر نام پوشه شما مثلا TEST_ID12 است (بدون صفر)، اینجا را تغییر دهید
PREFIX_TEST_1  = 'TEST_ID001'
PREFIX_TEST_3  = 'TEST_ID003'
PREFIX_TEST_12 = 'TEST_ID0012' # فرض بر این است که 012 نامگذاری شده

# ==========================================
# توابع کمکی
# ==========================================

def parse_accuracy_reports(base_dir, folder_prefix, num_seeds, epochs):
    """
    اطلاعات دقت (Accuracy) را از فایل‌های متنی استخراج می‌کند.
    """
    all_accuracies = np.zeros((num_seeds, epochs))
    # پر کردن با NaN تا اگر دیتایی نبود، نمودار خراب نشود (اختیاری)
    all_accuracies[:] = np.nan 

    print(f"--- Reading data for {folder_prefix} ---")
    
    found_any = False
    for seed in range(num_seeds):
        # ساخت نام پوشه: مثلا TEST_ID001_SEED_0
        folder_name = f'{folder_prefix}_SEED_{seed}'
        file_path = os.path.join(base_dir, folder_name, 'accuracy_stats', 'report.txt')
        
        if not os.path.exists(file_path):
            # اگر فایلی پیدا نشد، فقط رد می‌شویم (ممکن است هنوز ران نشده باشد)
            continue
            
        found_any = True
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
                # پیدا کردن اعداد top1
                pattern = r"Test epoch \d+:.*?top1=([\d\.]+)"
                matches = re.findall(pattern, content)
                
                # تبدیل به درصد
                accs = [float(m) * 100 for m in matches]
                
                # ذخیره در ماتریس
                length = len(accs)
                if length > 0:
                    # اگر تعداد اپوک‌ها بیشتر از حد انتظار بود، برش می‌زنیم
                    if length >= epochs:
                        all_accuracies[seed, :] = accs[:epochs]
                    else:
                        # اگر کمتر بود، تا همانجا که هست پر می‌کنیم
                        all_accuracies[seed, :length] = accs
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not found_any:
        print(f"WARNING: No data found for {folder_prefix}. Check folder names!")
        
    return all_accuracies

def plot_mean_std(data_dict, title, save_name):
    """
    رسم نمودار میانگین و انحراف معیار
    """
    plt.figure(figsize=(10, 7))
    
    # تنظیمات ظاهری
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.minorticks_on()
    
    # لیست رنگ‌ها و استایل‌ها
    styles = [
        {'color': '#1f77b4', 'fmt': '-'},  # آبی برای مدل اول
        {'color': '#ff7f0e', 'fmt': '-'},  # نارنجی برای مدل دوم
        {'color': '#2ca02c', 'fmt': '-'},  # سبز برای مدل سوم (روش پیشنهادی)
    ]
    
    for i, (label, data) in enumerate(data_dict.items()):
        # حذف ردیف‌هایی که کلاً NaN هستند (Seedهایی که ران نشدند)
        # استفاده از nanmean و nanstd برای نادیده گرفتن مقادیر خالی
        mean_acc = np.nanmean(data, axis=0)
        std_acc = np.nanstd(data, axis=0)
        
        # اگر همه دیتا NaN باشد، mean_acc هم NaN می‌شود، پس رسم نمی‌کنیم
        if np.isnan(mean_acc).all():
            print(f"Skipping plot for {label} (No valid data)")
            continue

        epochs = range(1, len(mean_acc) + 1)
        style = styles[i % len(styles)]
        
        # رسم خط میانگین
        plt.plot(epochs, mean_acc, label=label, color=style['color'], linestyle=style['fmt'], linewidth=2)
        
        # رسم سایه (انحراف معیار)
        plt.fill_between(epochs, 
                         mean_acc - std_acc, 
                         mean_acc + std_acc, 
                         color=style['color'], alpha=0.15)

    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel('Epochs', fontsize=13)
    plt.ylabel('Test Accuracy (%)', fontsize=13)
    
    # تنظیم محدوده محور Y برای زیبایی (اختیاری - بر اساس دیتای MNIST)
    # plt.ylim(80, 100) 
    
    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.tight_layout()
    
    # ذخیره فایل
    plt.savefig(save_name, dpi=300)
    print(f"\nPlot saved successfully as: {save_name}")
    plt.show()

# ==========================================
# بخش اصلی اجرا
# ==========================================
if __name__ == "__main__":
    
    # 1. خواندن داده‌ها
    data_test1 = parse_accuracy_reports(RESULTS_DIR, PREFIX_TEST_1, NUM_SEEDS, NUM_EPOCHS)
    data_test3 = parse_accuracy_reports(RESULTS_DIR, PREFIX_TEST_3, NUM_SEEDS, NUM_EPOCHS)
    data_test12 = parse_accuracy_reports(RESULTS_DIR, PREFIX_TEST_12, NUM_SEEDS, NUM_EPOCHS)
    
    # 2. آماده‌سازی دیکشنری برای رسم
    # کلیدها (Keys) همان متن‌هایی هستند که در راهنمای نمودار (Legend) نمایش داده می‌شوند
    plot_data = {
        'Base Model': data_test1,
        'TCL/TRL Model': data_test3,
        'Our Methods': data_test12
    }
    
    # 3. رسم و ذخیره
    plot_mean_std(plot_data, 
                  title='Comparison of Proposed Method vs Baselines (30 Seeds)', 
                  save_name='final_comparison_plot.png')
