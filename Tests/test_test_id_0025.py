import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# تنظیمات
# ==========================================
RESULTS_DIR = '../results'
SEED_PREFIX = 'TEST_ID0025'

START_EPOCH = 1    
END_EPOCH = 15     
NUM_SEEDS = 16     

# ==========================================
# توابع
# ==========================================
def get_test_accuracy(folder_name):
    """
    این تابع داده‌های Test Accuracy را می‌خواند.
    """
    file_path = os.path.join(RESULTS_DIR, folder_name, 'accuracy_stats', 'report.txt')
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # --- تغییر کلیدی: جستجو برای Test epoch ---
            pattern = r"Test epoch \d+:.*?top1=([\d\.]+)"
            matches = re.findall(pattern, content)
            accs = [float(m) for m in matches]
            return accs
    except Exception as e:
        print(f"Error reading {folder_name}: {e}")
        return []

# ==========================================
# اجرای اصلی
# ==========================================
if __name__ == "__main__":
    
    all_seeds_data = []
    print(f"Processing {NUM_SEEDS} seeds for {SEED_PREFIX} (TEST DATA)...")

    # جمع‌آوری داده‌ها
    for seed in range(NUM_SEEDS):
        folder_name = f'{SEED_PREFIX}_SEED_{seed}'
        # فراخوانی تابع جدید برای تست
        accs = get_test_accuracy(folder_name)
        if len(accs) >= END_EPOCH:
            all_seeds_data.append(accs[:END_EPOCH])

    if len(all_seeds_data) > 0:
        data_np = np.array(all_seeds_data)
        
        # برش زمانی (ایپاک 6 تا 15)
        idx_start = START_EPOCH - 1
        idx_end = END_EPOCH 
        sliced_data = data_np[:, idx_start:idx_end]
        
        # محاسبات
        y_mean = np.mean(sliced_data, axis=0)
        y_min = np.min(sliced_data, axis=0)
        y_max = np.max(sliced_data, axis=0)
        y_var = np.var(sliced_data, axis=0)
        
        x_epochs = list(range(START_EPOCH, END_EPOCH + 1))

        # --- شروع رسم ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.grid(True, linestyle='--', alpha=0.5)

        # 1. رسم بازه (سایه) - رنگ قرمز برای تست (اختیاری)
        # معمولا تست را با رنگ متفاوت (مثل نارنجی یا قرمز) نشان می‌دهند، اما آبی هم اوکی است.
        # اینجا من رنگ را 'orange' گذاشتم تا با train فرق کند. اگر آبی می‌خواهید 'blue' کنید.
        plot_color = 'darkorange' 
        
        ax.fill_between(x_epochs, y_min, y_max, 
                         color=plot_color, alpha=0.15, label='Min-Max Range')
        
        # 2. رسم خط میانگین
        ax.plot(x_epochs, y_mean, 
                 color=plot_color, linewidth=2, marker='o', markersize=6, label='Mean Accuracy')

        # 3. نوشتن اعداد واریانس
        y_range = np.max(y_max) - np.min(y_min)
        # اگر تغییرات خیلی کم بود، آفست دیفالت بگذاریم تا ارور ندهد
        if y_range == 0: y_range = 1.0 
        
        offset = y_range * 0.05 
        
        for x, y, var in zip(x_epochs, y_mean, y_var):
            label = f"{var:.1e}" 
            
            ax.text(x, y + offset, label, 
                     fontsize=8, 
                     ha='center', 
                     va='bottom', 
                     color='darkred', 
                     rotation=45) 

        # تنظیم عنوان و لیبل‌ها (مخصوص Test)
        ax.set_title(f'Test Accuracy Base Model Tiny Image Net Dataset', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.legend(loc='lower right')
        
        # تنظیم دقیق محور X
        ax.set_xticks(np.arange(START_EPOCH, END_EPOCH + 1, step=1))
        
        # تنظیم محدوده Y
        ylim_top = np.max(y_max) + (offset * 4)
        ylim_bottom = np.min(y_min) - (offset * 2)
        ax.set_ylim(ylim_bottom, ylim_top)

        plt.tight_layout()
        # نام فایل خروجی متفاوت
        plt.savefig('images/new_test_test_id_0025.png', dpi=300)
        print("\nPlot saved as 'new_test_test_id_0025.png'")
        plt.show()

    else:
        print("Error: No valid data found!")