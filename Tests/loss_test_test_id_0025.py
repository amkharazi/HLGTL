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
def get_test_loss(folder_name):
    """
    این تابع داده‌های Test Loss را می‌خواند.
    """
    file_path = os.path.join(RESULTS_DIR, folder_name, 'accuracy_stats', 'report.txt')
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # --- تغییر کلیدی: جستجو برای مقدار loss در خطوط Test epoch ---
            # الگوی قبلی: Test epoch 8: ... loss=0.00465...
            pattern = r"Test epoch \d+:.*?loss=([\d\.]+)"
            matches = re.findall(pattern, content)
            
            # تبدیل به float
            losses = [float(m) for m in matches]
            return losses
    except Exception as e:
        print(f"Error reading {folder_name}: {e}")
        return []

# ==========================================
# اجرای اصلی
# ==========================================
if __name__ == "__main__":
    
    all_seeds_data = []
    print(f"Processing {NUM_SEEDS} seeds for {SEED_PREFIX} (TEST LOSS)...")

    # جمع‌آوری داده‌ها
    for seed in range(NUM_SEEDS):
        folder_name = f'{SEED_PREFIX}_SEED_{seed}'
        # فراخوانی تابع جدید برای دریافت Loss
        losses = get_test_loss(folder_name)
        
        if len(losses) >= END_EPOCH:
            all_seeds_data.append(losses[:END_EPOCH])

    if len(all_seeds_data) > 0:
        data_np = np.array(all_seeds_data)
        
        # برش زمانی (ایپاک 6 تا 15)
        idx_start = START_EPOCH - 1
        idx_end = END_EPOCH 
        sliced_data = data_np[:, idx_start:idx_end]
        
        # محاسبات آماری
        y_mean = np.mean(sliced_data, axis=0)
        y_min = np.min(sliced_data, axis=0)
        y_max = np.max(sliced_data, axis=0)
        y_var = np.var(sliced_data, axis=0)
        
        x_epochs = list(range(START_EPOCH, END_EPOCH + 1))

        # --- شروع رسم ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.grid(True, linestyle='--', alpha=0.5)

        # رنگ نمودار (قرمز/نارنجی برای تست مناسب است)
        plot_color = 'darkorange' 
        
        # 1. رسم بازه (سایه Min-Max)
        ax.fill_between(x_epochs, y_min, y_max, 
                         color=plot_color, alpha=0.15, label='Min-Max Range')
        
        # 2. رسم خط میانگین
        ax.plot(x_epochs, y_mean, 
                 color=plot_color, linewidth=2, marker='o', markersize=6, label='Mean Loss')

        # 3. نوشتن اعداد واریانس
        y_range = np.max(y_max) - np.min(y_min)
        # جلوگیری از تقسیم بر صفر یا آفست صفر در صورت ثابت بودن دیتا
        if y_range == 0: y_range = np.max(y_max) * 0.1 if np.max(y_max) !=0 else 1.0
        
        offset = y_range * 0.05 
        
        for x, y, var in zip(x_epochs, y_mean, y_var):
            # فرمت علمی برای واریانس (چون اعداد Loss کوچک هستند، واریانس خیلی کوچک می‌شود)
            label = f"{var:.1e}" 
            
            ax.text(x, y + offset, label, 
                     fontsize=8, 
                     ha='center', 
                     va='bottom', 
                     color='darkred', 
                     rotation=45) 

        # تنظیم عنوان و لیبل‌ها
        ax.set_title(f'Test Loss Base Model Tiny Image Net Dataset', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Test Loss', fontsize=12)
        ax.legend(loc='upper right') # برای Loss معمولاً Legend بالا سمت راست بهتر است (چون نمودار نزولی است)
        
        # تنظیم دقیق محور X
        ax.set_xticks(np.arange(START_EPOCH, END_EPOCH + 1, step=1))
        
        # تنظیم محدوده Y برای اینکه متن‌ها بیرون نزنند
        ylim_top = np.max(y_max) + (offset * 4)
        ylim_bottom = np.min(y_min) - (offset * 2)
        # چک میکنیم که پایین نمودار منفی نشود (چون Loss منفی نداریم)
        if ylim_bottom < 0: ylim_bottom = 0
            
        ax.set_ylim(ylim_bottom, ylim_top)

        plt.tight_layout()
        
        # ساخت پوشه images اگر وجود ندارد
        os.makedirs('images', exist_ok=True)
        
        # نام فایل خروجی
        save_path = 'images/test_loss_test_id_0025.png'
        plt.savefig(save_path, dpi=300)
        print(f"\nPlot saved as '{save_path}'")
        plt.show()

    else:
        print("Error: No valid data found!")