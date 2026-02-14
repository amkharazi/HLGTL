import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# تنظیمات
# ==========================================
RESULTS_DIR = '../results'
SEED_PREFIX = 'TEST_ID0032'

START_EPOCH = 6    
END_EPOCH = 15     
NUM_SEEDS = 16     

# ==========================================
# توابع
# ==========================================
def get_train_accuracy(folder_name):
    file_path = os.path.join(RESULTS_DIR, folder_name, 'accuracy_stats', 'report.txt')
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            pattern = r"Train epoch \d+:.*?top1=([\d\.]+)"
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
    print(f"Processing {NUM_SEEDS} seeds for {SEED_PREFIX}...")

    # جمع‌آوری داده‌ها
    for seed in range(NUM_SEEDS):
        folder_name = f'{SEED_PREFIX}_SEED_{seed}'
        accs = get_train_accuracy(folder_name)
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

        # 1. رسم بازه (سایه)
        ax.fill_between(x_epochs, y_min, y_max, 
                         color='blue', alpha=0.15, label='Min-Max Range')
        
        # 2. رسم خط میانگین
        ax.plot(x_epochs, y_mean, 
                 color='blue', linewidth=2, marker='o', markersize=6, label='Mean Accuracy')

        # 3. نوشتن اعداد واریانس (اصلاح شده)
        # محاسبه فاصله داینامیک: 5 درصدِ کل ارتفاع نمودار
        y_range = np.max(y_max) - np.min(y_min)
        offset = y_range * 0.05 
        
        for x, y, var in zip(x_epochs, y_mean, y_var):
            # فرمت علمی: e.g., 1.25e-04
            label = f"{var:.1e}" 
            
            ax.text(x, y + offset, label, 
                     fontsize=8, 
                     ha='center', 
                     va='bottom', 
                     color='darkred', 
                     rotation=45) # چرخش متن برای جلوگیری از تداخل

        # تنظیم عنوان و لیبل‌ها
        ax.set_title(f'Train Accuracy Our Method On Tiny Image Net Dataset', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Train Accuracy (%)', fontsize=12)
        ax.legend(loc='lower right')
        
        # تنظیم دقیق محور X
        ax.set_xticks(np.arange(START_EPOCH, END_EPOCH + 1, step=1))
        
        # تنظیم محدوده Y برای اینکه متن‌ها بیرون نزنند
        # کمی فضای خالی بالا اضافه می‌کنیم
        ylim_top = np.max(y_max) + (offset * 4)
        ylim_bottom = np.min(y_min) - (offset * 2)
        ax.set_ylim(ylim_bottom, ylim_top)

        plt.tight_layout()
        plt.savefig('images/train_variance_fixed.png', dpi=300)
        print("\nPlot saved as 'train_variance_fixed.png'")
        plt.show()

    else:
        print("Error: No valid data found!")