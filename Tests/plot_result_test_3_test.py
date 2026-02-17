import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_accuracy_reports(base_dir, num_seeds, epochs):
    """
    این تابع وارد پوشه نتایج می‌شود و اعداد را از فایل‌های متنی استخراج می‌کند.
    """
    # ماتریسی برای ذخیره دقت‌ها: سطرها=Seedها، ستون‌ها=Epochها
    all_accuracies = np.zeros((num_seeds, epochs))
    
    for seed in range(num_seeds):
        # ساخت مسیر پوشه بر اساس الگوی نامگذاری کد شما
        # توجه: باید نام پوشه دقیقاً مشابه کدی باشد که ران کردید
        folder_name = f'TEST_ID003_SEED_{seed}' 
        file_path = os.path.join(base_dir, folder_name, 'accuracy_stats', 'report.txt')
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
            
            # استفاده از Regex برای پیدا کردن دقت تست در هر اپوک
            # الگوی کد شما: Test epoch {epoch}: top1={top1_acc}%
            # این الگو به دنبال top1=یک عدد و سپس علامت % یا , می گردد
            pattern = r"Test epoch \d+:.*?top1=([\d\.]+)"
            matches = re.findall(pattern, content)
            
            # تبدیل رشته‌ها به عدد
            # اگر در کد شما عدد بین ۰ تا ۱ است، در ۱۰۰ ضرب می‌کنیم تا درصد شود
            # اگر خود عدد درصد است، ضربدر ۱۰۰ را بردارید
            accs = [float(m) * 100 for m in matches] 
            
            # هندل کردن حالتی که تعداد اپوک‌های ذخیره شده کمتر یا بیشتر باشد
            if len(accs) >= epochs:
                all_accuracies[seed, :] = accs[:epochs]
            else:
                # اگر ران کامل نشده باشد با NaN پر می‌کنیم یا آخرین مقدار را تکرار می‌کنیم
                print(f"Seed {seed} has only {len(accs)} epochs recorded.")
                all_accuracies[seed, :len(accs)] = accs

    return all_accuracies

def plot_mean_std(data_dict, title='Model Performance over Seeds'):
    """
    این تابع دیکشنری از مدل‌ها را می‌گیرد و نمودار مقایسه‌ای می‌کشد.
    data_dict = {'Base Model': matrix_data, 'My Model 1': matrix_data, ...}
    """
    plt.figure(figsize=(10, 6))
    
    # تنظیم استایل
    plt.grid(True, linestyle='--', alpha=0.6)
    
    colors = ['blue', 'red', 'green', 'orange'] # رنگ برای مدل‌های مختلف
    
    for i, (model_name, data) in enumerate(data_dict.items()):
        # محاسبه میانگین و انحراف معیار در طول Seedها (axis 0)
        mean_acc = np.mean(data, axis=0)
        std_acc = np.std(data, axis=0)
        
        epochs = range(1, len(mean_acc) + 1)
        color = colors[i % len(colors)]
        
        # رسم خط میانگین
        plt.plot(epochs, mean_acc, label=f'{model_name} (Mean)', color=color, linewidth=2)
        
        # رسم سایه (انحراف معیار)
        plt.fill_between(epochs, 
                         mean_acc - std_acc, 
                         mean_acc + std_acc, 
                         color=color, alpha=0.2, label=f'{model_name} (Std Dev)')

    plt.title(title, fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('TCL_TRL.png', dpi=300) # ذخیره با کیفیت بالا برای مقاله
    plt.show()

# --- بخش اصلی اجرا ---
if __name__ == "__main__":
    # تنظیمات
    RESULTS_DIR = '../results'  # مسیر پوشه results خود را چک کنید
    NUM_SEEDS = 30
    NUM_EPOCHS = 35 # مجموع اپوک‌های فاز ۱ (۳۰) + فاز ۲ (۵)
    
    # 1. بارگذاری داده‌های مدل Base
    # فرض بر این است که نام پوشه‌های مدل بیس به فرمت TEST_ID001_SEED_x است
    # اگر برای مدل‌های دیگر نام پوشه فرق دارد، باید منطق نام‌گذاری در تابع parse را تغییر دهید
    # یا تابع parse را طوری تغییر دهید که پیشوند نام پوشه را هم بگیرد.
    
    # بیایید تابع parse را کمی هوشمندتر کنیم که پیشوند بگیرد (در کد بالا ساده نوشتم)
    # فرض کنیم الان فقط دیتای همین کد Base را می‌خواهید:
    
    base_model_data = parse_accuracy_reports(RESULTS_DIR, NUM_SEEDS, NUM_EPOCHS)
    
    # اگر مدل‌های دیگر شما در پوشه‌هایی با نام متفاوت هستند (مثلا MYMODEL_SEED_0)
    # باید تابع parse را برای آن‌ها هم صدا بزنید و در دیکشنری زیر بگذارید
    
    # فعلاً فقط مدل بیس را رسم می‌کنیم
    models_data = {
        'Base Model (ResNet50)': base_model_data,
        # 'Proposed Model 1': my_model1_data,  <-- اینجا مدل‌های دیگر را اضافه کنید
        # 'Proposed Model 2': my_model2_data
    }
    
    plot_mean_std(models_data, title='Comparison of Accuracy across 30 Random Seeds')
