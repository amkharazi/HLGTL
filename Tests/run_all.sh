#!/bin/bash

# این دستور باعث می‌شود اگر یکی از اسکریپت‌ها ارور داد، کل پروسه متوقف شود (برای دیباگینگ)
# اگر می‌خواهید حتی با ارور هم ادامه دهد، خط زیر را حذف کنید
set -e

echo "Starting execution of all python scripts..."

# لیست فایل‌ها بر اساس تصویر شما
# نکته: اگر ترتیب خاصی مد نظر دارید (مثلا اول train بعد test)، جای خطوط را عوض کنید
# files=(
#     "loss_test_test_id_0025.py"
#     "loss_test_test_id_0028.py"
#     "loss_test_test_id_0032.py"
#     "loss_train_test_id_0025.py"
#     "loss_train_test_id_0028.py"
#     "loss_train_test_id_0032.py"
#     "test_test_id_0025.py"
#     "test_test_id_0028.py"
#     "test_test_id_0032.py"
#     "train_test_id_0025.py"
#     "train_test_id_0028.py"
#     "train_test_id_0032.py"
# )

<<<<<<< HEAD



=======
>>>>>>> 611bee2077e31ee1a6bf8029e3ad344868af8aa9
files=(
    "loss_test_test_id_0025.py"
    "loss_train_test_id_0025.py"
    "test_test_id_0025.py"
    "train_test_id_0025.py"
)


# حلقه برای اجرای فایل‌ها
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "----------------------------------------"
        echo "Running: $file"
        echo "----------------------------------------"
        
        # اجرا با پایتون (اگر از محیط مجازی خاصی استفاده می‌کنید مطمئن شوید فعال است)
        python "$file"
        
        echo "Finished: $file"
        echo ""
    else
        echo "Warning: File $file not found!"
    fi
done

echo "All scripts executed successfully."