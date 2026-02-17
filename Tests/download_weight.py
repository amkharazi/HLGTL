import torch
import torchvision.models as models
import os

# 1. تنظیم مسیر ذخیره‌سازی (یک پوشه عقب‌تر، داخل weights)
save_dir = '../weights'
file_name = 'resnet50_weights.pth'
save_path = os.path.join(save_dir, file_name)

# 2. ساخت پوشه weights اگر وجود ندارد
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Directory created: {save_dir}")

print("Downloading ResNet50 weights from torchvision...")

# 3. دانلود مدل با وزن‌های ImageNet
# استفاده از وزن‌های جدید (IMAGENET1K_V1)
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 4. ذخیره فقط وزن‌ها (state_dict) در فایل
torch.save(resnet50.state_dict(), save_path)

print(f"Success! Weights saved to: {os.path.abspath(save_path)}")
