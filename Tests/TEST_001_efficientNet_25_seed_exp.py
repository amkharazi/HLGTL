# Author: A.M.Kharazi
# License: BSD 3 clause
# Check Test Plan for more details
# Test EfficientNet-B4 model on Tiny-Imagenet-200 dataset
# New Classifier - Basic Model with dropout
# Optimizer Adam - Avoid Catastrophic Forgetting
# No Scheduler
# Tiny-Imagenet-200 dataset -> (3, 192, 192)
# Pretrained
# Trasfer Learning
# Without Adaptive avg pooling
########################################################

import sys
import os
import time
import random
import numpy as np
import torch
from torch import nn
from torch import optim
import torchvision.transforms as transforms

sys.path.append('..')

from Utils.Accuracy_measures import topk_accuracy
from Utils.TinyImageNet_loader import get_tinyimagenet_dataloaders
from Utils.Num_parameter import count_parameters
from Models.EfficientNetB4 import EfficientNetB4, out_shape_efficientnet_b4

# ==========================================
# تابع تنظیم Seed برای بازتولیدپذیری
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Seed set to: {seed}')

# ==========================================
# تنظیمات اصلی
# ==========================================
TEST_ID_BASE = 'TEST_ID0025_efficient_net_exp'  # شناسه پایه تست شما
SEEDS = [0, 1, 2]             # لیست سیدهایی که می‌خواهید اجرا کنید
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 192
BATCH_SIZE = 64
NUM_CLASSES = 200

if __name__ == '__main__':
    print(f'Device is set to : {DEVICE}')

    # تعریف ترنسفرم‌ها (یکبار تعریف کافی است)
    tiny_transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomCrop(IMAGE_SIZE, padding=5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    tiny_transform_val = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    tiny_transform_test = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # لود کردن دیتاست (چون دیتاست ثابت است، می‌توان بیرون حلقه لود کرد یا داخل، اما دیتالودر بهتر است داخل باشد اگر شافل دارد)
    # اما برای اطمینان از سیدینگ دیتالودر، بهتر است داخل حلقه باشد یا سید دیتالودر هندل شود.
    # در اینجا برای سادگی و اطمینان، دیتالودر را در هر دور می‌گیریم (هرچند سربار کمی دارد).
    
    # ==========================================
    # شروع حلقه روی Seed ها
    # ==========================================
    for seed in SEEDS:
        print(f'\n{"="*40}')
        print(f'Starting Run for SEED: {seed}')
        print(f'{"="*40}\n')
        
        # 1. تنظیم Seed
        set_seed(seed)

        # 2. آماده‌سازی پوشه‌های خروجی مخصوص این Seed
        # نام پوشه: TEST_ID0025_SEED_0, TEST_ID0025_SEED_1, ...
        current_test_id = f'{TEST_ID_BASE}_SEED_{seed}'
        result_dir = os.path.join('../results', current_test_id)
        accuracy_stats_dir = os.path.join(result_dir, 'accuracy_stats')
        model_stats_dir = os.path.join(result_dir, 'model_stats')

        os.makedirs(accuracy_stats_dir, exist_ok=True)
        os.makedirs(model_stats_dir, exist_ok=True)

        # 3. دریافت دیتالودرها (برای اعمال صحیح سید روی شافلینگ)
        train_loader, test_loader, _ = get_tinyimagenet_dataloaders(
            data_dir='../datasets',
            transform_train=tiny_transform_train,
            transform_val=tiny_transform_val,
            transform_test=tiny_transform_test,
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE
        )

        # 4. ساخت مدل
        out_shape = out_shape_efficientnet_b4(in_shape=(IMAGE_SIZE, IMAGE_SIZE), batch_size=2)
        in_features = out_shape[1] * out_shape[2] * out_shape[3]

        new_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_features, out_features=NUM_CLASSES, bias=True),
        )

        model = EfficientNetB4(
            pretrained=True,
            weights_path='../weights/efficientnet_b4_weights.pth',
            tensorized=False,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE),
            num_classes=NUM_CLASSES,
            avg_pool=False,
            new_classifier=new_classifier
        ).to(DEVICE)

        # 5. اطلاعات مدل
        num_parameters = count_parameters(model)
        classifier_parameters = count_parameters(model.classifier)
        print(f'Model Parameters: {num_parameters}')
        
        # ذخیره اطلاعات مدل (فقط یکبار در ابتدای کار هر سید)
        with open(os.path.join(model_stats_dir, 'model_info.txt'), 'w') as f:
            f.write(
                f'Seed: {seed}\n'
                f'total number of parameters:\n{num_parameters}\n'
                f'total number of classifier parameters:\n{classifier_parameters}\n'
                f'classifier in_features:\n{in_features}\n'
            )

        # 6. تعریف Loss و Optimizer
        criterion = nn.CrossEntropyLoss()

        param_groups = [
                {'params': model.classifier.parameters(), 'lr': 0.00005},
                {'params': model.features.parameters(), 'lr': 0.00001},
            ]
        avgpool_params = list(model.avgpool.parameters())
        if len(avgpool_params) > 0:
            param_groups.append({'params': avgpool_params, 'lr': 0.00005})

        optimizer = optim.Adam(param_groups)

        # 7. توابع Train و Test (محلی برای دسترسی به متغیرهای داخل حلقه)
        def train_epoch(loader, epoch, model, optimizer, criterion):
            model.train()
            start_time = time.time()
            running_loss = 0.0
            correct = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

            for _, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                accuracies = topk_accuracy(outputs, targets, topk=(1, 2, 3, 4, 5))
                for k in accuracies:
                    correct[k] += accuracies[k]['correct']

            elapsed_time = time.time() - start_time
            # محاسبه میانگین‌ها
            total_samples = len(loader.dataset)
            top1_acc = (correct[1] / total_samples) * 100 # درصد
            top2_acc = (correct[2] / total_samples) * 100
            top3_acc = (correct[3] / total_samples) * 100
            top4_acc = (correct[4] / total_samples) * 100
            top5_acc = (correct[5] / total_samples) * 100
            avg_loss = running_loss / len(loader) # میانگین لاس بر هر بچ

            report = (
                f'Train epoch {epoch}: top1={top1_acc:.2f}%, top5={top5_acc:.2f}%, '
                f'loss={avg_loss:.4f}, time={elapsed_time:.2f}s'
            )
            print(report)
            return report

        def test_epoch(loader, epoch, model, criterion):
            model.eval()
            start_time = time.time()
            running_loss = 0.0
            correct = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(loader):
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    running_loss += loss.item()
                    accuracies = topk_accuracy(outputs, targets, topk=(1, 2, 3, 4, 5))
                    for k in accuracies:
                        correct[k] += accuracies[k]['correct']

            elapsed_time = time.time() - start_time
            total_samples = len(loader.dataset)
            top1_acc = (correct[1] / total_samples) * 100
            top2_acc = (correct[2] / total_samples) * 100
            top3_acc = (correct[3] / total_samples) * 100
            top4_acc = (correct[4] / total_samples) * 100
            top5_acc = (correct[5] / total_samples) * 100
            avg_loss = running_loss / len(loader)

            report = (
                f'Test epoch {epoch}: top1={top1_acc:.2f}%, top5={top5_acc:.2f}%, '
                f'loss={avg_loss:.4f}, time={elapsed_time:.2f}s'
            )
            print(report)
            return report

        # 8. فاز اول آموزش (فریز کردن لایه‌های اولیه)
        # طبق کد اصلی شما: layer < 3 فریز شوند
        layer_idx = 0
        for child in model.children():
            layer_idx += 1
            if layer_idx < 3:
                for param in child.parameters():
                    param.requires_grad = False
        
        n_epoch_phase1 = 5
        print(f'Training Phase 1 (Frozen) for {n_epoch_phase1} epochs...')
        
        for epoch in range(1, n_epoch_phase1 + 1):
            rep_train = train_epoch(train_loader, epoch, model, optimizer, criterion)
            rep_test = test_epoch(test_loader, epoch, model, criterion)

            full_report = rep_train + '\n' + rep_test + '\n\n'
            
            # ذخیره گزارش
            with open(os.path.join(accuracy_stats_dir, 'report.txt'), 'a') as f:
                f.write(full_report)

            # ذخیره مدل (مثلا هر 5 ایپاک)
            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(model_stats_dir, f'Model_epoch_{epoch}.pth'))

        # 9. فاز دوم آموزش (آنفریز کردن)
        layer_idx = 0
        for child in model.children():
            layer_idx += 1
            if layer_idx < 3:
                for param in child.parameters():
                    param.requires_grad = True
        
        n_epoch_phase2 = 10
        print(f'Training Phase 2 (Unfrozen) for {n_epoch_phase2} epochs...')

        for epoch in range(n_epoch_phase1 + 1, n_epoch_phase1 + n_epoch_phase2 + 1):
            rep_train = train_epoch(train_loader, epoch, model, optimizer, criterion)
            rep_test = test_epoch(test_loader, epoch, model, criterion)

            full_report = rep_train + '\n' + rep_test + '\n\n'
            
            with open(os.path.join(accuracy_stats_dir, 'report.txt'), 'a') as f:
                f.write(full_report)

            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(model_stats_dir, f'Model_epoch_{epoch}.pth'))

        print(f'Run for Seed {seed} completed.\n')

    print('All seeds completed successfully.')