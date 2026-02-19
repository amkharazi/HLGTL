# Author: A.M.Kharazi
# License: BSD 3 clause
# Check Test Plan for more details
# Test EfficientNet-B4 model on Tiny-Imagenet-200 dataset
# New Classifier - Our Methods
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
import tltorch

sys.path.append('..')

from Utils.Accuracy_measures import topk_accuracy
from Utils.TinyImageNet_loader import get_tinyimagenet_dataloaders
from Utils.Num_parameter import count_parameters
from Utils.Reshape import reshape
from Models.EfficientNetB4 import EfficientNetB4, out_shape_efficientnet_b4

# ==========================================
# تابع تنظیم Seed
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
TEST_ID_BASE = 'TEST_ID0032_efficient_net'  # شناسه پایه برای متد ما
SEEDS = [0, 1, 2]             # لیست سیدها
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 192
BATCH_SIZE = 64
NUM_CLASSES = 200

if __name__ == '__main__':
    print(f'Device is set to : {DEVICE}')

    # تعریف ترنسفرم‌ها (یکبار تعریف کافی است)
    tiny_transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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

    # ==========================================
    # شروع حلقه روی Seed ها
    # ==========================================
    for seed in SEEDS:
        print(f'\n{"="*40}')
        print(f'Starting Run for SEED: {seed}')
        print(f'{"="*40}\n')
        
        # 1. تنظیم Seed
        set_seed(seed)

        # 2. آماده‌سازی پوشه‌های خروجی
        current_test_id = f'{TEST_ID_BASE}_SEED_{seed}'
        result_dir = os.path.join('../results', current_test_id)
        accuracy_stats_dir = os.path.join(result_dir, 'accuracy_stats')
        model_stats_dir = os.path.join(result_dir, 'model_stats')

        os.makedirs(accuracy_stats_dir, exist_ok=True)
        os.makedirs(model_stats_dir, exist_ok=True)

        # 3. دریافت دیتالودرها
        train_loader, test_loader, _ = get_tinyimagenet_dataloaders(
            data_dir='../datasets',
            transform_train=tiny_transform_train,
            transform_val=tiny_transform_val,
            transform_test=tiny_transform_test,
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE
        )

        # 4. محاسبات Reshape
        feat_shape = out_shape_efficientnet_b4(in_shape=(IMAGE_SIZE, IMAGE_SIZE), batch_size=2)  # (B,C,H,W)
        C, H, W = int(feat_shape[1]), int(feat_shape[2]), int(feat_shape[3]) # (1792, 6, 6)

        split = [2, 2, 2]
        split_prod = 1
        for s in split:
            split_prod *= s

        C2 = C * 4
        H2, W2 = H // 2, W // 2
        if (H % 2) != 0 or (W % 2) != 0:
            raise ValueError(f'Expected even H,W for reshape(map_type=2). Got H={H}, W={W}')
        if (C2 % split_prod) != 0:
            raise ValueError(f'Expected (C*4) divisible by {split_prod}. Got C={C} -> C*4={C2}')

        C_last = C2 // split_prod
        trl_input_shape = (split[0], split[1], split[2], C_last, H2, W2) # (2 2 2 896 3 3)

        # 5. ساخت مدل (Reshape + TRL)
        new_classifier = nn.Sequential(
            reshape(split=split, map_type=2, device=DEVICE),
            tltorch.TRL(
                input_shape=trl_input_shape,
                output_shape=(NUM_CLASSES),
                factorization='Tucker',
                rank=(1, 1, 1, 100, 1, 1, 100)
            ),
        )

        model = EfficientNetB4(
            pretrained=True,
            weights_path='../weights/efficientnet_b4_weights.pth',
            tensorized=True,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE),
            num_classes=NUM_CLASSES,
            avg_pool=False,
            new_classifier=new_classifier
        ).to(DEVICE)

        # 6. اطلاعات مدل
        num_parameters = count_parameters(model)
        classifier_parameters = count_parameters(model.classifier)
        print(f'Model Parameters: {num_parameters}')
        print(f'Feature map before avgpool: (C,H,W)=({C},{H},{W})')
        print(f'TRL input shape: {trl_input_shape}')

        with open(os.path.join(model_stats_dir, 'model_info.txt'), 'w') as f:
            f.write(
                f'Seed: {seed}\n'
                f'total number of parameters:\n{num_parameters}\n'
                f'total number of classifier parameters:\n{classifier_parameters}\n'
                f'feature map (C,H,W) before avgpool:\n({C},{H},{W})\n'
                f'trl input shape:\n{trl_input_shape}\n'
            )

        # 7. تعریف Loss و Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([
            {'params': model.classifier.parameters(), 'lr': 0.001},
            {'params': model.features.parameters(), 'lr': 0.00001},
        ])

        # 8. توابع Train و Test
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
            total_samples = len(loader.dataset)
            top1_acc = (correct[1] / total_samples) * 100
            top5_acc = (correct[5] / total_samples) * 100
            avg_loss = running_loss / len(loader)

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
            top5_acc = (correct[5] / total_samples) * 100
            avg_loss = running_loss / len(loader)

            report = (
                f'Test epoch {epoch}: top1={top1_acc:.2f}%, top5={top5_acc:.2f}%, '
                f'loss={avg_loss:.4f}, time={elapsed_time:.2f}s'
            )
            print(report)
            return report

        # 9. فاز اول: فریز کردن Feature Extractor
        print('Freezing Feature Extractor...')
        for p in model.features.parameters():
            p.requires_grad = False
        
        n_epoch_phase1 = 5
        print(f'Training Phase 1 (Frozen) for {n_epoch_phase1} epochs...')
        
        for epoch in range(1, n_epoch_phase1 + 1):
            rep_train = train_epoch(train_loader, epoch, model, optimizer, criterion)
            rep_test = test_epoch(test_loader, epoch, model, criterion)

            full_report = rep_train + '\n' + rep_test + '\n\n'
            
            with open(os.path.join(accuracy_stats_dir, 'report.txt'), 'a') as f:
                f.write(full_report)

            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(model_stats_dir, f'Model_epoch_{epoch}.pth'))

        # 10. فاز دوم: آنفریز کردن
        print('Unfreezing Feature Extractor...')
        for p in model.features.parameters():
            p.requires_grad = True
        
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