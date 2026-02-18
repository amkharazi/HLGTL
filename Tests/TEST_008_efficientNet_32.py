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
sys.path.append('..')

from Utils.Accuracy_measures import topk_accuracy
from Utils.TinyImageNet_loader import get_tinyimagenet_dataloaders
from Utils.Num_parameter import count_parameters
from Utils.Reshape import reshape

from Models.EfficientNetB4 import EfficientNetB4, out_shape_efficientnet_b4

import torchvision.transforms as transforms
from torch import nn
from torch import optim
import tltorch

import time
import torch
import os


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device is set to : {device}')

    image_size = 192

    tiny_transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    tiny_transform_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    tiny_transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_loader, test_loader, _ = get_tinyimagenet_dataloaders(
        data_dir='../datasets',
        transform_train=tiny_transform_train,
        transform_val=tiny_transform_val,
        transform_test=tiny_transform_test,
        batch_size=64,
        image_size=192
    )

    feat_shape = out_shape_efficientnet_b4(in_shape=(192, 192), batch_size=2)  # (B,C,H,W)
    C, H, W = int(feat_shape[1]), int(feat_shape[2]), int(feat_shape[3]) # (1792, 6, 6)

    split = [16, 2, 2]
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
    trl_input_shape = (split[0], split[1], split[2], C_last, H2, W2) # (16 2 2 112 3 3)

    new_classifier = nn.Sequential(
        reshape(split=split, map_type=1, device=device),
        tltorch.TRL(
            input_shape=trl_input_shape,
            output_shape=(200),
            factorization='Tucker',
            rank=(4, 1, 1, 50, 1, 1, 200)
        ),
    )

    model = EfficientNetB4(
        pretrained=True,
        weights_path='../weights/efficientnet_b4_weights.pth',
        tensorized=True,
        input_shape=(192, 192),
        num_classes=200,
        avg_pool=False,
        new_classifier=new_classifier
    ).to(device)

    num_parameters = count_parameters(model)
    classifier_parameters = count_parameters(model.classifier)
    print(f'This Model has {num_parameters} parameters')
    print(f'This Model has {classifier_parameters} classifier parameters')
    print(f'Feature map before avgpool: (C,H,W)=({C},{H},{W})')
    print(f'TRL input shape: {trl_input_shape}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.classifier.parameters(), 'lr': 0.001},
        {'params': model.features.parameters(), 'lr': 0.00001},
    ])

    def train_epoch(loader, epoch):
        model.train()

        start_time = time.time()
        running_loss = 0.0
        correct = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

        for _, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

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
        top1_acc, top2_acc, top3_acc, top4_acc, top5_acc = [(correct[k] / len(loader.dataset)) for k in correct]
        avg_loss = running_loss / len(loader.dataset)

        report_train = (
            f'Train epoch {epoch}: top1={top1_acc}%, top2={top2_acc}%, top3={top3_acc}%, '
            f'top4={top4_acc}%, top5={top5_acc}%, loss={avg_loss}, time={elapsed_time}s'
        )
        print(report_train)
        return report_train

    def test_epoch(loader, epoch):
        model.eval()

        start_time = time.time()
        running_loss = 0.0
        correct = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

        with torch.no_grad():
            for _, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                accuracies = topk_accuracy(outputs, targets, topk=(1, 2, 3, 4, 5))
                for k in accuracies:
                    correct[k] += accuracies[k]['correct']

        elapsed_time = time.time() - start_time
        top1_acc, top2_acc, top3_acc, top4_acc, top5_acc = [(correct[k] / len(loader.dataset)) for k in correct]
        avg_loss = running_loss / len(loader.dataset)

        report_test = (
            f'Test epoch {epoch}: top1={top1_acc}%, top2={top2_acc}%, top3={top3_acc}%, '
            f'top4={top4_acc}%, top5={top5_acc}%, loss={avg_loss}, time={elapsed_time}s'
        )
        print(report_test)
        return report_test

    TEST_ID = 'Test_ID008_EfficientNetB4_32_ourmethod'
    result_dir = os.path.join('../results', TEST_ID)
    result_subdir = os.path.join(result_dir, 'accuracy_stats')
    model_subdir = os.path.join(result_dir, 'model_stats')

    os.makedirs(result_subdir, exist_ok=True)
    os.makedirs(model_subdir, exist_ok=True)

    with open(os.path.join(result_dir, 'model_stats', 'model_info.txt'), 'a') as f:
        f.write(
            f'total number of parameters:\n{num_parameters}\n'
            f'total number of classifier parameters:\n{classifier_parameters}\n'
            f'feature map (C,H,W) before avgpool:\n({C},{H},{W})\n'
            f'trl input shape:\n{trl_input_shape}\n'
        )

    for p in model.features.parameters():
        p.requires_grad = False

    n_epoch = 5
    print(f'Training for {len(range(n_epoch))} epochs\n')
    for epoch in range(1, n_epoch + 1):
        report_train = train_epoch(train_loader, epoch)
        report_test = test_epoch(test_loader, epoch)

        report = report_train + '\n' + report_test + '\n\n'
        if epoch % 5 == 0:
            model_path = os.path.join(result_dir, 'model_stats', f'Model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_path)

        with open(os.path.join(result_dir, 'accuracy_stats', 'report.txt'), 'a') as f:
            f.write(report)

    for p in model.features.parameters():
        p.requires_grad = True

    n_epoch_additional = 10
    print(f'Training for Additional {len(range(n_epoch_additional))} epochs\n')
    for epoch in range(n_epoch + 1, n_epoch + n_epoch_additional + 1):
        report_train = train_epoch(train_loader, epoch)
        report_test = test_epoch(test_loader, epoch)

        report = report_train + '\n' + report_test + '\n\n'
        if epoch % 5 == 0:
            model_path = os.path.join(result_dir, 'model_stats', f'Model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_path)

        with open(os.path.join(result_dir, 'accuracy_stats', 'report.txt'), 'a') as f:
            f.write(report)
