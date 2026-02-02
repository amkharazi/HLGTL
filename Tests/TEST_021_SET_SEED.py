# Author: A.M.Kharazi
# License: BSD 3 clause
# Check Test Plan for more details 
# Test ResNet50 model on CIFAR10 dataset
# New Classifier - Our Methods
# Optimizer Adam - Default
# No Scheduler
# CIFAR10 dataset -> (3, 192, 192) 
# Pretrained
# Trasfer Learning
# Without Adaptive avg pooling
########################################################

# Add all .py files to path
import sys
sys.path.append('..')

# Import Libraries
from Utils.Accuracy_measures import topk_accuracy
from Utils.Cifar10_loader import get_cifar10_dataloaders
from Utils.Num_parameter import count_parameters
from Models.Resnet50 import Resnet50
from Utils.Reshape import reshape

import torchvision.transforms as transforms
from torch import nn
from torch import optim
import tltorch

import time
import torch
import os
import random
import numpy as np

# --- ADDED: Set Seed Function ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'--- Seed set to: {seed} ---')


if __name__ == '__main__':
    
    # Setup the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f'Device is set to : {device}')

    # Set up the transforms (Stateless, can be defined once)
    image_size = 192

    cifar10_transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=2),
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    cifar10_transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # --- MAIN LOOP FOR 30 SEEDS ---
    total_seeds = 30
    
    for seed_id in range(total_seeds):
        print(f"\n{'='*20} STARTING RUN FOR SEED {seed_id} {'='*20}")
        
        # 1. Set the seed for this iteration
        set_seed(seed_id)

        # 2. Re-initialize Dataloaders (Ensures shuffling is reset)
        train_loader, test_loader = get_cifar10_dataloaders(
                                            data_dir = '../datasets',
                                            batch_size = 64,
                                            image_size = 192,
                                            transform_train = cifar10_transform_train ,
                                            transform_test = cifar10_transform_test)
        
        # 3. Re-initialize Classifier (Fresh weights)
        new_classifier = nn.Sequential(
            reshape(split=[16,2,2], map_type=1, device=device),
            tltorch.TRL(input_shape=(16,2,2,128,3,3), output_shape=(10), factorization='Tucker', rank=(4,1,1,20,1,1,10)),
        )
        
        # 4. Re-initialize Model (Fresh weights)
        model = Resnet50(pretrained=True,
                          weights_path='../weights/resnet50_weights.pth',
                          tensorized=True,
                          input_shape=(192,192),
                          num_classes=10,
                          avg_pool=False,
                          new_classifier=new_classifier).to(device)
        
        # Count parameters
        num_parameters = count_parameters(model)
        classifier_parameters = count_parameters(model.classifier)
        if seed_id == 0:
            print(f'This Model has {num_parameters} parameters')
            print(f'This Model has {classifier_parameters} classifier parameters')
        
        # 5. Re-initialize Optimizer and Criterion
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        
        # Define train and test functions inside loop or pass model/optim as args
        # (Updated to accept current model and optimizer)
        def train_epoch(loader, epoch, current_model, current_optimizer):
            current_model.train()
            start_time = time.time()
            running_loss = 0.0
            correct = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0} 

            for _, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                current_optimizer.zero_grad()
                outputs = current_model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                current_optimizer.step()

                running_loss += loss.item()
                accuracies = topk_accuracy(outputs, targets, topk=(1, 2, 3, 4, 5))
                for k in accuracies:
                    correct[k] += accuracies[k]['correct']

            elapsed_time = time.time() - start_time
            top1_acc, top2_acc, top3_acc, top4_acc, top5_acc = [(correct[k]/len(loader.dataset)) for k in correct]
            avg_loss = running_loss / len(loader.dataset)
        
            report_train = f'Train epoch {epoch}: top1={top1_acc}%, top2={top2_acc}%, top3={top3_acc}%, top4={top4_acc}%, top5={top5_acc}%, loss={avg_loss}, time={elapsed_time}s'
            print(report_train)
            return report_train

        def test_epoch(loader, epoch, current_model):
            current_model.eval()
            start_time = time.time()
            running_loss = 0.0
            correct = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0} 

            for _, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = current_model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                accuracies = topk_accuracy(outputs, targets, topk=(1, 2, 3, 4, 5))
                for k in accuracies:
                    correct[k] += accuracies[k]['correct']

            elapsed_time = time.time() - start_time
            top1_acc, top2_acc, top3_acc, top4_acc, top5_acc = [(correct[k]/len(loader.dataset)) for k in correct]
            avg_loss = running_loss / len(loader.dataset)
        
            report_test = f'Test epoch {epoch}: top1={top1_acc}%, top2={top2_acc}%, top3={top3_acc}%, top4={top4_acc}%, top5={top5_acc}%, loss={avg_loss}, time={elapsed_time}s'
            print(report_test)
            return report_test
        
        # 6. Set up the directories for THIS seed
        TEST_ID = f'TEST_ID0021_SEED_{seed_id}'
        result_dir = os.path.join('../results', TEST_ID)
        result_subdir = os.path.join(result_dir, 'accuracy_stats')
        model_subdir = os.path.join(result_dir, 'model_stats')

        os.makedirs(result_subdir, exist_ok=True)
        os.makedirs(model_subdir, exist_ok=True)
        
        with open(os.path.join(result_dir, 'model_stats', 'model_info.txt'), 'a') as f:
            f.write(f'total number of parameters:\n{num_parameters}\ntotal number of classifier parameters:\n{classifier_parameters}')
        
        # Freeze Convolutional Layers
        layer = 0
        for child in model.children():
            layer+=1
            if layer < 3:
                for param in child.parameters():
                    param.requires_grad = False
        
        # Train and Test The Model - Frozen Layers
        n_epoch = 30
        print(f'Training for {len(range(n_epoch))} epochs [Seed {seed_id}]\n')
        for epoch in range(1,n_epoch+1):
            report_train = train_epoch(train_loader, epoch, model, optimizer)
            report_test = test_epoch(test_loader, epoch, model)
        
            report = report_train + '\n' + report_test + '\n\n'
            if epoch % 10 == 0:
                model_path = os.path.join(result_dir, 'model_stats', f'Model_epoch_{epoch}.pth')
                torch.save(model.state_dict(), model_path)
            with open(os.path.join(result_dir, 'accuracy_stats', 'report.txt'), 'a') as f:
                f.write(report)
                
        # Unfreeze all layers
        layer = 0
        for child in model.children():
            layer+=1
            if layer < 3:
                for param in child.parameters():
                    param.requires_grad = True
                    
        # Train and Test The Model - Unfrozen Layers
        n_epoch_additional = 5
        print(f'Training for Additional {len(range(n_epoch_additional))} epochs [Seed {seed_id}]\n')
        for epoch in range(n_epoch+1,n_epoch+n_epoch_additional+1):
            report_train = train_epoch(train_loader, epoch, model, optimizer)
            report_test = test_epoch(test_loader, epoch, model)
        
            report = report_train + '\n' + report_test + '\n\n'
            if epoch % 5 == 0:
                model_path = os.path.join(result_dir, 'model_stats', f'Model_epoch_{epoch}.pth')
                torch.save(model.state_dict(), model_path)
            with open(os.path.join(result_dir, 'accuracy_stats', 'report.txt'), 'a') as f:
                f.write(report)

    print("\n--- All 30 Seeds Completed ---")