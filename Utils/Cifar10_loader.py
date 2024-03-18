import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

def get_cifar10_dataloaders(data_dir, transform=None, batch_size=64, image_size=192):
    if transform is None:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=False)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Example usage:
# train_loader, test_loader = get_cifar10_dataloaders(data_dir='./dataset', image_size=192)
