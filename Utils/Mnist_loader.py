import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

def get_mnist_dataloaders(data_dir, transform=None, batch_size=64, image_size=32):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.Grayscale(num_output_channels=3), 
            transforms.RandomRotation(10), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Example usage:
# train_loader, test_loader = get_mnist_dataloaders(data_dir='./dataset', image_size=32)
