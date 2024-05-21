import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

def get_mnist_dataloaders(data_dir='../datasets', transform_train=None, transform_test=None, batch_size=64, image_size=192):
    '''
    creates pytorch dataloaders for MNIST dataset
    ----------
    data_dir : str
            directory of your datasets, default vaule is ../datasets
            
    transform_train : torchvision.transforms
            train dataset transformations

    transform_test : torchvision.transforms
            test dataset transformations

    batch_size : int
            
    image_size  : int
        squared size image size 
    
    reutrns a data loader
    '''
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=2), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform_train, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform_test, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Example usage:
# train_loader, test_loader = get_mnist_dataloaders(data_dir='./dataset', image_size=192)
