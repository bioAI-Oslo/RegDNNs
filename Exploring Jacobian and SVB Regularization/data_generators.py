import torch
from torchvision import datasets, transforms


def data_loader_MNIST():
    """Load MNIST for use with LeNet, as in Hoffman 2019 (batch size 100)."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Values from Hoffman 2019
        ]
    )

    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=100, shuffle=True, num_workers=5
    )

    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=100, shuffle=False, num_workers=5
    )

    return train_loader, test_loader


def data_loader_CIFAR10():
    """Load CIFAR10 data for use with DDNet as described in Hoffman 2019 (batch size 100). Pictures are 32x32x3"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],  # Values from Hoffman 2019
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=100, shuffle=True, num_workers=5
    )

    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=100, shuffle=False, num_workers=5
    )
    return train_loader, test_loader
