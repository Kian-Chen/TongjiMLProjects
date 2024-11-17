from data_provider.data_loader import  CIFAR100Loader, CIFAR10Loader, MNISTLoader

data_dict = {
    'mnist': MNISTLoader,
    'cifar-10': CIFAR10Loader,
    'cifar-100': CIFAR100Loader
}

def data_provider(args):
    Data = data_dict[args.data]
    data_set = Data(data_dir=args.data_dir)
    return data_set