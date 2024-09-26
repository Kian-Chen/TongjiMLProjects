import logging, os, sys, gc, time, re
from datetime import datetime
import torch, random, numpy, torchtext
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, Subset
import matplotlib.pyplot as plt

from fvcore.nn import FlopCountAnalysis, parameter_count

from models.mlp import *
from models.kanbefair import *
from models.bspline_mlp import *
from models.utils import *

from data_provider.special import get_scipyfunction_dataset, get_special_dataset_1d

def get_timestamp():
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    print('Log directory: ', log_dir)
    return logger, formatter

def get_loader(args, shuffle = True, use_cuda = True):
    train_kwargs = {'batch_size': args.batch_size, 'num_workers': 4}
    test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': 4}

    if shuffle:
        train_kwargs.update({'shuffle': True})
        test_kwargs.update({'shuffle': False})
    else:
        train_kwargs.update({'shuffle': False})
        test_kwargs.update({'shuffle': False})

    if args.dataset in [
            "Special_ellipj",
            "Special_ellipkinc",
            "Special_ellipeinc",
            "Special_jv",
            "Special_yv",
            "Special_kv",
            "Special_iv",
            "Special_lpmv0",
            "Special_lpmv1",
            "Special_lpmv2",
            "Special_sphharm01",
            "Special_sphharm11",
            "Special_sphharm02",
            "Special_sphharm12",
            "Special_sphharm22",
            ]:
        train_dataset, test_dataset = get_scipyfunction_dataset(args)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 1
        input_size = 2
    elif args.dataset in ["Special_1d_poisson","Special_1d_gelu"]:
        train_dataset, test_dataset = get_special_dataset_1d(args)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 1
        input_size = 1

    else:
        raise NotImplementedError

    return train_loader, test_loader, num_classes, input_size

def get_model(args):
    if args.model == "MLP":
        model = MLP(args)
    elif args.model == "KAN":
        model = KANbeFair(args)
    elif args.model == "MLP_Text":
        model = MLP_Text(args)
    elif args.model == "KAN_Text":
        model = KANbeFair_Text(args)
    elif args.model == "BSpline_MLP":
        model = BSpline_MLP(args)
    elif args.model == "BSpline_First_MLP":
        model = BSpline_First_MLP(args)
    else:
        raise NotImplementedError
    return model

def randomness_control(seed):
    print("seed",seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_matrix(matrix, path):
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap='inferno')
    fig.colorbar(cax)
    fig.savefig(path)

def get_filename(path):
    base_name = os.path.basename(path)  # filename.extension
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension

def measure_time_memory(f):
    def wrapped(*args, **kwargs):
        if torch.cuda.is_available():
            start_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_max_memory_allocated()
        else:
            start_memory = 0

        start_time = time.time()

        result = f(*args, **kwargs)

        end_time = time.time()

        if torch.cuda.is_available():
            end_memory = torch.cuda.max_memory_allocated()
        else:
            end_memory = 0

        print(f"Function {f.__name__} executed in {end_time - start_time:.4f} seconds.")
        print(f"Memory usage increased by {(end_memory - start_memory) / (1024 ** 2):.2f} MB to {(end_memory) / (1024 ** 2):.2f} MB.")
        
        return result
    return wrapped

def classwise_validation(logits, label, targets, args):
    accuracies = []
    for target in targets:
        accuracies.append(accuracy = get_accuracy(logits, label, target))
    return accuracies

def get_accuracy(probability, label, target = None):
    prediction = probability.max(dim = 1)[1]
    if target is None:
        return ((prediction == label).sum() / label.numel()).item()
    else:
        mask = label == target
        return ((prediction[mask]== label[mask]).sum() / label[mask].numel()).item()

def get_activation(args):
    if args.activation_name == 'relu':
        return nn.ReLU
    elif args.activation_name == 'square_relu':
        return Square_ReLU
    elif args.activation_name == 'sigmoid':
        return nn.Sigmoid
    elif args.activation_name == 'tanh':
        return nn.Tanh
    elif args.activation_name == 'softmax':
        return nn.Softmax(dim=1)
    elif args.activation_name == 'silu':
        return nn.SiLU
    elif args.activation_name == 'gelu':
        return nn.GELU
    elif args.activation_name == 'glu':
        return nn.GLU
    elif args.activation_name == 'polynomial2':
        return Polynomial2
    elif args.activation_name == 'polynomial3':
        return Polynomial3
    elif args.activation_name == 'polynomial5':
        return Polynomial5
    else:
        raise ValueError(f'Unknown activation function: {args.activation_name}')

def get_shortcut_function(args):
    if args.kan_shortcut_name == 'silu':
        return nn.SiLU()
    elif args.kan_shortcut_name == 'identity':
        return nn.Identity()
    elif args.kan_shortcut_name == 'zero':

        class Zero(nn.Module):
            def __init__(self):
                super(Zero, self).__init__()
            def forward(self, x):
                return x * 0

        return Zero()
    else:
        raise ValueError(f'Unknown kan shortcut function: {args.kan_shortcut_name}')
    
def get_model_complexity(model, logger, args, method = "coustomized"):

    # using fvcore
    if method == "fvcore":
        parameter_dict = parameter_count(model)
        num_parameters = parameter_dict[""]

        flops_dict = FlopCountAnalysis(model, torch.randn(2, args.input_size))
        flops = flops_dict.total()
    elif method == "coustomized":
        num_parameters = model.total_parameters()
        flops = model.total_flops()
    else:
        raise NotImplementedError

    if logger is not None:
        logger.info(f"Number of parameters: {num_parameters:,}; Number of FLOPs: {flops:,}")

    return num_parameters, flops

def write_results(args, subfix = "", **kwargs):
    result_base = "../results"
    result_file = f"results{subfix}.csv"

    dataset, model, general_parameters, specific_parameter = args.exp_id.split("/")[2:]
    general_parameters = general_parameters.split("__")
    specific_parameter = specific_parameter.split("__")

    result_file_path = os.path.join(result_base, result_file)
    
    s = [get_timestamp(), dataset, model] + general_parameters + specific_parameter + [str(kwargs[key]) for key in kwargs]
    s = ",".join(s) + "\n"

    with open(result_file_path, "a") as f:
        f.write(s)

def todevice(obj, device):
    if isinstance(obj, (list,tuple)):
        obj = [o.to(device) for o in obj]
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    else:
        raise NotImplementedError
    return obj