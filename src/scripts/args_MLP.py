import argparse

def get_args(rest_args):   
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--chdir', type=str, default="/home/autolab/jhl/KANbeFair-main/src", #required=True,
                        help='basedir')

    parser.add_argument('--layers_width', type=int, default=[5], nargs='+', #required=True,
                        help='the width of each hidden layer')
    parser.add_argument('--batch_norm', action='store_true', default=False,
                        help='whether use batch normalization')
    parser.add_argument('--activation_name', type=str, default="gelu", 
                        help='activation function')
    parser.add_argument('--pre_train_ckpt', type=str, default="", 
                        help='path of the pretrained model')

    parser.add_argument('--dataset', type=str, default="Special_ellipkinc", #required=True,
                        help='dataset')

    parser.add_argument('--batch-size', type=int, default=1024,
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, # 100 MNIST pretrain, 5 Finetune
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='supported optimizer: adam, lbfgs')
    # parser.add_argument('--gamma', type=float, default=0.7,
    #                     help='Learning rate step gamma (default: 0.7, 1.0 for fewshot)')
    parser.add_argument('--loss', type=str, default="mse",
                        help='loss function')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1314,
                        help='random seed (default: 1314)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-model-interval', type = int, default=-1, 
                        help='whether save model along training')
    
    parser.add_argument('--log-dir', type = str, default='/home/autolab/jhl/KANbeFair-main/logs', 
                        help='logdir')
    
    args = parser.parse_args(rest_args)

    return args