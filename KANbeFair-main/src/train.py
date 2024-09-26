from __future__ import print_function
import sys
import os
import argparse
import warnings
import torch
import torch.optim as optim
from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
from fvcore.common.timer import Timer

from utils.utils import *
from models.kan.LBFGS import *
from torch.utils.tensorboard import SummaryWriter

import scripts.args_KAN as args_KAN,scripts.args_MLP as args_MLP

warnings.simplefilter(action='ignore', category=UserWarning)

def train(args, model, device, train_loader, optimizer, epoch, logger, start_index):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader, start_index):
        data, target = todevice(data, device), todevice(target, device)

        if args.optimizer in ["adam",'sgd']:

            optimizer.zero_grad()
            output = model(data)

            if args.loss == "cross_entropy":
                losses = [F.cross_entropy(output, target)]
            elif args.loss == "mse":
                losses = [F.mse_loss(output, target)]
            else:
                raise NotImplementedError
            
            loss = 0
            for l in losses:
                loss = loss + l
            loss.backward()
            optimizer.step()

        elif args.optimizer == "lbfgs":
            # print("lbfgs")

            def closure():
                optimizer.zero_grad()
                output = model(data)
                if args.loss == "cross_entropy":
                    losses = [F.cross_entropy(output, target)]
                elif args.loss == "mse":
                    losses = [F.mse_loss(output, target)]
                else:
                    raise NotImplementedError

                loss = 0
                for l in losses:
                    loss = loss + l

                loss.backward()
                return loss

            optimizer.step(closure)

        if batch_idx % args.log_interval == 0:

            with torch.no_grad():
                output = model(data)
                if args.loss == "cross_entropy":
                    losses = [F.cross_entropy(output, target)]
                elif args.loss == "mse":
                    losses = [F.mse_loss(output, target)]
                else:
                    raise NotImplementedError
                
                logger.add_scalar('Train/Loss', sum(losses).item(), epoch * len(train_loader) + batch_idx)
                # logger_info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: '.format(
                #     epoch, (batch_idx - start_index) * len(data), len(train_loader.dataset),
                #     100. * (batch_idx - start_index) / len(train_loader)) + ",".join([str(l.item()) for l in losses])
                # logger.info(logger_info)

        if args.save_model_along and (batch_idx + 1) % args.save_model_interval == 0:
            torch.save(model.state_dict(), f"{args.exp_id}/{args.operation}_{batch_idx + 1}.pt")
            # logger.info(f"model was saved to {args.exp_id}/{args.operation}_{batch_idx + 1}.pt")

        if args.dry_run:
            break

    return model

def test(args, model, device, test_loader, epoch, logger, name):
    model.eval()

    if args.loss == "cross_entropy":
        
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = todevice(data, device), todevice(target, device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        
        logger.add_scalar('Test/Loss', test_loss, epoch)
        logger.add_scalar('Test/Accuracy', 100. * correct / len(test_loader.dataset), epoch)
        #logger.info("\t" + name + ' batch output: {}, batch target: {}'.format(output.detach().cpu().numpy(), target.detach().cpu().numpy()))

        # logger.info("\t"+name+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        #     test_loss, correct, len(test_loader.dataset),
        #     100. * correct / len(test_loader.dataset)))

        return 100. * correct / len(test_loader.dataset)
    
    elif args.loss == "mse":
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = todevice(data, device), todevice(target, device)
                output = model(data)
                per_sample_loss = F.mse_loss(output, target, reduction='none')
                per_sample_rmse = torch.sqrt(per_sample_loss)
                test_loss += per_sample_rmse.sum().item()  # sum up batch loss

        test_loss /= len(test_loader.dataset)
        
        logger.add_scalar('Test/Loss', test_loss, epoch)
        #logger.info("\t"+name+' set: Average loss: {:.6f}'.format(test_loss))

        return test_loss
    
    else:
        raise NotImplementedError

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="KAN", #required=True,
                        help='network structure')

    args, rest_args = parser.parse_known_args()
    model = args.model

    if model == 'KAN':
        args = args_KAN.get_args(rest_args)
    elif model == 'MLP':
        args = args_MLP.get_args(rest_args)
    else:
        raise NotImplementedError
    
    args.model = model
    os.chdir(args.chdir)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    randomness_control(args.seed)

    args.save_model_along = args.save_model_interval > 0

    args.exp_id = args.log_dir + f"/{args.dataset}/{args.model}/"
    args.exp_id = args.exp_id + f"{'_'.join([str(w) for w in args.layers_width])}__{args.batch_norm}__{args.activation_name}"
    args.exp_id = args.exp_id + f"__{args.batch_size}__{args.epochs}__{args.lr}__{args.seed}__{args.optimizer}"
    os.makedirs(args.exp_id, exist_ok = True)
    ################# id for KAN #################
    if args.model in ["KAN", "KAN_Text"]:
        args.exp_id = args.exp_id + f"/{args.kan_bspline_grid}__{args.kan_bspline_order}__{args.kan_shortcut_name}"+ f"__{'_'.join([str(w) for w in args.kan_grid_range])}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for KAN #################
    ################# id for BSpline MLP #################
    elif args.model == "BSpline_MLP":
        args.exp_id = args.exp_id + f"/{args.kan_bspline_grid}__{args.kan_bspline_order}"+ f"__{'_'.join([str(w) for w in args.kan_grid_range])}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for BSpline MLP #################
    ################# id for BSpline First MLP #################
    elif args.model == "BSpline_First_MLP":
        args.exp_id = args.exp_id + f"/{args.kan_bspline_grid}__{args.kan_bspline_order}"+ f"__{'_'.join([str(w) for w in args.kan_grid_range])}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for BSpline First MLP #################
    ################# id for MLP #################
    elif args.model in ["MLP", "MLP_Text"]:
        args.exp_id = args.exp_id + f"/default"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for MLP #################
    else:
        raise NotImplementedError
    
    # if os.path.exists(os.path.join(args.exp_id, "log")):
    #     with open(os.path.join(args.exp_id, "log"), "r") as f:
    #         lines = f.readlines()
    #         if len(lines) > 0:
    #             if "training process was finished" in lines[-1]:
    #                 raise ValueError("training process was finished")

    #logger, formatter = get_logger(args.exp_id, None, "log", level=logging.INFO)
    
    logger = SummaryWriter(log_dir=args.exp_id)

    train_loader, test_loader, num_classes, input_size = get_loader(args, use_cuda = use_cuda)

    args.output_size = num_classes
    args.input_size = input_size

    args.activation = get_activation(args)
    if(args.model == "KAN"):
        args.kan_shortcut_function = get_shortcut_function(args)

    model = get_model(args)
    # logger.info(model)
    # num_parameters, flops = get_model_complexity(model, logger, args)
    model = model.to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == "lbfgs":
        optimizer = LBFGS(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            history_size=10, 
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-32, 
            tolerance_change=1e-32, 
            tolerance_ys=1e-32)
    else:
        raise NotImplementedError

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if args.loss == "cross_entropy":
        best_test_metric = 0 
    elif args.loss == "mse":
        best_test_metric = 1e10 
    else:
        raise NotImplementedError
    corresponding_train_metric = 0

    fvctimer = Timer()
    for epoch in range(1, args.epochs + 1):
        if fvctimer.is_paused():
            fvctimer.resume()
        else:
            fvctimer.reset()
        train(args, model, device, train_loader, optimizer, epoch, logger, start_index = (epoch - 1) *len(train_loader))
        fvctimer.pause()
        train_metric = test(args, model, device, train_loader, epoch, logger, name = "train")
        test_metric = test(args, model, device, test_loader, epoch, logger, name = "test")
        
        if args.loss == "cross_entropy":
            if test_metric > best_test_metric:
                best_test_metric = test_metric
                corresponding_train_metric = train_metric
        elif args.loss == "mse":
            if test_metric < best_test_metric:
                best_test_metric = test_metric
                corresponding_train_metric = train_metric
        else:
            raise NotImplementedError


        # scheduler.step()

    # total_training_time = fvctimer.seconds()
    # average_training_time_per_epoch = fvctimer.avg_seconds()
    # logger.info(f"total training time: {total_training_time:,} seconds; average training time per epoch: {average_training_time_per_epoch:,} seconds")

    # write_results(
    #     args,
    #     train_metric = corresponding_train_metric,
    #     test_metric = best_test_metric,
    #     num_parameters = num_parameters,
    #     flops = flops,
    #     total_training_time = total_training_time,
    #     average_training_time_per_epoch = average_training_time_per_epoch
    # )

    if args.save_model:
        torch.save(
            {   
                "args" : args,
                "state_dict" : model.state_dict(),
                "metrics" : {
                    "train_metric" : corresponding_train_metric,
                    "test_metric" : best_test_metric,
                    "num_parameters" : num_parameters,
                    "flops" : flops,
                    "total_training_time" : total_training_time,
                    "average_training_time_per_epoch" : average_training_time_per_epoch
                }
            }, f"{args.exp_id}/ckpt.pt")
        # logger.info(f"model was saved to {args.exp_id}/ckpt.pt")

    # logger.info(f"training process was finished")
    logger.close()

if __name__ == '__main__':
    main()
