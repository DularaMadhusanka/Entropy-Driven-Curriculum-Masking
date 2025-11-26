import argparse
import torch
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="CBM / EGM Training Arguments")
    
    # Paths
    parser.add_argument('--data', type=str, default='./data/oxford-iiit-pet', help="Path to dataset")
    parser.add_argument('--dataset', type=str, default='oxford', choices=['cifar10', 'oxford'])
    parser.add_argument('--model_name', type=str, default='resnet18')
    
    # Method Switch
    parser.add_argument('--mask_metric', type=str, default='entropy', choices=['gradient', 'entropy'])

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.01) 
    parser.add_argument('--batch_size', type=int, default=32)
    

    parser.add_argument('--num_epochs', type=int, default=100) 
    parser.add_argument('--decay_epoch', type=int, default=30)
    parser.add_argument('--stop_decay_epoch', type=int, default=90)
    parser.add_argument('--decay_step', type=int, default=20)

    # Curriculum
    parser.add_argument('--max_mask_ratio', type=float, default=0.75)
    parser.add_argument('--schedule_type', type=str, default='linear_repeat')

    args = parser.parse_args()

    # Device Setup
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")

    args.percent = []
    
    if args.schedule_type == 'constant':
        args.percent = [args.max_mask_ratio] * args.num_epochs

    elif args.schedule_type == 'linear':
        for epoch in range(args.num_epochs):
            ratio = (epoch / args.num_epochs) * args.max_mask_ratio
            args.percent.append(ratio)

    elif args.schedule_type == 'linear_repeat':
        # 100 epochs / 10 interval = 10 repeats
        repeat_interval = 10 
        for epoch in range(args.num_epochs):
            progress_in_cycle = (epoch % repeat_interval) / repeat_interval
            ratio = progress_in_cycle * args.max_mask_ratio
            args.percent.append(ratio)

    return args

if __name__ == "__main__":
    args = get_args()
    print(f"Schedule Length: {len(args.percent)}") # Should print 100
    print(f"First 15 values: {[round(x,2) for x in args.percent[:15]]}")