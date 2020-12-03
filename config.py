import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    #workspace:
    parser.add_argument('--use_cuda', action='store', help='use cuda', default=True, type=bool)
    parser.add_argument('--save_path', action='store',help='model save path', default='trained_model/')
    parser.add_argument('--use_wandb', action='store', help='use wandb for watch', default=True, type=bool)
    parser.add_argument('--wandb_project', action='store', help='wandb project name', default='cyclegan_lightning')
    parser.add_argument('--offline', action='store', help='wandb offline mode', default=False, type=bool)
    #data
    parser.add_argument('--num_traindata', action='store', help='the number of train data', default=1000, type=int)
    parser.add_argument('--num_validdata', action='store', help='the number of valid data', default=16  , type=int)
    parser.add_argument('--size', action='store', help='data image size', default=256, type=int)
    # model
    parser.add_argument('--channel', action='store', help='rgb:3', default=3, type=int)
    parser.add_argument('--norm', action='store', help='batch|instance', default='instance')
    parser.add_argument('--dropout', action='store', help='use dropout', default=False, type=bool)
    parser.add_argument('--bias', action='store', help='use bias', default=False, type=bool)
    parser.add_argument('--initalize', action='store', help='initalize type (normal|xavier|kaiming)', default='xavier')
    parser.add_argument('--init_gain', action='store', help='initalize gain param', default=0.02, type=float)
    parser.add_argument('--paddingtype', action='store', help='zero|', default='zero')
    ## G
    parser.add_argument('--G', action='store', help='resnet', default='resnet')
    parser.add_argument('--G_filter', action='store', help="the number of G's filter", default=64, type=int)
    parser.add_argument('--G_blocks', action='store', help="the number of resnet's blocks", default=9, type=int)
    parser.add_argument('--G_mode', action='store', help="(vanilla|lsgan)", default='lsgan')
    parser.add_argument('--lambda_id', action='store', help='lambda of identity loss', default=10.0, type=float)
    parser.add_argument('--lambda_cycle', action='store', help='lambda of cycle loss', default=5.0, type=float)

    ## D
    parser.add_argument('--D_filter', action='store', help="the number of D's filter", default=64, type=int)    
    parser.add_argument('--D_layers', action='store', help="the number of D's layers", default=3, type=int)
    
    # optimizer
    parser.add_argument('--lr', action='store', help='learning rate', default=0.0002, type=float)
    parser.add_argument('--beta1', action='store', help='parameter for adam optimizer beta1', default=0.5, type=float)
    parser.add_argument('--beta2', action='store', help='parameter for adam optimizer beta2', default=0.999, type=float)
    parser.add_argument('--batch_size', action='store', help='batch size', default=1, type=int)
    parser.add_argument('--epochs', action='store', help='epochs', default=10, type=int)

    return parser

if __name__ == "__main__":
    parser = get_arguments()
    opt = parser.parse_args()
    print(opt)
    print(opt.use_cuda)
    