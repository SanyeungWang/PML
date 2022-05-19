import argparse

parser = argparse.ArgumentParser(description='Age Estimator')
"""log"""
parser.add_argument('--output_dir', type=str,
                    default='./out/imdb',
                    help='path to save networks weights')
parser.add_argument('--log_dir', type=str,
                    default='./log',
                    help='path to images for training')
parser.add_argument('--print_freq', type=int,
                    default=1,
                    help='how often are printing train log')
"""dataset"""
parser.add_argument('--train_img', type=str,
                    default='./IMDB_WIKI/train',
                    help='path to images for training')
parser.add_argument('--train_label', type=str,
                    default='./IMDB_csv/train.csv',
                    help='path to .csv file which contains labels of images for training')
parser.add_argument('--val_img', type=str,
                    default='./IMDB_WIKI/val',
                    help='path to images for test')
parser.add_argument('--val_label', type=str,
                    default='./IMDB_csv/val.csv',
                    help='path to .csv file which contains labels of images for test')
"""train"""
parser.add_argument("--lr", type=float, default=3e-4, help='learning rate 5e-4 for batchsize 300')
parser.add_argument("--multiplier", type=int, default=16, help="")
parser.add_argument("--warmup_epoch", type=int, default=50, help="")
parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--dampening', type=float, default=0, help='SGD dampening')
parser.add_argument('--nesterov', action='store_true', help='SGD nesterov')
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
parser.add_argument('--amsgrad', action='store_true', help='ADAM amsgrad')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--gamma', type=float, default=0.8, help='learning rate decay factor for step decay')
parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument("--epochs", type=int, default=300, help='number of epochs to train')
"""train parameters"""
parser.add_argument("--train_batch_size", type=int, default=800, help='resnet34 300')
parser.add_argument("--val_batch_size", type=int, default=800)
parser.add_argument("--height", type=int, default=224, help='height of input image')
parser.add_argument("--width", type=int, default=224, help='width of input image')

"""distributed"""
parser.add_argument('--rank', default=0, help='rank of current process')
parser.add_argument('--local_rank', default=0, type=int, help='rank of current process')
parser.add_argument('--word_size', default=6, help="word size")
parser.add_argument('--init_method', default='env://', help="init-method")

"""model"""
parser.add_argument("--dataset_name", type=str, default='imdb_margin', help="dataset name")
parser.add_argument("--resume", type=bool, default=True, help='load final pth')
parser.add_argument("--model_name", type=str, default='Resnet34', help='which model to train')
parser.add_argument('--nThread', type=int, default=8, help='number of threads for data loading')
parser.add_argument("--margin", type=bool, default=False, help='use margin or not')
parser.add_argument("--pretrained", type=bool, default=False, help='use pretrain or not')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'NADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | NADAM | RMSprop) finetune use SGD')
parser.add_argument("--pretrained_path", type=str,
                    default='./out/fg_001_margin_Resnet34/best.pth',
                    help='where path to load pretrained model')
args = parser.parse_args()
