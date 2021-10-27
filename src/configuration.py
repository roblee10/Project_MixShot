import configargparse


def parser_args():
    parser = configargparse.ArgParser(description='PyTorch ImageNet Training')
    parser.add('-c', '--config', required=True,
               is_config_file=True, help='config file')
    ### dataset
    parser.add_argument('--data-type', type=str, choices=('miniImageNet', 'tieredImageNet'),
                        default='miniImageNet',
                        help='Data type')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--num-classes', type=int, default=64,
                        help='use all data to train the network')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--disable-train-augment', action='store_true',
                        help='disable training augmentation')
    parser.add_argument('--disable-random-resize', action='store_true',
                        help='disable random resizing')
    parser.add_argument('--enlarge', action='store_true',
                        help='enlarge the image size then center crop')
    ### network setting
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        help='network architecture')
    parser.add_argument('--scheduler', default='step', choices=('step', 'multi_step', 'cosine'),
                        help='scheduler, the detail is shown in train.py')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-stepsize', default=30, type=int,
                        help='learning rate decay step size ("step" scheduler) (default: 30)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='gamma for learning rate decay (default: 0.1)')
    parser.add_argument('--optimizer', default='SGD', choices=('SGD', 'Adam'))
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', action='store_true',
                        help='use nesterov for SGD, disable it in default')

    ### meta val setting
    parser.add_argument('--meta-test-iter', type=int, default=10000,
                        help='number of iterations for meta test')
    parser.add_argument('--meta-val-iter', type=int, default=500,
                        help='number of iterations for meta val')
    parser.add_argument('--meta-val-way', type=int, default=5,
                        help='number of ways for meta val/test')
    parser.add_argument('--meta-val-shot', type=int, default=5,
                        help='number of shots for meta val/test')
    parser.add_argument('--meta-val-query', type=int, default=15,
                        help='number of queries for meta val/test')
    parser.add_argument('--meta-val-interval', type=int, default=10,
                        help='do mate val in every D epochs')
    parser.add_argument('--meta-val-metric', type=str, choices=('euclidean', 'cosine', 'l1', 'l2'),
                        default='euclidean',
                        help='meta-val/test evaluate metric')
    parser.add_argument('--num_NN', type=int, default=1,
                        help='number of nearest neighbors, set this number >1 when do kNN')
    parser.add_argument('--eval-fc', action='store_true',
                        help='do evaluate with final fc layer.')
    ### meta train setting
    parser.add_argument('--do-meta-train', action='store_true',
                        help='do prototypical training')
    parser.add_argument('--meta-train-iter', type=int, default=100,
                        help='number of iterations for meta val')
    parser.add_argument('--meta-train-way', type=int, default=30,
                        help='number of ways for meta val')
    parser.add_argument('--meta-train-shot', type=int, default=1,
                        help='number of shots for meta val')
    parser.add_argument('--meta-train-query', type=int, default=15,
                        help='number of queries for meta val')
    parser.add_argument('--meta-train-metric', type=str, choices=('euclidean', 'cosine', 'l1', 'l2'),
                        default='euclidean',
                        help='meta-train evaluate metric')
    ### others
    parser.add_argument('--split-dir', default=None, type=str,
                        help='path to the folder stored split files.')
    parser.add_argument('--save-path', default='result/default', type=str,
                        help='path to folder stored the log and checkpoint')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--disable-tqdm', action='store_true',
                        help='disable tqdm.')
    parser.add_argument('--print-freq', '-p', default=150, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the final result')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='path to the pretrained model, used for fine-tuning')
    parser.add_argument('--logname', type=str, default='training.log',
                        help='name of the training log file')

    # Case Study
    parser.add_argument('--case-study', action='store_true',
                    help='get image file names for test')

    # Clonining
    parser.add_argument('--meta-cloning', action='store_true',
                    help='clone and augment support images when evaluating')
    parser.add_argument('--clone-factor-1shot', type=int, default=1,
                        help='how much to clone for one shot')
    parser.add_argument('--clone-factor-5shot', type=int, default=1,
                        help='how much to clone for five shot')
    parser.add_argument('--clone-augment-method', type=str, choices=('randomresizedcrop', 'rotation', 'rrc+rotation', 'none'),
                        default='randomresizedcrop',
                        help='augment method for cloning')

    # UFA
    parser.add_argument('--pointing-aug-eval', action='store_true',
                    help='eval using pointing augmentation')
    parser.add_argument('--pointing-aug-sample-type', type=str, choices=('equal_per_class', 'random_generation', 'select_class', 'none'),
                        default='select_class',
                        help='how to sample training data for pointing augmenation')
    parser.add_argument('--pointing-aug-equalperclass-num', type=int, default=1,
                    help='samples per class for equal_per_class type')
    parser.add_argument('--pointing-aug-randomgeneration-num', type=int, default=10,
                    help='sample number for random_generation')
    parser.add_argument('--pointing-aug-selectclass-num', type=int, default=1,
                    help='sample number for select_class')
    parser.add_argument('--select-class-type', type=str, choices=('far', 'close', 'random'),
                        default='random',
                        help='How to choose samples in the similar class data')
    parser.add_argument('--select-class-metric', type=str, choices=('UN', 'L2N', 'CL2N'),
                        default='UN',
                        help='Distance Metric for selecting samples in the similar class data')

    return parser.parse_args()
