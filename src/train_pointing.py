import logging
import os
import random
import shutil
import time
import warnings
import collections
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
import tqdm
import configuration
import copy
from numpy import linalg as LA
from scipy.stats import mode
from tensorboardX import SummaryWriter

import DataHandler
import models


best_prec1 = -1
global first

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# def main():
#     global args
#     args = configuration.parser_args()

#     original_lr = args.lr
#     save_path = args.save_path
#     args.lr = 0.004
#     adjust_value=0.001
#     for i in range(6):
#         args.save_path = save_path+"_"+str(args.lr)
#         sub_main()
#         args.lr= round(args.lr + adjust_value,3)

#     args.lr = original_lr
#     adjust_value=0.0001
#     for i in range(9):
#         args.lr = round(args.lr - adjust_value,4)
#         args.save_path = save_path+"_"+str(args.lr)
#         sub_main()


def main():
    # global best_prec1
    # best_prec1 = -1
    global args, best_prec1
    args = configuration.parser_args()

    ### initial logger
    log = setup_logger(args.save_path + '/'+ args.logname)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

    # TensorboardX
    writer = SummaryWriter(logdir=args.save_path + '/tensorboard')

    # create model
    log.info("=> creating model '{}'".format(args.arch))


    if args.ce_train:

        # Load Pretrained model for imabalanced training
        if args.do_imbal_train:
            model_CE = models.__dict__[args.arch](num_classes=args.num_classes + args.imbal_way, 
                                                    remove_linear=args.do_meta_train) # eliminate fully connected layer if meta-train
            
            model_CE = torch.nn.DataParallel(model_CE).cuda()
            optimizer_CE = get_optimizer(model_CE)
            
            if args.pretrain:
                pretrain = args.pretrain + '/CE_checkpoint.pth.tar' # 이 부분 수정하기 다시(args.save_path -> args.pretrain)
                if os.path.isfile(pretrain):
                    log.info("=> loading pretrained weight '{}'".format(pretrain))
                    checkpoint = torch.load(pretrain)
                    model_dict = model_CE.state_dict()
                    params = checkpoint['state_dict']
                    # temp = collections.OrderedDict()
                    # for k,v in params.items():
                    #     if k in model_dict and k != 'module.logits.weight' and k!= 'module.logits.bias'
                    params = {k: v for k, v in params.items() if k in model_dict and k != 'module.logits.weight' and k != 'module.logits.bias'}
                    model_dict.update(params)
                    model_CE.load_state_dict(model_dict)
                else:
                    log.info('[Attention]: Do not find pretrained model {}'.format(pretrain))
                    print('[Attention]: Do not find pretrained model {}'.format(pretrain))
        else:
            model_CE = models.__dict__[args.arch](num_classes=args.num_classes, remove_linear=args.do_meta_train) # eliminate fully connected layer if meta-train
            model_CE = torch.nn.DataParallel(model_CE).cuda()
            # model_CE = model_CE.to(device)
            optimizer_CE = get_optimizer(model_CE)

    if args.trp_train:
        model_TRP = models.__dict__[args.arch](num_classes=args.num_classes, remove_linear=args.do_meta_train) # eliminate fully connected layer if meta-train
        model_TRP = torch.nn.DataParallel(model_TRP).cuda()
        optimizer_TRP = get_optimizer(model_TRP)
    if args.mix_train:
        model_MIX = models.__dict__[args.arch](num_classes=args.num_classes, remove_linear=args.do_meta_train) # eliminate fully connected layer if meta-train
        model_MIX = torch.nn.DataParallel(model_MIX).cuda()
        optimizer_MIX = get_optimizer(model_MIX)

    # if args.imbal_train:
    #     model_IMBAL = models.__dict__[args.arch](num_classes=args.num_classes, remove_linear = args.do_meta_train)
    #     model_IMBAL = torch.nn.DataParallel(model_IMBAL).cuda()
    #     optimizer_IMBAL = get_optimizer(model_IMBAL)

        
    # log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model_CE.parameters()])))

    criterion_CE = nn.CrossEntropyLoss().cuda()
    criterion_TRP = nn.TripletMarginLoss().cuda() 

    # Evaluation
    if args.evaluate or args.no_meta_eval:
        if args.ce_train:
            log.info('---Cross Entropy Loss Evaluation---')
            do_extract_and_evaluate(model_CE, log)
        if args.trp_train:
            log.info('---Triplet Loss Evaluation---')
            do_extract_and_evaluate(model_TRP, log)
        if args.mix_train:
            log.info('---Mixed Loss Evaluation---')
            do_extract_and_evaluate(model_MIX, log)
        return
    if args.do_imbal_eval:
        imbalanced_evaluation(model_CE,log)
        return
    if args.pointing_aug_eval:

        pointing_augmentation_eval(model_CE, log)

        return


    if args.do_meta_train:
        sample_info = [args.meta_train_iter, args.meta_train_way, args.meta_train_shot, args.meta_train_query]
        train_loader = get_dataloader('train', not args.disable_train_augment, sample=sample_info)
    # Imbalanced Training Dataset
    elif args.do_imbal_train:
        train_loader = get_dataloader(None, not args.disable_train_augment, shuffle=True, out_name=False, imbal_data = True)
    else:
        train_loader = get_dataloader('train', not args.disable_train_augment, shuffle=True, out_name=False)
    
    # for triplet-training loader
    if args.trp_train or args.mix_train:
        triplet_info = [args.triplet_iter,args.triplet_batch, args.triplet_rand, args.triplet_base, args.triplet_same, args.triplet_diff]
        triplet_loader = get_dataloader('train', aug=False, triplet = triplet_info)

    if args.ce_train:
        scheduler_CE = get_scheduler(len(train_loader), optimizer_CE)
    if args.trp_train:
        scheduler_TRP = get_scheduler(len(triplet_loader), optimizer_TRP)
    if args.mix_train:
        scheduler_MIX = get_scheduler(len(triplet_loader), optimizer_MIX)

    # Validation loader
    sample_info = [args.meta_val_iter, args.meta_val_way, args.meta_val_shot, args.meta_val_query]
    val_loader = get_dataloader('val', False, sample=sample_info)

    # analyzation loader
    analyze_info = [args.anlz_iter, args.anlz_class, args.anlz_base, args.anlz_same, args.anlz_diff, args.anlz_rand]
    # anlz_tr_loader = get_dataloader('train', aug=False, analyze = analyze_info)
    # anlz_val_loader = get_dataloader('val', aug=False, analyze = analyze_info)

    tqdm_loop = warp_tqdm(list(range(args.start_epoch, args.epochs)))

    # mean loader
    #=================================================================================
    mean_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False)  
    #=================================================================================   

    for epoch in tqdm_loop:

        if args.ce_train:
            scheduler_CE.step(epoch)
            log.info('---Cross Entropy Loss Train---')
            #=================================================================================
            train(train_loader, model_CE, criterion_CE, optimizer_CE, epoch, scheduler_CE, log)
            #=================================================================================

        if args.trp_train:
            scheduler_TRP.step(epoch)
            log.info('---Triplet Loss Train---')
            triplet_train(triplet_loader, model_TRP, criterion_TRP, optimizer_TRP, epoch, scheduler_TRP, log)

        if args.mix_train:
            scheduler_MIX.step(epoch)
            log.info('---Mixed Loss Train---')
            triplet_train(triplet_loader, model_MIX, criterion_TRP, optimizer_MIX, epoch, scheduler_MIX, log, criterion_CE = criterion_CE)

        is_best = False

        if args.analyze and (epoch +1) % args.anlz_interval == 0:
            anlz_tr_loader = get_dataloader('train', aug=False, analyze = analyze_info)
            anlz_val_loader = get_dataloader('val', aug=False, analyze = analyze_info)
            if args.ce_train:
                log.info('---Cross Entropy Loss Distance Compare---')
                analyze(model_CE, anlz_tr_loader, anlz_val_loader, epoch=epoch, log=log, writer=writer, boardname='CE')
            if args.trp_train:
                log.info('---Triplet Loss Distance Compare---')
                analyze(model_TRP, anlz_tr_loader, anlz_val_loader, epoch=epoch, log=log, writer=writer, boardname='TRP')
            if args.mix_train:
                log.info('---Mixed Loss Distance Compare---')
                analyze(model_MIX, anlz_tr_loader, anlz_val_loader, epoch=epoch, log=log, writer=writer, boardname='MIX')

        if (epoch + 1) % args.meta_val_interval == 0:
            if  args.ce_train:
                prec1_CE = meta_val(val_loader, model_CE)
                log.info('Epoch: [{}]\t''Cross Entropy Meta Val : {}'.format(epoch, prec1_CE))

            if  args.trp_train:
                prec1_TRP = meta_val(val_loader, model_TRP)
                log.info('Epoch: [{}]\t''Triplet Meta Val : {}'.format(epoch, prec1_TRP))

            if  args.mix_train:
                prec1_MIX = meta_val(val_loader, model_MIX)
                log.info('Epoch: [{}]\t''Mixed Meta Val : {}'.format(epoch, prec1_MIX))

            # is_best = prec1 > best_prec1
            # best_prec1 = max(prec1, best_prec1)
            # if not args.disable_tqdm:
            #     tqdm_loop.set_description('Best Acc {:.2f}'.format(best_prec1 * 100.))

        if args.ce_train:
        #remember best prec@1 and save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                # 'scheduler': scheduler.state_dict()
                'arch': args.arch,
                'state_dict': model_CE.state_dict(),
                # 'best_prec1': best_prec1,
                'optimizer': optimizer_CE.state_dict(),
            }, is_best, filename='CE_checkpoint.pth.tar'  ,folder=args.save_path)

        if args.trp_train:
            save_checkpoint({
                'epoch': epoch + 1,
                # 'scheduler': scheduler.state_dict()
                'arch': args.arch,
                'state_dict': model_TRP.state_dict(),
                # 'best_prec1': best_prec1,
                'optimizer': optimizer_TRP.state_dict(),
            }, is_best, filename='TRP_checkpoint.pth.tar'  ,folder=args.save_path)

        if args.mix_train:
            save_checkpoint({
                'epoch': epoch + 1,
                # 'scheduler': scheduler.state_dict()
                'arch': args.arch,
                'state_dict': model_MIX.state_dict(),
                # 'best_prec1': best_prec1,
                'optimizer': optimizer_MIX.state_dict(),
            }, is_best, filename='MIX_checkpoint.pth.tar', folder=args.save_path)
        
    writer.close()


    # do evaluate at the end
    if  args.ce_train:
        if args.do_imbal_train: # change the variable name later
            imbalanced_evaluation(model_CE, log)
        else:
            log.info('---Cross Entropy Loss Evaluation---')
            do_extract_and_evaluate(model_CE, log)
    if  args.trp_train:
        log.info('---Triplet Loss Evaluation---')
        do_extract_and_evaluate(model_TRP, log)
    if  args.mix_train:
        log.info('---Mixed Loss Evaluation---')
        do_extract_and_evaluate(model_MIX, log)

    return


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


def metric_prediction(gallery, query, train_label, metric_type):
    gallery = gallery.view(gallery.shape[0], -1) # torch.view ->    # (25,1600) 
    query = query.view(query.shape[0], -1)                          # (75,1600)
    distance = get_metric(metric_type)(gallery, query)              # (75,25)
    predict = torch.argmin(distance, dim=1)                         # (75,1)  , choose the closes one out of 25 and save the index
    predict = torch.take(train_label, predict) # input tensor(train_label) is treated as if it were viewd as a 1-D tesnor. index is mentioned in second parameter(predict)
                                               # get the label of the closes image, predict.shape = (75,)
    return predict

# validation(few-shot)
def meta_val(test_loader, model, train_mean=None):
    top1 = AverageMeter()
    model.eval()

    with torch.no_grad():
        # assume 5 shot 5 way  15 query for each way(75 query images) 
        # convolution feature shape (1600)
        tqdm_test_loader = warp_tqdm(test_loader)
        for i, (inputs, target) in enumerate(tqdm_test_loader):
            target = target.cuda(0, non_blocking=True)
            output = model(inputs, True)[0].cuda(0) # why [0] -> if feature = True, two parameters are returned, x(not pass fc), x1(pass fc)
            if train_mean is not None:
                output = output - train_mean
            train_out = output[:args.meta_val_way * args.meta_val_shot]   # train_out.shape = (25,1600) 
            train_label = target[:args.meta_val_way * args.meta_val_shot] # train_label.shape = (25,1)
            test_out = output[args.meta_val_way * args.meta_val_shot:]    # test_out.shape =  (75,1600)
            test_label = target[args.meta_val_way * args.meta_val_shot:]  # test_label.shape = (75,1)
            
            # delete this code to just compare the closest image, not using mean
            train_out = train_out.reshape(args.meta_val_way, args.meta_val_shot, -1).mean(1) # each class feature averaged, train_out.shape = (5,1600)
            train_label = train_label[::args.meta_val_shot]

            prediction = metric_prediction(train_out, test_out, train_label, args.meta_val_metric)
            acc = (prediction == test_label).float().mean()
            top1.update(acc.item())
            if not args.disable_tqdm:
                tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg * 100))
    return top1.avg

# Training Function
def train(train_loader, model, criterion, optimizer, epoch, scheduler, log, mean_loader=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #================================================================================
    # Train Mean Loader
    if mean_loader is not None:
        # Get Train Mean
        model.eval()
        with torch.no_grad():
            output_mean = []

            for k, (temp_input, temp_label) in enumerate(warp_tqdm(mean_loader)):
                temp_outputs, _ = model(temp_input, feature = True)
                output_mean.append(temp_outputs)

            output_mean = torch.cat(output_mean, dim=0).mean(0)
            output_mean = torch.cat((output_mean,output_mean))

        if epoch == 0:
            global first
            first = copy.deepcopy(output_mean)
        
        check = first.eq(output_mean)
        check = check.view(-1).float().sum(0, keepdim=True)
        check = check.mul_(100.0 / first.view(-1).size(0))
        print('0 epoch and {}th epoch mean compare : {:.6f}%'.format(epoch ,check.item()))
        log.info('0 epoch and {}th epoch mean compare : {:.6f}%'.format(epoch ,check.item()))
        #=================================================================================


    # switch to train mode
    model.train()

    end = time.time()
    tqdm_train_loader = warp_tqdm(train_loader)
    for i, (input, target, ) in enumerate(tqdm_train_loader):
        # learning rate scheduler
        if args.scheduler == 'cosine':
            scheduler.step(epoch * len(train_loader) + i)

        if args.do_meta_train:
            target = torch.arange(args.meta_train_way)[:, None].repeat(1, args.meta_train_query).reshape(-1).long()
        target = target.cuda(non_blocking=True)

        # compute output
        r = np.random.rand(1)
        output = model(input)

        if args.do_meta_train:
            output = output.cuda(0)
            # assume 5-shot 5-way 15 query
            shot_proto = output[:args.meta_train_shot * args.meta_train_way] 
            query_proto = output[args.meta_train_shot * args.meta_train_way:] # shape = (75,feature size)
            shot_proto = shot_proto.reshape(args.meta_train_way, args.meta_train_shot, -1).mean(1) # shape = (5,feature size) -> same class features are averaged
            output = -get_metric(args.meta_train_metric)(shot_proto, query_proto) # shape = (75,5)

        
        loss = criterion(output, target)# When meta training 
                                        # output = (75,5), target = (75), since the output is distance, not probability distribution, use minus in output 
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1, _ = accuracy(output, target, topk=(1,5))

        top1.update(prec1[0], input.size(0))
        if not args.disable_tqdm: # 
            tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # top1.val : accuracy of the last batch
        # top1.avg : average accuracy for each epoch
        if (i+1) % args.print_freq == 0:
            log.info('Epoch: [{0}]\t'
                     'Time {batch_time.sum:.3f}\t'
                     'Loss {loss.avg:.4f}\t'
                     'Prec: {top1.avg:.3f}'.format( epoch, batch_time=batch_time, loss=losses, top1=top1))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', folder='result/default'):
    torch.save(state, folder + '/' + filename)
    if is_best: # only save when validation has best accuracy
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val 
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)  # get top k labels
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # returns two value (top k value, top k value labels)
        pred = pred.t()  # transpose, EX) (256,5) to (5,256) assuming top 5
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # compare predicted label and actual label

        res = []
        correct = correct.contiguous()
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger(filepath)
    # handler = logging.StreamHandler()
    # handler.setFormatter(file_formatter)
    # logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) != '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger

def get_scheduler(batches, optimiter):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    SCHEDULER = {'step': StepLR(optimiter, args.lr_stepsize, args.lr_gamma),  # Reduces lr by gamma ratio for each step
                 'multi_step': MultiStepLR(optimiter, milestones=[int(.5 * args.epochs), int(.75 * args.epochs)], # similar to StepLR, but decreases gamma ratio only in the designated epoch, not in every step.
                                           gamma=args.lr_gamma),
                 'cosine': CosineAnnealingLR(optimiter, batches * args.epochs, eta_min=1e-9)} # change learning rate like a cosine graph
    return SCHEDULER[args.scheduler]

def get_optimizer(module):
    # SGD lr : 0.1
    # Adam lr : 0.001
    OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                        nesterov=args.nesterov),
                 'Adam': torch.optim.Adam(module.parameters(), lr=args.lr)}
    return OPTIMIZER[args.optimizer]

def extract_feature(train_loader, val_loader, model, tag='last'):
    # return out mean, fcout mean, out feature, fcout features
    model.eval()
    with torch.no_grad():
        # get training mean
        out_mean, fc_out_mean = [], []

    # Training Data
    # Centering requires means feature vector of the training class
        for i, (inputs, _) in enumerate(warp_tqdm(train_loader)):

            outputs, fc_outputs = model(inputs, True)
            out_mean.append(outputs.cpu().data.numpy())
            if fc_outputs is not None:
                fc_out_mean.append(fc_outputs.cpu().data.numpy())
        out_mean = np.concatenate(out_mean, axis=0).mean(0)
        if len(fc_out_mean) > 0:
            fc_out_mean = np.concatenate(fc_out_mean, axis=0).mean(0)
        else:
            fc_out_mean = -1

        output_dict = collections.defaultdict(list)
        fc_output_dict = collections.defaultdict(list)
        output_idx = collections.defaultdict(list)

        # Validation(Test) Data
        # Save each feature in dictionary. use fc layer output if necessary
        if args.case_study:

            for i, (inputs, labels, test_index) in enumerate(warp_tqdm(val_loader)):
                # compute output
                outputs, fc_outputs = model(inputs, True)
                outputs = outputs.cpu().data.numpy()
                if fc_outputs is not None:
                    fc_outputs = fc_outputs.cpu().data.numpy()
                else:
                    fc_outputs = [None] * outputs.shape[0]
                for out, fc_out, label, idx in zip(outputs, fc_outputs, labels, test_index):
                    output_dict[label.item()].append(out)
                    output_idx[label.item()].append(idx)
                    fc_output_dict[label.item()].append(fc_out)
            all_info = [out_mean, fc_out_mean, output_dict, fc_output_dict, output_idx]
        
        else:
            for i, (inputs, labels) in enumerate(warp_tqdm(val_loader)):
                # compute output
                outputs, fc_outputs = model(inputs, True)
                outputs = outputs.cpu().data.numpy()
                if fc_outputs is not None:
                    fc_outputs = fc_outputs.cpu().data.numpy()
                else:
                    fc_outputs = [None] * outputs.shape[0]
                for out, fc_out, label in zip(outputs, fc_outputs, labels):
                    output_dict[label.item()].append(out)
                    fc_output_dict[label.item()].append(fc_out)
            all_info = [out_mean, fc_out_mean, output_dict, fc_output_dict]

        return all_info

def get_dataloader(split, aug=False, shuffle=True, out_name=False, sample=None, analyze=None, triplet = None, imbal_data = False, resample = None):
    # sample: iter, way, shot, query

    if aug:
        transform = DataHandler.with_augment(84, disable_random_resize=args.disable_random_resize)
    elif aug is None:
        transform = None
        sets = DataHandler.DatasetFolder(args.data, args.split_dir, split, transform, out_name=out_name)
        sampler = DataHandler.CategoriesSampler(sets.labels, *sample)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.workers,collate_fn = my_collate, pin_memory=True) # shuffle needs to be false
        return loader
    else:
        transform = DataHandler.without_augment(84, enlarge=args.enlarge) 

    if imbal_data:
        sets = DataHandler.ImbalancedDataset(args.data, args.split_dir, transform, args.imbal_shot, args.imbal_way, out_name=out_name)
    elif resample:
        sets = DataHandler.DatasetResampler(args.data, args.split_dir, split, *resample, transform, out_name=out_name)
    else:
        sets = DataHandler.DatasetFolder(args.data, args.split_dir, split, transform, out_name=out_name)
    
    # For Meta-training
    # defaulte 400 iteration, each 100(75query 15support)
    if sample is not None:
        sampler = DataHandler.CategoriesSampler(sets.labels, *sample)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.workers, pin_memory=True) # shuffle needs to be false
    elif analyze is not None:
        sampler = DataHandler.AnalyzeSampler(sets.labels, *analyze)
        loader = torch.utils.data.DataLoader(sets, batch_sampler = sampler,
                                             num_workers=args.workers, pin_memory = True)
    elif triplet is not None:
        sampler = DataHandler.TripletSampler(sets.labels,*triplet)
        loader = torch.utils.data.DataLoader(sets, batch_sampler = sampler,
                                             num_workers=args.workers, pin_memory = True)
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.workers, pin_memory=True)
    return loader

# def my_collate(batch):
#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
#     target = torch.LongTensor(target)
#     return [data, target]


def my_collate(batch):
    sup_pipeline = aug_pipeline(84, disable_random_resize=args.disable_random_resize)
    query_pipeline = noaug_pipeline(84, enlarge=args.enlarge )
    
    Sup = batch[:100]
    Query = batch[100:]
    Support = [query_pipeline(item[0]) for item in Sup]
    Query = [sup_pipeline(item[0]) for item in Query]

    Support = torch.stack(Support)
    Query = torch.stack(Query)
    data = torch.cat((Support,Query))

    # data = [sup_pipeline(item[0]) for item in batch]
    # data = torch.stack(data)

    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]



def warp_tqdm(data_loader):
    if args.disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm.tqdm(data_loader, total=len(data_loader))
    return tqdm_loader

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def load_checkpoint(model, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(args.save_path))
    elif type == 'last':
        # checkpoint = torch.load('{}/CE_checkpoint.pth.tar'.format(args.save_path))
        checkpoint = torch.load('{}/CE_checkpoint.pth.tar'.format(args.save_path))
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    model.load_state_dict(checkpoint['state_dict'])

def load_pretrained(model, log):
    pretrain = args.pretrain + '/CE_checkpoint.pth.tar' # 이 부분 수정하기 다시(args.save_path -> args.pretrain)
    if os.path.isfile(pretrain):
        log.info("=> loading pretrained weight '{}'".format(pretrain))
        checkpoint = torch.load(pretrain)
        model_dict = model.state_dict()
        params = checkpoint['state_dict']
        params = {k: v for k, v in params.items() if k in model_dict}
        model_dict.update(params)
        model.load_state_dict(model_dict)
    else:
        log.info('[Attention]: Do not find pretrained model {}'.format(pretrain))
        print('[Attention]: Do not find pretrained model {}'.format(pretrain))

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm
 

def noaug_pipeline(size=84, enlarge=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) 

    if enlarge:
        resize = int(size*256./224.)
    else:
        resize = size
    return transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])

def aug_pipeline(size=84, disable_random_resize=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) 

    if disable_random_resize:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(), # default = 0.5
            # transforms.RandomRotation(degrees=(30,-30)),
            transforms.ToTensor(),
            normalize,
        ])



def v2_eval_prediction(output, target, shot, Normalize=False, train_mean=None, clone_factor=1):

    if train_mean is not None:
        output = output - train_mean
    if Normalize:
        output = F.normalize(output,dim=1,p=2)

    train_out = output[:args.meta_val_way * shot * clone_factor]   # train_out.shape = (25,1600) 
    train_label = target[:args.meta_val_way * shot * clone_factor] # train_label.shape = (25,1)
    test_out = output[args.meta_val_way * shot * clone_factor:]    # test_out.shape =  (75,1600)
    test_label = target[args.meta_val_way * shot * clone_factor:]  # test_label.shape = (75,1)

    # delete this code to just compare the closest image, not using mean
    train_out = train_out.reshape(args.meta_val_way, shot * clone_factor, -1).mean(1) # each class feature averaged, train_out.shape = (5,1600)
    train_label = train_label[::shot * clone_factor]

    prediction = metric_prediction(train_out, test_out, train_label, args.meta_val_metric)
    acc = (prediction == test_label).float().mean()
    acc = acc.cpu().detach().numpy()

    return acc

# validation(few-shot)
def meta_eval_v2(train_loader, test_loader, model, shot, clone_factor=1):
    un_list = []
    l2n_list = []
    cl2n_list = []

    model.eval()
    
    # 1. Get train Mean for test set
    with torch.no_grad():
        train_mean = []

        for k, (train_input, train_label) in enumerate(warp_tqdm(train_loader)):
            train_outputs, _ = model(train_input, feature = True)
            train_mean.append(train_outputs)

        train_mean = torch.cat(train_mean, dim=0).mean(0)
        # train_mean = torch.cat((train_mean, train_mean))

    # 2. Train Mean 이용해 UN, L2N, CL2N 추출하기
    with torch.no_grad():
        tqdm_test_loader = warp_tqdm(test_loader)
        sup_pipeline = aug_pipeline(84, disable_random_resize=args.disable_random_resize)
        query_pipeline = noaug_pipeline(84, enlarge=args.enlarge )
        for i, (inputs, target) in enumerate(tqdm_test_loader):  # DataLoader Transform 부분으로 인한 오류
            target = target.cuda(0, non_blocking=True)
 
            # Get Output
            output = model(inputs, True)[0].cuda(0) # why [0] -> if feature = True, two parameters are returned, x(not pass fc), x1(pass fc)

            # Prediction
            un_list.append(v2_eval_prediction(output, target, shot, clone_factor=clone_factor))
            l2n_list.append(v2_eval_prediction(output, target, shot, Normalize=True, clone_factor=clone_factor))
            cl2n_list.append(v2_eval_prediction(output, target, shot, Normalize=True, train_mean = train_mean, clone_factor=clone_factor))

    #3. 95% confidence interval
    un_mean, un_conf = compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)

    #4. Return
    return [un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf]

def meta_evaluate_case_study(data, train_mean, out_idx, shot):
    un_list = []
    l2n_list = []
    cl2n_list = []

    episode_train = collections.defaultdict(list) # 여기에 train 에피소드 인덱스들 0~9999(만개 이므로) 에 각 5shot 이면 25개, 1shot 이면 5개 담고
    episode_test =  collections.defaultdict(list)
    episode_true = {'cl2n' : collections.defaultdict(list), 'l2n' : collections.defaultdict(list), 'un' : collections.defaultdict(list)}   # cases that are wrong in query set 
    episode_false = {'cl2n' : collections.defaultdict(list), 'l2n' : collections.defaultdict(list), 'un' : collections.defaultdict(list)}    # cases that are correct in query set

    for i in warp_tqdm(range(args.meta_test_iter)):
        train_data, test_data, train_label, test_label, train_idx, test_idx = sample_case_study(data, out_idx, shot) #25,75(5shot 15 query)

        episode_train[i] = train_idx
        episode_test[i] = test_idx

        test_idx=np.array(test_idx)
        # Centering + L2 normalization
        # normalizes the feature(not image itself)
        acc, episode_result = metric_class_type(train_data, test_data, train_label, test_label, 
                                                shot, train_mean=train_mean, norm_type='CL2N')
        cl2n_list.append(acc)
        episode_true['cl2n'][i] = list(test_idx[(episode_result)])
        episode_false['cl2n'][i] = list(test_idx[~episode_result])


        # L2 normalization
        acc, episode_result = metric_class_type(train_data, test_data, train_label, test_label, 
                                                shot, train_mean=train_mean, norm_type='L2N')
        l2n_list.append(acc)
        episode_true['l2n'][i] = list(test_idx[(episode_result)])
        episode_false['l2n'][i] =  list(test_idx[(~episode_result)])

        # Unormalized
        acc, episode_result = metric_class_type(train_data, test_data, train_label, test_label,
                                                shot, train_mean=train_mean, norm_type='UN')
        un_list.append(acc)
        episode_true['un'][i] = list(test_idx[(episode_result)]) 
        episode_false['un'][i] =  list(test_idx[(~episode_result)])

    un_mean, un_conf = compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    return [un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf], [episode_train, episode_test, episode_true, episode_false]



def meta_evaluate(data, train_mean, shot):
    un_list = []
    l2n_list = []
    cl2n_list = []
    for _ in warp_tqdm(range(args.meta_test_iter)):
        train_data, test_data, train_label, test_label = sample_case(data, shot)

        # Centering + L2 normalization
        acc = metric_class_type(train_data, test_data, train_label, test_label, shot, train_mean=train_mean,
                                norm_type='CL2N')
        cl2n_list.append(acc)
        
        # L2 normalization
        acc = metric_class_type(train_data, test_data, train_label, test_label, shot, train_mean=train_mean,
                                norm_type='L2N')
        l2n_list.append(acc)

        # Unormalized
        acc = metric_class_type(train_data, test_data, train_label, test_label, shot, train_mean=train_mean,
                                norm_type='UN')
        un_list.append(acc)
    un_mean, un_conf = compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    return [un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf]


def metric_class_type(gallery, query, train_label, test_label, shot, train_mean=None, norm_type='CL2N', clone_factor=1):
    if norm_type == 'CL2N': # subtract train mean on both support and query set and normalize
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query - train_mean
        query = query / LA.norm(query, 2, 1)[:, None]
    elif norm_type == 'L2N': # just normalize
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]

    # delete this code to just compare the closest image, not using mean
    gallery = gallery.reshape(args.meta_val_way, shot * clone_factor, gallery.shape[-1]).mean(1)
    train_label = train_label[::shot * clone_factor]

    # assume 5 shot 5 way
    subtract = gallery[:, None, :] - query 
    distance = LA.norm(subtract, 2, axis=-1) # get euclidean distance between support and query (L2 norm)  , shape = (25,75)
    idx = np.argpartition(distance, args.num_NN, axis=0)[:args.num_NN] # np.argpartition : get the index of the smallest distance in the list , num_NN: number of nearest neighbor, shape = (1,75)
                                                                       # if axis=0, it compares by columns. So it gives closest index between 25 images. argpartition arranges all list into Ascending order, 
                                                                       # but here takes only the first list which is the closest index list 
    nearest_samples = np.take(train_label, idx)  # input array is treated as if it were viewd as a 1-D tesnor. index is mentioned in second parameter 
    out = mode(nearest_samples, axis=0)[0]
    out = (out.astype(int)).reshape(-1)
    test_label = np.array(test_label)
    result = (out==test_label)  # Get the result of the prediction EX) [True,False,False,.....]
    acc = result.mean() # Get the accuracy for each episode
    if args.case_study:
        return acc, result
    else:
        return acc
# Sample data to meta-test format

def sample_case_study(ld_dict, out_idx, shot):
    sample_class = random.sample(list(ld_dict.keys()), args.meta_val_way)  # key of dict is label of the data, get 5 random label
    train_input = []
    test_input = []
    test_label = []
    train_label = []
    train_idx = []
    test_idx = []
    for each_class in sample_class:
        sample_number = random.sample(list(range(len(ld_dict[each_class]))), shot + args.meta_val_query)  # get each index of the sampled items
        samples = [ld_dict[each_class][x] for x in sample_number]
        samples_idx = [out_idx[each_class][x] for x in sample_number]
        
        # samples = random.sample(ld_dict[each_class], shot + args.meta_val_query) # each class has 20 images(5shot, 15query)

        train_label += [each_class] * len(samples[:shot])
        test_label += [each_class] * len(samples[shot:])
        train_input += samples[:shot]
        test_input += samples[shot:]
        train_idx += samples_idx[:shot]
        test_idx += samples_idx[shot:]

    train_input = np.array(train_input).astype(np.float32)  # 25 
    test_input = np.array(test_input).astype(np.float32)    # 75
    return train_input, test_input, train_label, test_label, train_idx, test_idx

def sample_case(ld_dict, shot):
    sample_class = random.sample(list(ld_dict.keys()), args.meta_val_way)
    train_input = []
    test_input = []
    test_label = []
    train_label = []
    for each_class in sample_class:
        samples = random.sample(ld_dict[each_class], shot + args.meta_val_query)
        train_label += [each_class] * len(samples[:shot])
        test_label += [each_class] * len(samples[shot:])
        train_input += samples[:shot]
        test_input += samples[shot:]
    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    return train_input, test_input, train_label, test_label

def do_extract_and_evaluate(model, log):
    if args.no_meta_eval:
        fc_evaluate(model,log)
        return

    load_checkpoint(model, 'last')
    train_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False)

    if(args.case_study):
        val_loader = get_dataloader('test', aug=False, shuffle=False, out_name=True)
        out_mean, fc_out_mean, out_dict, fc_out_dict, out_idx = extract_feature(train_loader, val_loader, model, 'last')
        accuracy_info_shot1, episode_shot1 = meta_evaluate_case_study(out_dict, out_mean, out_idx, 1)
        accuracy_info_shot5, episode_shot5 = meta_evaluate_case_study(out_dict, out_mean, out_idx, 5)

        save_pickle('episode_shot1.pkl', episode_shot1)
        save_pickle('episode_shot5.pkl', episode_shot5)

    elif(args.meta_cloning):
        # Validation loader
                        # test iteration,       test way,      test shot,       test query,    test clone factor
        sample_info = [args.meta_test_iter, args.meta_val_way,    1,       args.meta_val_query,      20]
        test_loader = get_dataloader('test', aug=None, shuffle=False, out_name=False, sample=sample_info)
        accuracy_info_shot1 = meta_eval_v2(train_loader, test_loader, model, shot=1, clone_factor=20)

                        # test iteration,       test way,      test shot,       test query,    test clone factor
        sample_info = [args.meta_test_iter, args.meta_val_way,    5,       args.meta_val_query,      4]
        test_loader = get_dataloader('test', aug=None, shuffle=False, out_name=False, sample=sample_info)
        accuracy_info_shot5 = meta_eval_v2(train_loader, test_loader, model, shot=5, clone_factor=4)

    elif(args.pointing_augmentation):
        test_loader = get_dataloader('test', aug=False, shuffle=False, out_name=True)

    else:
        val_loader = get_dataloader('test', aug=False, shuffle=False, out_name=False)
        out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'last')
        accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, 1)
        accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, 5)
        accuracy_info_shot20 = meta_evaluate(out_dict, out_mean, 20)
        accuracy_info_shot50 = meta_evaluate(out_dict, out_mean, 50)

    # print(
    #     'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'
    #     .format('GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5,'GVP 20Shot', *accuracy_info_shot20, 'GVP_50Shot', *accuracy_info_shot50 ))
    # log.info(
    #     'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'
    #     .format('GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5,'GVP 20Shot', *accuracy_info_shot20, 'GVP_50Shot', *accuracy_info_shot50 ))
    print(
        'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'
        .format('GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    log.info(
        'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'
        .format('GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
   
   
   
   
    # if args.eval_fc: # Use logit for Nearest Neighbor
    #     accuracy_info = meta_evaluate(fc_out_dict, fc_out_mean, 1)
    #     print('{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format('Logits', *accuracy_info))
    #     log.info('{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format('Logits', *accuracy_info))

    # load_checkpoint(model, 'best')
    # out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'best')
    # accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, 1)
    # accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, 5)
    # print(
    #     'Meta Test: BEST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
    #         'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    # log.info(
    #     'Meta Test: BEST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
    #         'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    # if args.eval_fc:
    #     accuracy_info = meta_evaluate(fc_out_dict, fc_out_mean, 1)
    #     print('{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format('Logits', *accuracy_info))
    #     log.info('{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format('Logits', *accuracy_info))


def do_extract_and_validate(model, log):

    train_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False)
    val_loader = get_dataloader('val', aug=False, shuffle=False, out_name=False)
    # load_checkpoint(model, 'last')

    # Extract the average of the output. When using FC Layer, it extracts the average of FC Layer output
    out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'last')
    accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, 1)
    # accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, 5)
    print(
        'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1))
    log.info(
        'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1))
    # return accuracy_info_shot1


def fc_evaluate(model, log):
    test_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False)

    load_checkpoint(model, 'last')
    acc_info = get_fc_result(model,test_loader)
    print('FC Test: LAST\n'
            'accuracy: {acc_info.avg:.4f}\t'
            'test size: {acc_info.count}'.format(acc_info=acc_info))
    log.info('\nFC Test: LAST\n'
            'accuracy: {acc_info.avg:.4f}\t'
            'test size: {acc_info.count}'.format(acc_info=acc_info))

    load_checkpoint(model, 'best')
    acc_info = get_fc_result(model,test_loader)
    print('FC Test: BEST\n'
            'accuracy: {acc_info.avg:.4f}\t'
            'test size: {acc_info.count}'.format(acc_info=acc_info))
    log.info('\nFC Test: BEST\n'
            'accuracy: {acc_info.avg:.4f}\t'
            'test size: {acc_info.count}'.format(acc_info=acc_info))

def get_fc_result(model,test_loader):
    top1 = AverageMeter()
    model.eval()
    with torch.no_grad():
        tqdm_test_loader = warp_tqdm(test_loader)
        for i, (input, target) in enumerate(tqdm_test_loader):
            target = target.cuda(non_blocking=True)
            output = model(input)
            prec1, _ = accuracy(output, target, topk=(1,5))
            top1.update(prec1[0],input.size(0))
            if not args.disable_tqdm: 
                tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg))
    return top1

def distance_compare(model,loader,feature = True):
    same = AverageMeter()
    diff = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (inputs,_) in enumerate(warp_tqdm(loader)):
            if feature:
                output = model(inputs, feature = feature)[0].cuda(0)
            else:
                output = model(inputs, feature = feature).cuda(0)
            # L2 normalization
            # 파이토치에 normalize 해주는 함수 존재한다.
            # normalize 해주는 함수 써서 해보기
            # output = output / torch.norm(output,2,1)[:, None]
            output = F.normalize(output,dim=1,p=2)


            base_out = output[:args.anlz_base]
            same_out = output[args.anlz_base : args.anlz_base + args.anlz_same]
            diff_out = output[args.anlz_base + args.anlz_same : args.anlz_base + args.anlz_same + args.anlz_diff]

            # distance = get_metric(args.anlz_metric)(diff_out,base_out) # (4,1600), (1,1600) -> (1,4)
            # diff_idx = torch.argmin(distance, dim=1).item()
            # diff_out = diff_out[diff_idx : diff_idx+1]

            # print(_)
            # print(inputs.shape)
            same_distance = get_metric(args.anlz_metric)(same_out,base_out)
            diff_distance = get_metric(args.anlz_metric)(diff_out,base_out)
            # print(same_distance)
            # print(diff_distance)
            same_distance, diff_distance = same_distance.reshape(-1).mean(), diff_distance.reshape(-1).mean()

            # positive_distance = torch.norm((same_out-base_out),p=2,dim=1)
            # negative_distance = torch.norm((diff_out-base_out),p=2,dim=1)
            # print(positive_distance)
            # print(negative_distance)
        
            same.update(same_distance.detach().cpu().numpy())
            diff.update(diff_distance.detach().cpu().numpy())
    return same,diff

def analyze(model, anlz_tr_loader,anlz_val_loader, epoch, log, writer, boardname):
    tr_same, tr_diff = distance_compare(model,anlz_tr_loader)
    log.info('Epoch: [{}]\t''Tr  Dist Compare: Same {:.4f} Diff {:.4f}'.format(epoch, tr_same.avg, tr_diff.avg))
    val_same, val_diff = distance_compare(model,anlz_val_loader)
    log.info('Epoch: [{}]\t''Val Dist Compare: Same {:.4f} Diff {:.4f}'.format(epoch, val_same.avg, val_diff.avg))

    writer.add_scalars('No_FC/tr_distance_'+boardname, {'same':tr_same.avg, 'diff': tr_diff.avg} ,epoch+1)
    writer.add_scalars('No_FC/val_distance_'+boardname, {'same':val_same.avg, 'diff': val_diff.avg} ,epoch+1)
    writer.add_scalars('No_FC/tr_distance_interval', 
                        {boardname +'_interval': tr_diff.avg - tr_same.avg},epoch+1)
    writer.add_scalars('No_FC/val_distance_interval', 
                        {boardname +'_interval': val_diff.avg - val_same.avg},epoch+1)
    
    if args.anlz_FC:
        FC_tr_same, FC_tr_diff = distance_compare(model,anlz_tr_loader,feature = False)
        FC_val_same, FC_val_diff = distance_compare(model,anlz_val_loader,feature = False)
        writer.add_scalars('With_FC/tr_distance_'+boardname, {'same':FC_tr_same.avg, 'diff': FC_tr_diff.avg} ,epoch+1)
        writer.add_scalars('With_FC/val_distance_'+boardname, {'same':FC_val_same.avg, 'diff': FC_val_diff.avg} ,epoch+1)
        writer.add_scalars('With_FC/tr_distance_interval',
                            {boardname +'_interval': FC_tr_diff.avg - FC_tr_same.avg} ,epoch+1)
        writer.add_scalars('With_FC/val_distance_interval',
                            {boardname +'_interval': FC_val_diff.avg - FC_val_same.avg} ,epoch+1)


def triplet_loss(anchor,positive,negative,margin=1.0):
    positive_distance = torch.norm((anchor-positive),p=2,dim=1)
    negative_distance = torch.norm((anchor-negative),p=2,dim=1)

    loss = torch.sum(positive_distance - negative_distance + margin)
    return F.relu(loss)

def triplet_train(train_loader, model, criterion_TRP, optimizer, epoch, scheduler, log, criterion_CE=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()
    tqdm_train_loader = warp_tqdm(train_loader)

    for i, (input, target) in enumerate(tqdm_train_loader):
        output, output_FC = model(input, feature = True)
        output = output.cuda(0)

        # L2 normalization
        # 파이토치에 normalization 함수 존재한다
        # 그럼 triple loss 는 넣었을때 자동으로 L2 normalization 해주는 건가?
        # 그럼 왜 Normlize 하고 넣었을때랑 안하고 넣었을때랑 결과가 다른건가?
        # 일단 해야할게 있으니 나중에 시간되면 시도해보기

        # output = F.normalize(output,dim=1,p=2)
        # output = output / torch.norm(output, 2, 1)[:, None]                 # detach is a breakpoint so that no gradient will flow above this point.
        
        anchor = output[:args.triplet_batch*args.triplet_base]
        positive = output[args.triplet_batch*args.triplet_base : args.triplet_batch*(args.triplet_base+args.triplet_same)]
        negative = output[args.triplet_batch*(args.triplet_base+args.triplet_same) : args.triplet_batch*(args.triplet_base+args.triplet_same+args.triplet_diff)]

        # anchor = anchor / torch.norm(anchor, 2, 1)[:, None]    # [:,None] : creates an axis with length 1
        # positive = positive / torch.norm(positive, 2, 1)[:, None]
        # negative = negative / torch.norm(negative, 2, 1)[:, None]

        # get nearest distance 

        # negative_idx = []
        # for j in range(args.triplet_batch): 
        #     distance = get_metric(args.triplet_metric)(negative[j*args.triplet_diff : (j+1)*args.triplet_diff] , anchor[j:j+1]) # (4,1600), (1,1600) -> (1,4)
        #     negative_idx.append((j*args.triplet_diff) + torch.argmin(distance, dim=1))
        # negative = negative[negative_idx]

        # positive = positive.reshape(args.triplet_batch, args.triplet_same, -1).mean(1)
        # negative = negative.reshape(args.triplet_batch, args.triplet_diff, -1).mean(1)
        loss = criterion_TRP(anchor,positive,negative)

        if criterion_CE != None :
            output_FC = output_FC.cuda(0)
            target = target.cuda(non_blocking=True)
            loss_CE= criterion_CE(output_FC,target)
            loss = loss + loss_CE
        losses.update(loss.item()) # need to use .item to clear the gpu memory allocated in loss
        optimizer.zero_grad()
        loss.backward()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        optimizer.step()

        # prec1, _ = accuracy(output_FC, target, topk=(1,5))
        # top1.update(prec1[0], input.size(0))
        # if not args.disable_tqdm: # 
        #     tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            log.info('Epoch: [{0}]\t'
                     'Time {batch_time.sum:.3f}\t'
                     'Loss {loss.avg:.4f}\t'.format( epoch, batch_time=batch_time, loss=losses))


def imbalanced_evaluation(model,log):

    sample_info = [args.imbal_iter, args.imbal_way, args.imbal_shot, args.meta_val_query]
                # [iteration, way, shot, query]
    test_loader = get_dataloader('test', False, out_name=True, sample = sample_info)
    tqdm_test_loader = warp_tqdm(test_loader)

    # ======================================
    # param_original = model.state_dict()
    original_model = model
    # ======================================

    test_top1 = AverageMeter()

    label_train = []
    label_test = []
    for j in range(args.imbal_way):
        label_train += [(args.num_classes + j)] * args.imbal_shot
        label_test += [(args.num_classes + j)] * args.meta_val_query
    # label_train = torch.tensor(target_train, dtype=torch.int64)
    # label_test = torch.tensor(target_test, dtype = torch.int64)

    for i, (inputs, _ , filename) in enumerate(tqdm_test_loader):
        # Get model
        ind_model = copy.deepcopy(original_model) 

        # training criterion
        criterion = nn.CrossEntropyLoss().cuda() 

        # ind_model = original_model
        # param_copy = ind_model.state_dict()

        # #=============================================================================================================================
        # # Parameter Check
        # CE_model = models.__dict__[args.arch](num_classes=args.num_classes + args.imbal_way, 
        #                                         remove_linear=args.do_meta_train) # eliminate fully connected layer if meta-train
        # CE_model = torch.nn.DataParallel(CE_model).cuda()
        # optimizer_CE = get_optimizer(CE_model)
        
        # pretrain = args.pretrain + '/CE_checkpoint.pth.tar' # 이 부분 수정하기 다시(args.save_path -> args.pretrain)
        # if os.path.isfile(pretrain):
        #     # log.info("=> loading pretrained weight '{}'".format(pretrain))
        #     checkpoint = torch.load(pretrain)
        #     model_dict = CE_model.state_dict()
        #     params = checkpoint['state_dict']
        #     params = {k: v for k, v in params.items() if k in model_dict and k != 'module.logits.weight' and k != 'module.logits.bias'}
        #     model_dict.update(params)
        #     CE_model.load_state_dict(model_dict)
        # else:
        #     log.info('[Attention]: Do not find pretrained model {}'.format(pretrain))
        #     print('[Attention]: Do not find pretrained model {}'.format(pretrain))

        # param_loaded = CE_model.state_dict()

        # # param_original, param_copy
        # print("===========================================================")
        # print("Original vs Shallow Copy")
        # log.info("===========================================================")
        # log.info("Original vs Shallow Copy")
        # for (name1,param1),(name2,param2) in zip(param_original.items(), param_copy.items()):
        #     param_check = param1.eq(param2)
        #     param_check = param_check.view(-1).float().sum(0, keepdim=True)
        #     param_check = param_check.mul_(100.0 / param1.view(-1).size(0))
        #     print("[{}] Parameter Match Rate : {:.2f} %".format(name1, param_check.item()))
        #     log.info("[{}] Parameter Match Rate : {:.2f} %".format(name1, param_check.item()))
        # print("===========================================================")
        # log.info("===========================================================")
        
        # # param_original, param_loaded
        # print("===========================================================")
        # print("Original vs Loaded")
        # log.info("===========================================================")
        # log.info("Original vs Loaded")
        # for (name1,param1),(name2,param2) in zip(param_original.items(), param_loaded.items()):
        #     param_check = param1.eq(param2)
        #     param_check = param_check.view(-1).float().sum(0, keepdim=True)
        #     param_check = param_check.mul_(100.0 / param1.view(-1).size(0))
        #     print("[{}] Parameter Match Rate : {:.2f} %".format(name1, param_check.item()))
        #     log.info("[{}] Parameter Match Rate : {:.2f} %".format(name1, param_check.item()))
        # print("===========================================================")
        # log.info("===========================================================")
        
        # # param_copy, param_loaded
        # print("===========================================================")
        # print("Shallow Copy vs Loaded")
        # log.info("===========================================================")
        # log.info("Shallow Copy vs Loaded")
        # for (name1,param1),(name2,param2) in zip(param_copy.items(), param_loaded.items()):
        #     param_check = param1.eq(param2)
        #     param_check = param_check.view(-1).float().sum(0, keepdim=True)
        #     param_check = param_check.mul_(100.0 / param1.view(-1).size(0))
        #     print("[{}] Parameter Match Rate : {:.2f} %".format(name1, param_check.item()))
        #     log.info("[{}] Parameter Match Rate : {:.2f} %".format(name1, param_check.item()))
        # print("===========================================================")
        # log.info("===========================================================")

        # #=============================================================================================================================

        # Get dataset
        support_len = args.imbal_way * args.imbal_shot
        query_len = args.imbal_way * args.meta_val_query
        input_train = inputs[0: support_len]
        input_test = inputs[support_len : support_len + query_len]
        filename_train = filename[0 : support_len]

        resample_info = [filename_train, label_train, args.imbal_shot, args.imbal_way]
        train_loader = get_dataloader('train', not args.disable_train_augment , shuffle=True, out_name=False, resample = resample_info )

        # Train Mean Loader
        mean_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False, resample = resample_info)
        _cnt_ = 0

        # get optimizer and scheduler
        optimizer = get_optimizer(ind_model)
        scheduler = get_scheduler(len(train_loader), optimizer) 
        
        for tr_epoch in range(args.epochs):
            tr_top1 = AverageMeter()

            #=================================================================================
            # Get Train Mean

            ind_model.eval()
            with torch.no_grad():
                output_mean = [] 
                for k, (temp_input, temp_label) in enumerate(warp_tqdm(mean_loader)):
                    temp_outputs, _ = ind_model(temp_input, feature = True)
                    output_mean.append(temp_outputs)
                output_mean = torch.cat(output_mean, dim=0).mean(0)
                output_mean = torch.cat((output_mean,output_mean))

            if _cnt_ == 0:
                global first
                first = copy.deepcopy(output_mean)

            _cnt_ +=1
            check = first.eq(output_mean)
            check = check.view(-1).float().sum(0, keepdim=True)
            check = check.mul_(100.0 / first.view(-1).size(0))
            # print('1 epoch and {}th epoch mean compare : {:.6f}%'.format(_cnt_ ,check.item()))
            # log.info('1 epoch and {}th epoch mean compare : {:.6f}%'.format(_cnt_ ,check.item()))
            #=================================================================================

            scheduler.step(tr_epoch)
            ind_model.train()
            tqdm_train_loader = warp_tqdm(train_loader)
            for j, (data, label) in enumerate(tqdm_train_loader):
                label = label.cuda(non_blocking=True)

                # output = ind_model(data)
                # ind_model.module.set_mean(output_mean)
                # ind_model.module.train_mean = output_mean.cuda()
                output = ind_model(data, train_mean = output_mean)

                loss = criterion(output,label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prec1, _ = accuracy(output,label, topk =(1,5))

                tr_top1.update(prec1[0], data.size(0))
                if not args.disable_tqdm: # 
                    tqdm_train_loader.set_description('Episode: {} / Epoch: {} / Acc: {:.2f}'.format(i+1, tr_epoch+1, tr_top1.avg))

        # Get train Mean for test set
        ind_model.eval()
        with torch.no_grad():
            output_mean = []

            for k, (temp_input, temp_label) in enumerate(warp_tqdm(mean_loader)):
                temp_outputs, _ = ind_model(temp_input, feature = True)
                output_mean.append(temp_outputs)

            output_mean = torch.cat(output_mean, dim=0).mean(0)
            output_mean = torch.cat((output_mean,output_mean))


        # TEST
        ind_model.eval()
        with torch.no_grad():
            # target_test = target_test.cuda(non_blocking=True)
            output = ind_model(input_test, train_mean = output_mean)
            # target = target_test.cpu()

            output = output[:, args.num_classes:args.num_classes + args.imbal_way]
            label_test = torch.tensor(label_test, dtype = torch.int64)

            # change to use accuracy function
            _, pred = output.data.cpu().topk(1,dim=1)
            pred = (pred + args.num_classes).t()
            correct = pred.eq(label_test.view(1, -1))

            correct_tot = correct.view(-1).float().sum(0, keepdim=True)
            acc = correct_tot.mul_(100.0 / label_test.size(0))
            test_top1.update(acc[0], input_test.size(0))
            if not args.disable_tqdm: 
                tqdm_test_loader.set_description('Episode {}/{}, Test Acc {:.2f}'.format(i+1, args.imbal_iter, test_top1.avg))
            if i % 1 == 0: 
                log.info(' Episode {}/{}, Accuracy = {:.2f}'.format(i+1, args.imbal_iter, test_top1.avg))

            # For Independent Accuracy, NOT AVERAGE
            # if not args.disable_tqdm: 
            #     tqdm_test_loader.set_description('Episode {}/{}, Test Acc {:.2f}'.format(i+1, args.imbal_iter, acc[0]))
            # if i % 1 == 0: 
            #     log.info('[Independent Accuracy] Episode {}/{}, Accuracy = {:.2f}'.format(i+1, args.imbal_iter, acc[0]))

# def extract_feature_for_pointing_aug(model, train_loader, test_loader):

#     model.eval()
#     with torch.no_grad():
#         train_dict = collections.defaultdict(list)
#         train_mean = []
#         # For Random Samples
#         max_out = []
#         min_out = []
#         for i, (inputs, labels) in enumerate(warp_tqdm(train_loader)):
#             # compute output
#             outputs, _ = model(inputs, True)
#             outputs = outputs.cpu().data.numpy()  # if torch tensor is needed, change this part
#             train_mean.append(outputs)
  
#             for out, label in zip(outputs, labels):
#                 train_dict[label.item()].append(out)
#                 # For Random Samples
#                 max_out.append(max(out))
#                 min_out.append(min(out))

#         # For Random Samples
#         max_out = np.mean(np.array(max_out))
#         min_out = np.mean(np.array(min_out))
#         train_mean = np.concatenate(train_mean, axis=0).mean(0)


#         test_dict = collections.defaultdict(list)
#         fc_test_dict = collections.defaultdict(list)

#         for i, (inputs, labels) in enumerate(warp_tqdm(test_loader)):
#             # compute output
#             outputs, fc_outputs = model(inputs, True)
#             outputs = outputs.cpu().data.numpy()
#             # For Support set class selection
#             fc_outputs = fc_outputs.cpu().data.numpy()

#             for out, fc_out, label in zip(outputs, fc_outputs, labels):
#                 test_dict[label.item()].append(out)
#                 fc_test_dict[label.item()].append(fc_out)

#     return train_dict, train_mean, test_dict, max_out, min_out, fc_test_dict


def extract_feature_for_pointing_aug(model, train_loader):

    model.eval()
    with torch.no_grad():
        train_dict = collections.defaultdict(list)
        train_mean = []
        # For Random Samples
        max_out = []
        min_out = []
        for i, (inputs, labels) in enumerate(warp_tqdm(train_loader)):
            # compute output
            outputs, _ = model(inputs, True)
            outputs = outputs.cpu().data.numpy()  # if torch tensor is needed, change this part
            train_mean.append(outputs)
  
            for out, label in zip(outputs, labels):
                train_dict[label.item()].append(out)
                # For Random Samples
                max_out.append(max(out))
                min_out.append(min(out))

        # For Random Samples
        max_out = np.mean(np.array(max_out))
        min_out = np.mean(np.array(min_out))
        train_mean = np.concatenate(train_mean, axis=0).mean(0)

    return train_dict, train_mean


def extract_feature_cloning_iteration(model, test_loader, shot, clone_factor):

    model.eval()
    with torch.no_grad():
        support_data_dict = collections.defaultdict(list)
        support_label_dict = collections.defaultdict(list)
        query_data_dict = collections.defaultdict(list)
        query_label_dict = collections.defaultdict(list)
        fc_support_data_dict = collections.defaultdict(list)

        for i, (inputs, labels) in enumerate(warp_tqdm(test_loader)):
            # compute output
            outputs, fc_outputs = model(inputs, True)
            outputs = outputs.cpu().data.numpy()
            fc_outputs = fc_outputs.cpu().data.numpy()

            support_data_dict[i].extend(outputs[:args.meta_val_way * shot * clone_factor])
            support_label_dict[i].extend(labels[:args.meta_val_way * shot * clone_factor])
            query_data_dict[i].extend(outputs[args.meta_val_way * shot * clone_factor:])
            query_label_dict[i].extend(labels[args.meta_val_way * shot * clone_factor:])
            fc_support_data_dict[i].extend(fc_outputs[:args.meta_val_way * shot * clone_factor])


    iter_dict = {'support_data_dict' : support_data_dict, 'support_label_dict' : support_label_dict, 'query_data_dict' :  query_data_dict,
                 'query_label_dict': query_label_dict,'fc_support_data_dict': fc_support_data_dict} 

    return iter_dict


def get_samples(data_dict, samples_per_class, choose_class = None):

    classes = list(data_dict.keys())
    sampled_data = []
    for each_class in classes:
        samples = random.sample(data_dict[each_class], samples_per_class)
        sampled_data += samples
    sampled_data = np.array(sampled_data).astype(np.float32)

    return sampled_data

def random_samples(max_out, min_out, sample_num):

    sampled_data = []
    for i in range(sample_num):
        sampled_data.append(np.random.uniform(low = min_out, high = max_out, size = 512))
    sampled_data = np.array(sampled_data).astype(np.float32)

    return sampled_data

def pointing_aug_meta_test(train_dict, train_mean, train_samples, test_dict, shot, how_close_factor):
    un_list = []
    l2n_list = []
    cl2n_list = []

    for _ in warp_tqdm(range(args.meta_test_iter)):
        support_data, query_data, support_label, query_label = sample_case(test_dict, shot)

        augmented_data=[]
        extended_support_label = [] 
        for i, each_sample in enumerate(support_data):
            # each_sample = (512)
            # train_sample = (64,512)

            distance = train_samples - each_sample
            # distance = (64,512)
            # train - distance = (64,512) - (64,512)

            pointing_data = train_samples - (distance * how_close_factor)
            # pointing_data = (64,512)
            # pointing_data.mean(0) = (512)

            # If want to use mean, use below code
            augmented_data.append(pointing_data.mean(0))

            # else use the below code
            # augmented_data.extend(pointing_data)
            # extended_support_label += [support_label[i] for _ in range(len(pointing_data))]

        augmented_data = np.array(augmented_data).astype(np.float32)
        
        if(extended_support_label):
            support_label = extended_support_label

        # Centering + L2 normalization
        acc = metric_class_type(augmented_data, query_data, support_label, query_label, shot, train_mean=train_mean,
                                norm_type='CL2N')
        cl2n_list.append(acc)
        
        # L2 normalization
        acc = metric_class_type(augmented_data, query_data, support_label, query_label, shot, train_mean=train_mean,
                                norm_type='L2N')
        l2n_list.append(acc)

        # Unormalized
        acc = metric_class_type(augmented_data, query_data, support_label, query_label, shot, train_mean=train_mean,
                                norm_type='UN')
        un_list.append(acc)

    un_mean, un_conf = compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)

    return [un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf]


def pointing_aug_class_select(train_dict, train_mean, train_samples, test_dict, shot, how_close_factor, fc_test_dict = None):
    un_list = []
    l2n_list = []
    cl2n_list = []

    for _ in warp_tqdm(range(args.meta_test_iter)):
        support_data, query_data, support_label, query_label, fc_support_data = sample_case_fc(test_dict, fc_test_dict, shot)

        augmented_data=[]
        extended_support_label = [] 
        for each_sample, each_logit in zip(support_data,fc_support_data):
            # each_sample = (512)
            # train_sample = (64,512)
            select_class = np.argmax(each_logit)
            train_samples = train_dict[select_class]

            distance = train_samples - each_sample
            # distance = (64,512)
            # train - distance = (64,512) - (64,512)

            pointing_data = train_samples - (distance * how_close_factor)
            # pointing_data = (64,512)
            # pointing_data.mean(0) = (512)

            # If want to use mean, use below code
            augmented_data.append(pointing_data.mean(0))

            # else use the below code
            # augmented_data.extend(pointing_data)
            # extended_support_label += [support_label[i] for _ in range(len(pointing_data))]

        augmented_data = np.array(augmented_data).astype(np.float32)
        
        if(extended_support_label):
            support_label = extended_support_label

        # Centering + L2 normalization
        acc = metric_class_type(augmented_data, query_data, support_label, query_label, shot, train_mean=train_mean,
                                norm_type='CL2N')
        cl2n_list.append(acc)
        
        # L2 normalization
        acc = metric_class_type(augmented_data, query_data, support_label, query_label, shot, train_mean=train_mean,
                                norm_type='L2N')
        l2n_list.append(acc)

        # Unormalized
        acc = metric_class_type(augmented_data, query_data, support_label, query_label, shot, train_mean=train_mean,
                                norm_type='UN')
        un_list.append(acc)

    un_mean, un_conf = compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)

    return [un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf]




def pointing_aug_cloning(train_dict, train_mean, train_samples, test_dict, shot, clone_factor, how_close_factor, fc_test_dict = None):
    un_list = []
    l2n_list = []
    cl2n_list = []


    for i in warp_tqdm(range(args.meta_test_iter)):
        support_data = test_dict['support_data_dict'][i]
        query_data = test_dict['query_data_dict'][i]
        support_label = test_dict['support_label_dict'][i]
        query_label = test_dict['query_label_dict'][i]
        fc_support_data = test_dict['fc_support_data_dict'][i]

        augmented_data=[]
        extended_support_label = [] 
        for each_sample, each_logit in zip(support_data,fc_support_data):
            # each_sample = (512)
            # train_sample = (64,512)
            select_class = np.argmax(each_logit)
            train_samples = train_dict[select_class]

            distance = train_samples - each_sample
            # distance = (64,512)
            # train - distance = (64,512) - (64,512)

            pointing_data = train_samples - (distance * how_close_factor)
            # pointing_data = (64,512)
            # pointing_data.mean(0) = (512)

            # If want to use mean, use below code
            augmented_data.append(pointing_data.mean(0))

            # else use the below code
            # augmented_data.extend(pointing_data)
            # extended_support_label += [support_label[i] for _ in range(len(pointing_data))]

        augmented_data = np.array(augmented_data).astype(np.float32)
        
        if(extended_support_label):
            support_label = extended_support_label

        # Centering + L2 normalization
        acc = metric_class_type(augmented_data, query_data, support_label, query_label, shot, train_mean=train_mean,
                                norm_type='CL2N', clone_factor = clone_factor)
        cl2n_list.append(acc)
        
        # L2 normalization
        acc = metric_class_type(augmented_data, query_data, support_label, query_label, shot, train_mean=train_mean,
                                norm_type='L2N', clone_factor = clone_factor)
        l2n_list.append(acc)

        # Unormalized
        acc = metric_class_type(augmented_data, query_data, support_label, query_label, shot, train_mean=train_mean,
                                norm_type='UN', clone_factor = clone_factor)
        un_list.append(acc)

    un_mean, un_conf = compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)

    return [un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf]



def sample_case_fc(ld_dict, fc_dict, shot):
    sample_class = random.sample(list(ld_dict.keys()), args.meta_val_way)
    train_input = []
    test_input = []
    test_label = []
    train_label = []
    fc_train_input = []
    for each_class in sample_class:
        samples_idx = random.sample(range(len(ld_dict[each_class])), shot + args.meta_val_query)
        samples = [ld_dict[each_class][i] for i in samples_idx]
        samples_fc = [fc_dict[each_class][i] for i in samples_idx]

        train_label += [each_class] * len(samples[:shot])
        test_label += [each_class] * len(samples[shot:])
        train_input += samples[:shot]
        test_input += samples[shot:]
        fc_train_input += samples_fc[:shot]

    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    fc_train_input = np.array(fc_train_input).astype(np.float32)
    return train_input, test_input, train_label, test_label, fc_train_input


def pointing_augmentation_eval(model, log):

    load_checkpoint(model, 'last')
    train_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False)

    # test_loader = get_dataloader('test', aug=False, shuffle=False, out_name=False)

    # train_dict, train_mean, test_dict, max_out, min_out, fc_test_dict = extract_feature_for_pointing_aug(model, train_loader, test_loader)

# =======================================================================================
    # For Meta Cloning
    train_dict, train_mean = extract_feature_for_pointing_aug(model, train_loader)

    # if(args.meta_cloning):
    #     Validation loader
    #                     test iteration,       test way,      test shot,       test query,    test clone factor
    sample_info = [args.meta_test_iter, args.meta_val_way,    1,       args.meta_val_query,      20]
    test_loader = get_dataloader('test', aug=None, shuffle=False, out_name=False, sample=sample_info)
    test_dict_1shot = extract_feature_cloning_iteration(model, test_loader, 1, 20)

                    # test iteration,       test way,      test shot,       test query,    test clone factor
    sample_info = [args.meta_test_iter, args.meta_val_way,    5,       args.meta_val_query,      4]
    test_loader = get_dataloader('test', aug=None, shuffle=False, out_name=False, sample=sample_info)
    test_dict_5shot = extract_feature_cloning_iteration(model, test_loader, 5, 4)

    train_samples = []
# ===================================================================================================================
    # train_samples = get_samples(train_dict, samples_per_class = args.pointing_aug_samples_per_class)
    # train_samples = random_samples(max_out, min_out, sample_num = 1)

    factor_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # factor_list = [1.0]
    for how_close_factor in factor_list:
        # accuracy_info_shot1 = pointing_aug_meta_test(train_dict, train_mean, train_samples, test_dict, 1, how_close_factor)
        # accuracy_info_shot5 = pointing_aug_meta_test(train_dict, train_mean, train_samples, test_dict, 5, how_close_factor)
        # accuracy_info_shot1 = pointing_aug_class_select(train_dict, train_mean, train_samples, test_dict, 1, how_close_factor, fc_test_dict)
        # accuracy_info_shot5 = pointing_aug_class_select(train_dict, train_mean, train_samples, test_dict, 5, how_close_factor, fc_test_dict)
        accuracy_info_shot1 = pointing_aug_cloning(train_dict, train_mean, train_samples, test_dict_1shot, 1, 20, how_close_factor)
        accuracy_info_shot5 = pointing_aug_cloning(train_dict, train_mean, train_samples, test_dict_5shot, 5,  4, how_close_factor)

        print(
            'Meta Test: LAST\nHow_Close_Factor={}\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'
            .format(how_close_factor, 'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
        log.info(
            'Meta Test: LAST\nHow_Close_Factor={}\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'
            .format(how_close_factor, 'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))

    return



if __name__ == '__main__':
    main()