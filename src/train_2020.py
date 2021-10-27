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
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
import tqdm
import configuration
from numpy import linalg as LA
from scipy.stats import mode
from tensorboardX import SummaryWriter

import DataHandler
import models


best_prec1 = -1


def main():
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
    model = models.__dict__[args.arch](num_classes=args.num_classes, remove_linear=args.do_meta_train) # eliminate fully connected layer if meta-train

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = get_optimizer(model)

    if args.pretrain:
        pretrain = args.pretrain + '/checkpoint.pth.tar'
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

    # Evaluation
    if args.evaluate or args.no_meta_eval:
        do_extract_and_evaluate(model, log)
        return
    
    # for meta-training, use sample_info when generating dataloader
    if args.do_meta_train:
        sample_info = [args.meta_train_iter, args.meta_train_way, args.meta_train_shot, args.meta_train_query]
        train_loader = get_dataloader('train', not args.disable_train_augment, sample=sample_info)
    # Training loader
    else:
        train_loader = get_dataloader('train', not args.disable_train_augment, shuffle=True)
    # Validation loader
    sample_info = [args.meta_val_iter, args.meta_val_way, args.meta_val_shot, args.meta_val_query]
    val_loader = get_dataloader('val', False, sample=sample_info)
    # analyzation loader
    analyze_info = [args.anlz_iter, args.anlz_class, args.anlz_base, args.anlz_same, args.anlz_diff, args.anlz_rand]
    anlz_tr_loader = get_dataloader('train', aug=False, analyze = analyze_info)
    anlz_val_loader = get_dataloader('val', aug=False, analyze = analyze_info)

    scheduler = get_scheduler(len(train_loader), optimizer)
    tqdm_loop = warp_tqdm(list(range(args.start_epoch, args.epochs)))
    for epoch in tqdm_loop:
        scheduler.step(epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, scheduler, log)
        # evaluate on meta validation set
        # meta_val_interval : set when to validate during training
        is_best = False

        if args.analyze and (epoch +1) % args.anlz_interval == 0:
            tr_same, tr_diff = distance_compare(model,anlz_tr_loader)
            log.info('Tr  Dist Compare: Same {:.4f} Diff {:.4f}'.format(tr_same.avg, tr_diff.avg))
            val_same, val_diff = distance_compare(model,anlz_val_loader)
            log.info('Val Dist Compare: Same {:.4f} Diff {:.4f}'.format(val_same.avg, val_diff.avg))
            
            writer.add_scalars('No_FC/tr_distance', {'same':tr_same.avg, 'diff': tr_diff.avg} ,epoch+1)
            writer.add_scalars('No_FC/val_distance', {'same':val_same.avg, 'diff': val_diff.avg} ,epoch+1)
            writer.add_scalars('No_FC/distance_interval', 
                                {'Train_Data': tr_diff.avg-tr_same.avg, 'Val_Data': val_diff.avg-val_same.avg} ,epoch+1)
            if args.anlz_FC:
                FC_tr_same, FC_tr_diff = distance_compare(model,anlz_tr_loader,feature = False)
                FC_val_same, FC_val_diff = distance_compare(model,anlz_val_loader,feature = False)
                writer.add_scalars('With_FC/tr_distance', {'same':FC_tr_same.avg, 'diff': FC_tr_diff.avg} ,epoch+1)
                writer.add_scalars('With_FC/val_distance', {'same':FC_val_same.avg, 'diff': FC_val_diff.avg} ,epoch+1)
                writer.add_scalars('With_FC/distance_interval',
                                    {'Train_Data': FC_tr_diff.avg - FC_tr_same.avg, 'Val_Data': FC_val_diff.avg - FC_val_same.avg} ,epoch+1)


        if (epoch + 1) % args.meta_val_interval == 0:
            prec1 = meta_val(val_loader, model)
            log.info('Meta Val {}: {}'.format(epoch, prec1))
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if not args.disable_tqdm:
                tqdm_loop.set_description('Best Acc {:.2f}'.format(best_prec1 * 100.))

        # remember best prec@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            # 'scheduler': scheduler.state_dict()
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, folder=args.save_path)
    
    writer.close()
    # do evaluate at the end
    do_extract_and_evaluate(model, log)


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


def metric_prediction(gallery, query, train_label, metric_type):
    gallery = gallery.view(gallery.shape[0], -1) # torch.view -> reshape
    query = query.view(query.shape[0], -1)
    distance = get_metric(metric_type)(gallery, query)
    predict = torch.argmin(distance, dim=1)
    predict = torch.take(train_label, predict) # input tensor(train_label) is treated as if it were viewd as a 1-D tesnor. index is mentioned in second parameter(predict)

    return predict

# validation(few-shot)
def meta_val(test_loader, model, train_mean=None):
    top1 = AverageMeter()
    model.eval()

    with torch.no_grad():
        tqdm_test_loader = warp_tqdm(test_loader)
        for i, (inputs, target) in enumerate(tqdm_test_loader):
            target = target.cuda(0, non_blocking=True)
            output = model(inputs, True)[0].cuda(0) # why [0] -> if feature = True, two parameters are returned, x(not pass fc), x1(pass fc)
            if train_mean is not None:
                output = output - train_mean
            train_out = output[:args.meta_val_way * args.meta_val_shot]
            train_label = target[:args.meta_val_way * args.meta_val_shot]
            test_out = output[args.meta_val_way * args.meta_val_shot:]
            test_label = target[args.meta_val_way * args.meta_val_shot:]
            train_out = train_out.reshape(args.meta_val_way, args.meta_val_shot, -1).mean(1)
            train_label = train_label[::args.meta_val_shot]
            prediction = metric_prediction(train_out, test_out, train_label, args.meta_val_metric)
            acc = (prediction == test_label).float().mean()
            top1.update(acc.item())
            if not args.disable_tqdm:
                tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg * 100))
    return top1.avg

# Training Function
def train(train_loader, model, criterion, optimizer, epoch, scheduler, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    tqdm_train_loader = warp_tqdm(train_loader)
    for i, (input, target) in enumerate(tqdm_train_loader):
        
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
                                        # output = (75,5), target = (75), since the output is distance, not probability distribution, use minus in output to maximize the cross entropy of proper class

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
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger()
    # handler = logging.StreamHandler()
    # handler.setFormatter(file_formatter)
    # logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) is not '':
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

        # Validation(Test) Data
        # Save each feature in dictionary. use fc layer output if necessary
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

def get_dataloader(split, aug=False, shuffle=True, out_name=False, sample=None, analyze=None):
    # sample: iter, way, shot, query

    if aug:
        transform = DataHandler.with_augment(84, disable_random_resize=args.disable_random_resize)
    else:
        transform = DataHandler.without_augment(84, enlarge=args.enlarge)
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
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.workers, pin_memory=True)
    return loader

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
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(args.save_path))
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    model.load_state_dict(checkpoint['state_dict'])

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
    return un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf

def metric_class_type(gallery, query, train_label, test_label, shot, train_mean=None, norm_type='CL2N'):
    if norm_type == 'CL2N': # subtract train mean on both support and query set and normalize
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query - train_mean
        query = query / LA.norm(query, 2, 1)[:, None]
    elif norm_type == 'L2N': # just normalize
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]
    gallery = gallery.reshape(args.meta_val_way, shot, gallery.shape[-1]).mean(1)
    train_label = train_label[::shot]
    subtract = gallery[:, None, :] - query 
    distance = LA.norm(subtract, 2, axis=-1) # get euclidean distance between support and query (L2 norm)
    idx = np.argpartition(distance, args.num_NN, axis=0)[:args.num_NN] # get the index of the smallest distance in the list
    nearest_samples = np.take(train_label, idx)  # input array is treated as if it were viewd as a 1-D tesnor. index is mentioned in second parameter
    out = mode(nearest_samples, axis=0)[0]
    out = out.astype(int)
    test_label = np.array(test_label)
    acc = (out == test_label).mean()
    return acc

# Sample data to meta-test format
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

    train_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False)
    val_loader = get_dataloader('test', aug=False, shuffle=False, out_name=False)
    load_checkpoint(model, 'last')

    # Extract the average of the output. When using FC Layer, it extracts the average of FC Layer output
    out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'last')
    accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, 1)
    accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, 5)
    print(
        'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    log.info(
        'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    if args.eval_fc: # Use logit for Nearest Neighbor
        accuracy_info = meta_evaluate(fc_out_dict, fc_out_mean, 1)
        print('{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format('Logits', *accuracy_info))
        log.info('{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format('Logits', *accuracy_info))

    load_checkpoint(model, 'best')
    out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'best')
    accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, 1)
    accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, 5)
    print(
        'Meta Test: BEST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    log.info(
        'Meta Test: BEST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    if args.eval_fc:
        accuracy_info = meta_evaluate(fc_out_dict, fc_out_mean, 1)
        print('{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format('Logits', *accuracy_info))
        log.info('{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format('Logits', *accuracy_info))

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
            base_out = output[:args.anlz_base]
            same_out = output[args.anlz_base : args.anlz_base + args.anlz_same]
            diff_out = output[args.anlz_base + args.anlz_same : args.anlz_base + args.anlz_same + args.anlz_diff]
            
            same_distance = get_metric(args.anlz_metric)(same_out,base_out)
            diff_distance = get_metric(args.anlz_metric)(diff_out,base_out)
            same_distance, diff_distance = same_distance.reshape(-1).mean(), diff_distance.reshape(-1).mean()
        
            same.update(same_distance.detach().cpu().numpy())
            diff.update(diff_distance.detach().cpu().numpy())
    return same,diff


if __name__ == '__main__':
    main()
