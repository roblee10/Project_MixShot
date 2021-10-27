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

import DataHandler
import models
import utils


best_prec1 = -1
global first


def main():
    global args, best_prec1
    args = configuration.parser_args()

    ### initial logger
    log = utils.setup_logger(args.save_path + '/'+ args.logname)
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


    # create model
    log.info("=> creating model '{}'".format(args.arch))

    model_CE = models.__dict__[args.arch](num_classes=args.num_classes, remove_linear=args.do_meta_train) # eliminate fully connected layer if meta-train
    model_CE = torch.nn.DataParallel(model_CE).cuda()
    optimizer_CE = get_optimizer(model_CE)

    criterion_CE = nn.CrossEntropyLoss().cuda()

    # Evaluation
    if args.evaluate:
        log.info('---Cross Entropy Loss Evaluation---')
        do_extract_and_evaluate(model_CE, log)
        return

    if args.pointing_aug_eval:
        pointing_augmentation_eval(model_CE, log)
        return

    if args.do_meta_train:
        sample_info = [args.meta_train_iter, args.meta_train_way, args.meta_train_shot, args.meta_train_query]
        train_loader = get_dataloader('train', not args.disable_train_augment, sample=sample_info)
    else:
        train_loader = get_dataloader('train', not args.disable_train_augment, shuffle=True, out_name=False)
    
    scheduler_CE = get_scheduler(len(train_loader), optimizer_CE)
        
    # Validation loader
    sample_info = [args.meta_val_iter, args.meta_val_way, args.meta_val_shot, args.meta_val_query]
    val_loader = get_dataloader('val', False, sample=sample_info)

    tqdm_loop = warp_tqdm(list(range(args.start_epoch, args.epochs)))

    for epoch in tqdm_loop:

        scheduler_CE.step(epoch)
        log.info('---Cross Entropy Loss Train---')
        train(train_loader, model_CE, criterion_CE, optimizer_CE, epoch, scheduler_CE, log)

        is_best = False

        if (epoch + 1) % args.meta_val_interval == 0:
            prec1_CE = meta_val(val_loader, model_CE)
            log.info('Epoch: [{}]\t''Cross Entropy Meta Val : {}'.format(epoch, prec1_CE))

        #remember best prec@1 and save checkpoint
        utils.save_checkpoint({
            'epoch': epoch + 1,
            # 'scheduler': scheduler.state_dict()
            'arch': args.arch,
            'state_dict': model_CE.state_dict(),
            # 'best_prec1': best_prec1,
            'optimizer': optimizer_CE.state_dict(),
        }, is_best, filename='CE_checkpoint.pth.tar'  ,folder=args.save_path)

    # do evaluate at the end
    log.info('---Cross Entropy Loss Evaluation---')
    do_extract_and_evaluate(model_CE, log)

    return


# validation(few-shot)
def meta_val(test_loader, model, train_mean=None):
    top1 = utils.AverageMeter()
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

            prediction = utils.metric_prediction(train_out, test_out, train_label, args.meta_val_metric)
            acc = (prediction == test_label).float().mean()
            top1.update(acc.item())
            if not args.disable_tqdm:
                tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg * 100))
    return top1.avg

# Training Function
def train(train_loader, model, criterion, optimizer, epoch, scheduler, log):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

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
            output = -utils.get_metric(args.meta_train_metric)(shot_proto, query_proto) # shape = (75,5)

        
        loss = criterion(output, target)# When meta training 
                                        # output = (75,5), target = (75), since the output is distance, not probability distribution, use minus in output 
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1, _ = utils.accuracy(output, target, topk=(1,5))

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

def get_scheduler(batches, optimiter):

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

def get_dataloader(split, aug=False, shuffle=True, out_name=False, sample=None, multisample = None):
    # sample: iter, way, shot, query

    if aug:
        transform = DataHandler.with_augment(84, disable_random_resize=args.disable_random_resize)
    elif aug is None:
        transform = None
    else:
        transform = DataHandler.without_augment(84, enlarge=args.enlarge) 

    sets = DataHandler.DatasetFolder(args.data, args.split_dir, split, transform, out_name=out_name)
    
    # For Meta-training
    # defaulte 400 iteration, each 100(75query 15support)
    if sample is not None:
        sampler = DataHandler.CategoriesSampler(sets.labels, *sample)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.workers, pin_memory=True) # shuffle needs to be false
    elif multisample is not None:
        sampler = DataHandler.MultishotSampler(sets.labels, *multisample)
        if transform is None:
            loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.workers,collate_fn = my_collate, pin_memory=True) # shuffle needs to be false
        else:
            loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                                num_workers=args.workers, pin_memory=True) # shuffle needs to be false
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.workers, pin_memory=True)
    return loader


def my_collate(batch):
    sup_pipeline = DataHandler.aug_pipeline(84, clone_augment_method=args.clone_augment_method)
    query_pipeline = DataHandler.without_augment(84, enlarge=args.enlarge )
    
    shot1 = 1
    shot5 = 5
    shot1_idxrange = args.meta_val_way * shot1 * args.clone_factor_1shot
    support_range= shot1_idxrange + args.meta_val_way * shot5 * args.clone_factor_5shot

    Sup = batch[:support_range]
    Query = batch[support_range:]
    Support = [sup_pipeline(item[0]) for item in Sup]
    Query = [query_pipeline(item[0]) for item in Query]

    Support = torch.stack(Support)
    Query = torch.stack(Query)
    data = torch.cat((Support,Query))

    target = [item[1] for item in batch]
    target = torch.LongTensor(target)

    idx = [item[2] for item in batch]
    return [data, target, idx]


def warp_tqdm(data_loader):
    if args.disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm.tqdm(data_loader, total=len(data_loader))
    return tqdm_loader

def meta_evaluate_case_study(data, train_mean, out_idx, shot):
    un_list = []
    l2n_list = []
    cl2n_list = []

    episode_train = collections.defaultdict(list) 
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

    un_mean, un_conf = utils.compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = utils.compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = utils.compute_confidence_interval(cl2n_list)
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
    un_mean, un_conf = utils.compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = utils.compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = utils.compute_confidence_interval(cl2n_list)
    return [un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf]


def metric_class_type(gallery, query, train_label, test_label, shot, train_mean=None, norm_type='CL2N', clone_factor=1, case_study=False):
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
    if case_study:
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

    utils.load_checkpoint(model, args.save_path, 'last')
    train_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False)

    if(args.case_study):
        val_loader = get_dataloader('test', aug=False, shuffle=False, out_name=True)
        out_mean, fc_out_mean, out_dict, fc_out_dict, out_idx = extract_feature(train_loader, val_loader, model, 'last')
        accuracy_info_shot1, episode_shot1 = meta_evaluate_case_study(out_dict, out_mean, out_idx, 1)
        accuracy_info_shot5, episode_shot5 = meta_evaluate_case_study(out_dict, out_mean, out_idx, 5)

        utils.save_pickle('episode_shot1.pkl', episode_shot1)
        utils.save_pickle('episode_shot5.pkl', episode_shot5)

    else:
        val_loader = get_dataloader('test', aug=False, shuffle=False, out_name=False)
        out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'last')
        accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, 1)
        accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, 5)
        
    print(
        'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'
        .format('GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    log.info(
        'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'
        .format('GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))

def extract_train_feature(model, train_loader):

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

    return train_dict, train_mean, max_out, min_out


def extract_episode_feature(model, test_loader, shot1, shot5, shot1_clone_factor=1, shot5_clone_factor=1):

    model.eval()
    with torch.no_grad():
        shot1_data_dict = collections.defaultdict(list)
        shot1_label_dict = collections.defaultdict(list)
        shot1_idx_dict = collections.defaultdict(list)

        shot5_data_dict = collections.defaultdict(list)
        shot5_label_dict = collections.defaultdict(list)
        shot5_idx_dict = collections.defaultdict(list)

        query_data_dict = collections.defaultdict(list)
        query_label_dict = collections.defaultdict(list)
        query_idx_dict = collections.defaultdict(list)

        fc_shot1_data_dict = collections.defaultdict(list)
        fc_shot5_data_dict = collections.defaultdict(list)

        shot1_idxrange = args.meta_val_way * shot1 * shot1_clone_factor
        shot5_idxrange = shot1_idxrange +  args.meta_val_way * shot5 * shot5_clone_factor

        for i, (inputs, labels, idx) in enumerate(warp_tqdm(test_loader)):
            # compute output
            outputs, fc_outputs = model(inputs, True)
            outputs = outputs.cpu().data.numpy()
            fc_outputs = fc_outputs.cpu().data.numpy()

            shot1_data_dict[i].extend(outputs[:shot1_idxrange])
            shot1_label_dict[i].extend(labels[:shot1_idxrange])
            shot1_idx_dict[i].extend(idx[:shot1_idxrange])
            
            shot5_data_dict[i].extend(outputs[shot1_idxrange:shot5_idxrange])            
            shot5_label_dict[i].extend(labels[shot1_idxrange:shot5_idxrange]) 
            shot5_idx_dict[i].extend(idx[shot1_idxrange:shot5_idxrange])

            query_data_dict[i].extend(outputs[shot5_idxrange:])
            query_label_dict[i].extend(labels[shot5_idxrange:])
            query_idx_dict[i].extend(idx[shot5_idxrange:])

            fc_shot1_data_dict[i].extend(fc_outputs[:shot1_idxrange])
            fc_shot5_data_dict[i].extend(fc_outputs[shot1_idxrange:shot5_idxrange])           

    shot1_dict = {'support_data_dict' : shot1_data_dict, 'support_label_dict' : shot1_label_dict, 'support_idx_dict' : shot1_idx_dict,
                  'query_data_dict' : query_data_dict, 'query_label_dict' : query_label_dict, 'query_idx_dict' : query_idx_dict,
                  'fc_support_data_dict' : fc_shot1_data_dict}

    shot5_dict = {'support_data_dict' : shot5_data_dict, 'support_label_dict' : shot5_label_dict, 'support_idx_dict' : shot5_idx_dict,
                  'query_data_dict' : query_data_dict, 'query_label_dict' : query_label_dict, 'query_idx_dict' : query_idx_dict,
                  'fc_support_data_dict' : fc_shot5_data_dict}

    return shot1_dict, shot5_dict


def get_samples(data_dict, samples_per_class):

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

def select_samples(gallery, query, train_mean=None, norm_type='CL2N'):
    if norm_type == 'CL2N': # subtract train mean on both support and query set and normalize
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query - train_mean
        query = query / LA.norm(query, 2, 1)[:, None]
    elif norm_type == 'L2N': # just normalize
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]

    # assume 5 shot 5 way
    subtract = gallery[:, None, :] - query
    distance = LA.norm(subtract, 2, axis=-1) # get euclidean distance between support and query (L2 norm)  , shape = (25,75)

    if args.select_class_type == 'close':
        idx = np.argpartition(distance, args.pointing_aug_selectclass_num, axis=0)[:args.pointing_aug_selectclass_num]
    elif args.select_class_type == 'far':
        idx = np.argpartition(distance, -args.pointing_aug_selectclass_num, axis=0)[len(distance)-args.pointing_aug_selectclass_num:]

    idx = idx.reshape(-1)
    selected_samples = gallery[idx]
    return selected_samples

def meta_pointing_augmentation(train_dict, train_mean, train_samples, test_dict, shot, clone_factor, how_close_factor):
    un_list = []
    l2n_list = []
    cl2n_list = []

    episode_true = {'cl2n' : collections.defaultdict(list), 
                    'l2n' : collections.defaultdict(list), 
                    'un' : collections.defaultdict(list)}   # cases that are correct in query set 
    episode_false = {'cl2n' : collections.defaultdict(list), 
                     'l2n' : collections.defaultdict(list), 
                     'un' : collections.defaultdict(list)}    # cases that are wrong in query set
                    
    episode_train = test_dict['support_idx_dict'] 
    episode_test =  test_dict['query_idx_dict']

    for i in warp_tqdm(range(args.meta_test_iter)):

        support_data = test_dict['support_data_dict'][i]
        query_data = test_dict['query_data_dict'][i]
        support_label = test_dict['support_label_dict'][i]
        query_label = test_dict['query_label_dict'][i]
        fc_support_data = test_dict['fc_support_data_dict'][i]

        support_idx = episode_train[i]
        query_idx = episode_test[i]

        if how_close_factor != 1 and args.pointing_aug_sample_type != 'none':
            augmented_data=[]
            extended_support_label = [] 
            for each_sample, each_logit in zip(support_data,fc_support_data):
                # each_sample = (512)
                # train_sample = (64,512)

                if args.pointing_aug_sample_type == 'select_class':
                    select_class = np.argmax(each_logit)

                    if args.select_class_type == 'random':
                        # For Random Selection
                        train_samples = random.sample(train_dict[select_class], args.pointing_aug_selectclass_num)
                    else:
                        select_gallery = np.array(train_dict[select_class])
                        select_query = each_sample[None,:] 
                        train_samples = select_samples(select_gallery, select_query, train_mean=train_mean , norm_type=args.select_class_metric)

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

            support_data = augmented_data
        
            if(extended_support_label):
                support_label = extended_support_label

        support_data = np.array(support_data).astype(np.float32)
        query_data = np.array(query_data).astype(np.float32)
        fc_support_data = np.array(fc_support_data).astype(np.float32)
        query_idx = np.array(query_idx)

        # Centering + L2 normalization
        acc, episode_result_cl2n = metric_class_type(support_data, query_data, support_label, query_label, shot, train_mean=train_mean,
                                norm_type='CL2N', clone_factor = clone_factor, case_study=True)
        cl2n_list.append(acc)
        
        # L2 normalization
        acc, episode_result_l2n = metric_class_type(support_data, query_data, support_label, query_label, shot, train_mean=train_mean,
                                norm_type='L2N', clone_factor = clone_factor, case_study=True)
        l2n_list.append(acc)
        # Unormalized
        acc, episode_result_un = metric_class_type(support_data, query_data, support_label, query_label, shot, train_mean=train_mean,
                                norm_type='UN', clone_factor = clone_factor, case_study=True)
        un_list.append(acc)

    un_mean, un_conf = utils.compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = utils.compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = utils.compute_confidence_interval(cl2n_list)

    if args.case_study:
        episode_true['cl2n'][i] = list(query_idx[(episode_result_cl2n)])
        episode_false['cl2n'][i] = list(query_idx[~episode_result_cl2n])
        episode_true['l2n'][i] = list(query_idx[(episode_result_l2n)])
        episode_false['l2n'][i] = list(query_idx[~episode_result_l2n])
        episode_true['un'][i] = list(query_idx[(episode_result_un)])
        episode_false['un'][i] = list(query_idx[~episode_result_un])

    return [un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf], [episode_train, episode_test, episode_true, episode_false]


def pointing_augmentation_eval(model, log):

    utils.load_checkpoint(model, args.save_path, 'last')
    train_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False)
    train_dict, train_mean, max_out, min_out = extract_train_feature(model, train_loader)

    sample_info = [args.meta_test_iter, args.meta_val_way, 1, 5, args.meta_val_query, args.clone_factor_1shot, args.clone_factor_5shot] 
    if args.meta_cloning:
        pkl_dict1 = args.data_type + '_metadict_1shot_Cloning_' + args.clone_augment_method + '.pkl'
        pkl_dict2 = args.data_type + '_metadict_5shot_Cloning_' + args.clone_augment_method + '.pkl'
        test_loader = get_dataloader('test', aug=None, out_name = True, multisample=sample_info)
    else:
        pkl_dict1 = args.data_type + '_metadict_1shot_noCloning.pkl'
        pkl_dict2 = args.data_type + '_metadict_5shot_noCloning.pkl'
        test_loader = get_dataloader('test', aug=False, out_name = True, multisample=sample_info)

    if os.path.isfile(pkl_dict1) and os.path.isfile(pkl_dict2):
        metadict_1shot = utils.load_pickle(pkl_dict1)
        metadict_5shot = utils.load_pickle(pkl_dict2)
    else:
        metadict_1shot, metadict_5shot = extract_episode_feature(model, test_loader, 1, 5, args.clone_factor_1shot, args.clone_factor_5shot)
        utils.save_pickle(pkl_dict1, metadict_1shot)
        utils.save_pickle(pkl_dict2, metadict_5shot)
    
 
    factor_list = [0.9,0.7,0.5,0.3,0.1,0]

    for how_close_factor in factor_list:
        assert args.pointing_aug_sample_type in ['equal_per_class', 'random_generation', 'select_class', 'none']    

        if args.pointing_aug_sample_type == 'equal_per_class':
            train_samples = get_samples(train_dict, args.pointing_aug_equalperclass_num)
        elif args.pointing_aug_sample_type == 'random_generation':
            train_samples = random_samples(max_out, min_out, sample_num = args.pointing_aug_randomgeneration_num)
        else:
            train_samples = []

        accuracy_info_shot1, episode_shot1 = meta_pointing_augmentation(train_dict, train_mean, train_samples, metadict_1shot, 1, args.clone_factor_1shot, how_close_factor)
        accuracy_info_shot5, episode_shot5 = meta_pointing_augmentation(train_dict, train_mean, train_samples, metadict_5shot, 5, args.clone_factor_5shot, how_close_factor)

        if args.case_study:
            utils.save_pickle('episode_shot1.pkl', episode_shot1)
            utils.save_pickle('episode_shot5.pkl', episode_shot5)

        print(
            'Meta Test: LAST\nHow_Close_Factor={}\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'
            .format(how_close_factor, 'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
        log.info(
            'Meta Test: LAST\nHow_Close_Factor={}\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'
            .format(how_close_factor, 'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))

        if args.pointing_aug_sample_type == 'none':
            break
    
    return

if __name__ == '__main__':
    main()