import os
import PIL.Image as Image
import numpy as np
import pickle
import torch
import csv
import shutil
import collections

from torch.utils.data import Sampler
import torchvision.transforms as transforms

class DatasetFolder(object):

    def __init__(self, root, split_dir, split_type, transform, out_name=False):
        assert split_type in ['train', 'test', 'val', 'query', 'repr']

        # if split_type == 'val':
        #     split_type = 'test'

        split_file = os.path.join(split_dir, split_type + '.csv')
        assert os.path.isfile(split_file)
        with open(split_file, 'r') as f:
            split = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '']    

        data, ori_labels = [x[0] for x in split], [x[1] for x in split]


        # 엑셀에 저장된 이미지 index 값을 정수로 mapping 하는 과정
        label_key = sorted(np.unique(np.array(ori_labels)))
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]  

        # 여기서 val 에 있는 클래스 5개만 가져와서 1 shot 이면 클래스별 각 1개, 5 shot 이면 클래스별 각 5개 넣기
        self.root = root
        self.transform = transform
        self.data = data
        self.labels = mapped_labels  # label을 순서대로 0 부터 오름차순으로 매긴것, EX) 이미지 0~599번 까지는 0, 600~1199 까지는 1, so on...
        self.out_name = out_name
        self.length = len(self.data)

    def __len__(self):
        return self.length 

    def __getitem__(self, index):
        assert os.path.isfile(self.root+'/'+self.data[index])
        img = Image.open(self.root + '/' + self.data[index]).convert('RGB')
        label = self.labels[index]
        label = int(label)
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            # print(img.shape)
            return img, label, self.data[index]
        else:
            # print(img.shape)
            return img, label

class DatasetResampler(object):
    
    def __init__(self, root, split_dir, split_type, sup_data, sup_label, k_shot, n_way, transform, out_name=False):
        assert split_type in ['train', 'test', 'val', 'query', 'repr']

        split_file = os.path.join(split_dir, split_type + '.csv')
        assert os.path.isfile(split_file)
        with open(split_file, 'r') as f:
            split = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '']

        data, ori_labels = [x[0] for x in split], [x[1] for x in split]
        

        # 엑셀에 저장된 이미지 index 값을 정수로 mapping 하는 과정
        label_key = sorted(np.unique(np.array(ori_labels)))
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]  

        each_cnt = ori_labels.count(ori_labels[0])   # 600, each label image count
        resampler = int(each_cnt / k_shot) # how much each support image should be multiplied, EX) 600/5

        resampled_data = sup_data * resampler # EX) sup_set * 120
        resampled_label = sup_label * resampler

        data.extend(resampled_data)
        mapped_labels.extend(resampled_label)

        self.root = root
        self.transform = transform
        self.data = data
        self.labels = mapped_labels  # label을 순서대로 0 부터 오름차순으로 매긴것, EX) 이미지 0~599번 까지는 0, 600~1199 까지는 1, so on...
        self.out_name = out_name
        self.length = len(self.data)
        # print('Dataset Resampler test')

    def __len__(self):
        return self.length 

    def __getitem__(self, index):
        assert os.path.isfile(self.root+'/'+self.data[index])
        img = Image.open(self.root + '/' + self.data[index]).convert('RGB')
        label = self.labels[index]
        label = int(label)
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            # print(img.shape)
            return img, label, self.data[index]
        else:
            # print(img.shape)
            return img, label


class ImbalancedDataset(object):

    def __init__(self, root, split_dir, transform, k_shot, n_way, out_name=False):

        # Get Validation set       
        split_file_val = os.path.join(split_dir, 'val' + '.csv')
        assert os.path.isfile(split_file_val)
        with open(split_file_val, 'r') as f:
            split_val = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '']
        data_val, ori_labels_val = [x[0] for x in split_val], [x[1] for x in split_val]
        
        # Valdiation set dictionary
        val_dict = collections.defaultdict(list)
        for data, label in zip(data_val, ori_labels_val):
            val_dict[label].append(data)

        # Select Validation data
        label_key_val = sorted(np.unique(np.array(ori_labels_val)))
        # second parameter num is the N of K-shot N-way.
        selected_val_label = np.random.choice(label_key_val, n_way) # make constant as a variable
        selected_val_data = []
        for label in selected_val_label:
            # second parameter num is the K of K-shot N-way.
            selected_val_data.extend(np.random.choice(val_dict[label], k_shot)) # make constant as a variable

        # Get Training set
        split_file_train = os.path.join(split_dir, 'train' + '.csv')
        assert os.path.isfile(split_file_train)
        with open(split_file_train, 'r') as f:
            split_train = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '']

        data_train, ori_labels_train = [x[0] for x in split_train], [x[1] for x in split_train]

        # Extend Train Loader using Validation data(Imbalanced dataset)
        data_train.extend(selected_val_data)
        for label in selected_val_label:
            labels = [label]*5
            ori_labels_train.extend(labels)
        data = data_train
        ori_labels = ori_labels_train

        # 엑셀에 저장된 이미지 index 값을 정수로 mapping 하는 과정
        label_key, indexes = np.unique(np.array(ori_labels), return_index=True)
        label_key = [ori_labels[index] for index in sorted(indexes)]
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]  

        self.root = root
        self.transform = transform
        self.data = data
        self.labels = mapped_labels  # label을 순서대로 0 부터 오름차순으로 매긴것, EX) 이미지 0~599번 까지는 0, 600~1199 까지는 1, so on...
        self.out_name = out_name
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert os.path.isfile(self.root+'/'+self.data[index])
        img = Image.open(self.root + '/' + self.data[index]).convert('RGB')
        label = self.labels[index]
        label = int(label)
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            # print(img.shape)
            return img, label, self.data[index]
        else:
            # print(img.shape)
            return img, label




class CategoriesSampler(Sampler):

    def __init__(self, label, n_iter, n_way, n_shot, n_query, clone_factor=1):
    # Assume 5shot 5 way 15 query
        
        self.n_iter = n_iter    # assume 400
        self.n_way = n_way      # 5
        self.n_shot = n_shot    # 5
        self.n_query = n_query  # 15
        self.clone_factor = clone_factor

        label = np.array(label)
        self.m_ind = [] # store the index of the label
        unique = np.unique(label)
        unique = np.sort(unique) # unique eliminate duplication, so the total number of training classes are 64

        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        # self.m_ind has a shape of (64,600), which represents 600 images of 64 classes.

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter): # iterates 400 times
            batch_gallery = []
            batch_query = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way] # torch.randperm: creates a range of parameters in random order, EX) if 5 then [1,0,4,3,2]
            for c in classes:  # pick 5 classes here
                l = self.m_ind[c.item()] # Get the index of each class
                pos = torch.randperm(l.size()[0]) # randomly pick some index

                
                #=====================================================================
                # 그냥 이부분만 복사하여 Categorie Sampler 붙여넣기 
                for i in range(self.clone_factor): # range 안에 변수 들어가면 됨 -> 1일때는 어차피 그대로이므로
                    batch_gallery.append(l[pos[:self.n_shot]]) # use 5 of them as support set
                #=====================================================================
                # 뿔리기

                batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]]) # use 15 of them as query set
            # 그리하여 총 100개, 25개는 support, 75개는 query 이다.
            # Total of 100 for each iteration, 25 are support set and 75 are query set
            batch = torch.cat(batch_gallery + batch_query)
            yield batch


class MultishotSampler(Sampler):

    def __init__(self, label, n_iter, n_way, n1_shot, n2_shot, n_query, n1_clone_factor = 1, n2_clone_factor = 1):
    # Assume 5shot 5 way 15 query
        
        self.n_iter = n_iter    # assume 400
        self.n_way = n_way      # 5
        self.n1_shot = n1_shot  # 1
        self.n2_shot = n2_shot  # 5
        self.n_query = n_query  # 15
        self.n1_clone_factor = n1_clone_factor
        self.n2_clone_factor = n2_clone_factor
        self.max_shot = max(n1_shot,n2_shot)

        label = np.array(label)
        self.m_ind = [] # store the index of the label
        unique = np.unique(label)
        unique = np.sort(unique) # unique eliminate duplication, so the total number of training classes are 64

        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        # self.m_ind has a shape of (64,600), which represents 600 images of 64 classes.

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter): # iterates 400 times
            n1_gallery = []
            n2_gallery = []
            batch_query = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way] # torch.randperm: creates a range of parameters in random order, EX) if 5 then [1,0,4,3,2]
            for c in classes:  # pick 5 classes here
                l = self.m_ind[c.item()] # Get the index of each class
                pos = torch.randperm(l.size()[0]) # randomly pick some index
                #=====================================================================
                for i in range(self.n1_clone_factor): # range 안에 변수 들어가면 됨 -> 1일때는 어차피 그대로이므로
                    n1_gallery.append(l[pos[:self.n1_shot]])
                for i in range(self.n2_clone_factor):
                    n2_gallery.append(l[pos[:self.n2_shot]])
                #=====================================================================
                batch_query.append(l[pos[self.max_shot:self.max_shot + self.n_query]]) # use 15 of them as query set
            # 그리하여 총 100개, 25개는 support, 75개는 query 이다.
            # Total of 100 for each iteration, 25 are support set and 75 are query set
            batch = torch.cat(n1_gallery + n2_gallery + batch_query)
            yield batch


class AnalyzeSampler(Sampler):

    def __init__(self, label, n_iter, n_class, n_base, n_same, n_diff, rand_sampling):
    # Assume 5shot 5 way 15 query
        assert n_class > 0

        self.n_iter = n_iter    # assume 400
        self.n_class = n_class
        self.n_base = n_base
        self.n_same = n_same
        self.n_diff = n_diff

        self.rand_sampling = rand_sampling
        self.batch = []

        label = np.array(label)
        self.m_ind = [] # store the index of the label
        unique = np.unique(label)
        unique = np.sort(unique) # unique eliminate duplication, so the total number of training classes are 64, [0~63]

        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        # self.m_ind has a shape of (64,600), which represents 600 images of 64 classes.

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        if self.batch:
            for i in range(self.n_iter): 
                yield self.batch[i]    
        else:
            for i in range(self.n_iter): # iterates 400 times
                batch_same = []
                batch_diff = []
                classes = torch.randperm(len(self.m_ind))[:self.n_class] # torch.randperm: creates a range of parameters in random order, EX) if 5 then [1,0,4,3,2]
                for i,c in enumerate(classes):  # pick classes here
                    l = self.m_ind[c.item()] # Get the index of each class, .item() makes tensor to scalar
                    pos = torch.randperm(l.size()[0]) # randomly pick some index
                    if i == 0:
                        batch_same.append(l[pos[:self.n_base]])
                        batch_same.append(l[pos[self.n_base : self.n_base + self.n_same]]) 
                    else:
                        batch_diff.append(l[pos[:self.n_diff]]) 
                batch = torch.cat(batch_same + batch_diff)
                if not self.rand_sampling:
                    self.batch.append(batch)
                yield batch

class TripletSampler(Sampler):

    def __init__(self, label, n_iter, n_batch, rand_sampling, n_base, n_same, n_diff, n_class=2):

        self.n_iter = n_iter   
        self.n_batch = n_batch
        self.n_class = n_class
        self.n_base = n_base
        self.n_same = n_same
        self.n_diff = n_diff

        self.rand_sampling = rand_sampling
        self.batch = []

        label = np.array(label)
        self.m_ind = [] # store the index of the label
        unique = np.unique(label)
        unique = np.sort(unique) # unique eliminate duplication, so the total number of training classes are 64, [0~63]

        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        # self.m_ind has a shape of (64,600), which represents 600 images of 64 classes.

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        if self.batch:
            for i in range(self.n_iter): 
                yield self.batch[i]    
        else:
            for _ in range(self.n_iter):
                batch_anchor = []
                batch_positive = []
                batch_negative = []
                for _ in range(self.n_batch): # Triplet Batch
                    classes = torch.randperm(len(self.m_ind))[:self.n_class] # torch.randperm: creates a range of parameters in random order, EX) if 5 then [1,0,4,3,2]
                    for i,c in enumerate(classes):  # pick classes here
                        l = self.m_ind[c.item()] # Get the index of each class, .item() makes tensor to scalar
                        pos = torch.randperm(l.size()[0]) # randomly pick some index
                        if i == 0:
                            batch_anchor.append(l[pos[:self.n_base]])
                            batch_positive.append(l[pos[self.n_base : self.n_base + self.n_same]]) 
                        else:
                            batch_negative.append(l[pos[:self.n_diff]]) 
                batch = torch.cat(batch_anchor + batch_positive + batch_negative)

                if not self.rand_sampling:
                    self.batch.append(batch)

                yield batch


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) 

def without_augment(size=84, enlarge=False):
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

def with_augment(size=84, disable_random_resize=False):
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

def nothing():
    return



# Torchvision 기본 제공 Image 변환
# __all__ = ["Compose", "ToTensor", "PILToTensor", "ConvertImageDtype", "ToPILImage", "Normalize", "Resize", "Scale",
#            "CenterCrop", "Pad", "Lambda", "RandomApply", "RandomChoice", "RandomOrder", "RandomCrop",
#            "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop", "RandomSizedCrop", "FiveCrop", "TenCrop",
#            "LinearTransformation", "ColorJitter", "RandomRotation", "RandomAffine", "Grayscale", "RandomGrayscale",
#            "RandomPerspective", "RandomErasing", "GaussianBlur", "InterpolationMode", "RandomInvert", "RandomPosterize",
#            "RandomSolarize", "RandomAdjustSharpness", "RandomAutocontrast", "RandomEqualize"]

# CenterCrop : 가운데만 잘라내기
# ColorJitter : 이미지의 밝기, 대비, 채도 및 색조를 랜덤 변경
# FiveCrop : 이미지를 가운데 + 나머지 4개 코너 이렇게 총 5개로 잘라낸다
# Grayscale : 흑백 (정확히는 광도로만 나타내는것)
# Pad : 이미지 Padding으로 채워서 확장하기
# RandomAffine : 랜덤 아핀변환, 아핀변환은 무게중심과 중점을 보존한다
# RandomApply : 주어진 확률에 따라 랜덤으로 Tranformation 적용
# RandomCrop: 랜덤하게 이미지의 한 부분을 잘라낸다.
# RandomGrayScale: 주어진 확률에 따라 랜덤으로 흑백 적용
# RandomHorizontalFlip: 주어진 확률에 따라 랜덤하게 수평으로 이미지 뒤집음(즉 좌우 반전)
# RandomPerspective: 주어진 확률에 따라 랜덤하게 이미지가 투영되는곳을 변환, 즉 시선의 위치를 바꾸는것이다.
# RandomResizedCrop: 랜덤하게 이미지의 한 부분을 잘라내고 비율을 바꾼다.
# RandomRotation: 랜덤하게 주어진 각도에 따라 이미지를 회전시킨다
# RandomVerticalFlip : 주어진 확률에 따라 랜덤하게 수직으로 이미지 뒤집는다(즉 상하 반전)
# Resize : 이미지 크기 변경
# TenCrop : FiveCrop 에다가 Flip(수평 또는 수직) 을 적용하고 FiveCrop 하여 총 10개로 자른다.
# GaussianBlur : 가우시안 Blur 적용
# LinearTransformation : 선형 변환
# Normalize : 평균과 표준편차를 이용해 이미지 Normalize
# RandomErasing: 이미지의 랜덤하게 사각형 지역을 선택하여 잘라낸다.
# ConvertImageDtype: tensor data type을 변경한다
# ToPILImage: tensor 나 numpy 배열을 PIL Image 형식으로 변환
# ToTensor: PIL Image 나 numpy 배열을 tensor로 변환한다.
# Lambda : 사용자가 직접 구현한 이미지 변환을 적용한다
