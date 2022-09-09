import os
import sys
import argparse
import albumentations
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SequentialSampler
from torchvision import models as torchvision_models
from torch.utils.data import Dataset
import utils
import vision_transformer as vits
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
from sklearn.decomposition import PCA as sk_PCA

x0train = pd.read_csv(f'.../BBBC021_annotated.csv')
x0DMSO = pd.read_csv(f'.../BBBC021_DMSO.csv')

num_classes = 12

def extract_feature_pipeline(args, weights,channel):
    dataset_train = ReturnIndexDataset(x0train, channel)
    dataset_train2 = ReturnIndexDataset_DMSO(x0DMSO, channel)
    sampler = SequentialSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    sampler2 = SequentialSampler(dataset_train2)
    data_loader_train2 = torch.utils.data.DataLoader(
        dataset_train2,
        sampler=sampler2,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"Data loaded with {len(dataset_train)} imgs.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=12)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=12)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=12)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    print("Extracting features from train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print(train_features)
    print(train_features.size())
    
    print("Extracting features from DMSO set...")
    train_features2 = extract_features2(model, data_loader_train2, args.use_cuda)
    print(train_features2)
    print(train_features2.size())

    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, f"021_trainfeat.pth"))
        train_features_cpu = train_features.cpu()
        features_np = train_features_cpu.numpy() #convert to Numpy array
        df_csv = pd.DataFrame(features_np) #convert to a dataframe
        df_csv.to_csv("021_trainfeatures.csv",index=True) #save to file
        
    return train_features, train_features2#, test_features, train_labels, test_labels

@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index, compound, moa, replicate, treatment, plate, batch in metric_logger.log_every(data_loader, 10):
        index = index.cuda(non_blocking=True)
        compound = compound.cuda(non_blocking=True)
        moa = moa.cuda(non_blocking=True)
        replicate = replicate.cuda(non_blocking=True)
        treatment = treatment.cuda(non_blocking=True)
        plate = plate.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        feats = []
        for samp in range(4):
            a = samples[samp]
            a = a.cuda(non_blocking=True)
            if multiscale:
                feats_hold = utils.multi_scale(a, model)
            else:
                feats_hold = model(a).clone()
                
            feats.append(feats_hold)

        feats = torch.median(torch.stack(feats),dim=0)
        feats = feats[0]
        feats = feats.flatten()
        feats = torch.cat((feats,compound,moa,replicate,treatment,plate,batch),0)
        feats = feats.unsqueeze(0)          
        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features

def extract_features2(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        index = index.cuda(non_blocking=True)
        feats = []
        for samp in range(4):
            a = samples[samp]
            a = a.cuda(non_blocking=True)
            if multiscale:
                feats_hold = utils.multi_scale(a, model)
            else:
                feats_hold = model(a).clone()
                
            feats.append(feats_hold)

        feats = torch.median(torch.stack(feats),dim=0)
        feats = feats[0]
        feats = feats.flatten()
        feats = feats.unsqueeze(0)            
        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


def correct_tvn(features_DMSO, features_all):
    DMSO_features_cpu = features_DMSO.cpu()
    features_DMSO_np = DMSO_features_cpu.numpy() 
    features_cpu = features_all.cpu()
    features_all_np = features_cpu.numpy() 
    labels = features_all_np[:,-6:]
    features_all_np = features_all_np[:,:-6]
    p = sk_PCA(n_components=384, whiten=True).fit(features_DMSO_np)
    features_all = p.transform(features_all_np)
    features_all = np.concatenate([features_all,labels], axis=1)
    return features_all


def Aggregate_features_NSC(features, channel, epoch):
    features_np = features
    df = pd.DataFrame(features_np)
    df.rename(columns={ df.columns[384]: "compound" }, inplace = True)
    df.rename(columns={ df.columns[385]: "moa" }, inplace = True)
    df.rename(columns={ df.columns[386]: "replicate" }, inplace = True)
    df.rename(columns={ df.columns[387]: "treatment" }, inplace = True)
    df.rename(columns={ df.columns[388]: "plate" }, inplace = True)
    df.rename(columns={ df.columns[389]: "batch" }, inplace = True)
    df = df.groupby(['treatment','plate'],as_index=False).mean()
    print(df)
    df = df.groupby('treatment').median()
    df = df.drop("replicate", axis=1)
    df = df.drop("plate", axis=1)
    df - df.drop("batch",axis=1)
    df.to_csv(f"aggregated_features_{channel}_epoch_{epoch}.csv",index=True)
    return df


def Aggregate_features_NSCB(features, channel, epoch):
    features_np = features
    df = pd.DataFrame(features_np)
    df.rename(columns={ df.columns[384]: "compound" }, inplace = True)
    df.rename(columns={ df.columns[385]: "moa" }, inplace = True)
    df.rename(columns={ df.columns[386]: "replicate" }, inplace = True)
    df.rename(columns={ df.columns[387]: "treatment" }, inplace = True)
    df.rename(columns={ df.columns[388]: "plate" }, inplace = True)
    df.rename(columns={ df.columns[389]: "batch" }, inplace = True)
    df = df.groupby(['treatment','plate'],as_index=False).mean()
    print(df)
    df = df.groupby('treatment').median()
    df = df.drop("replicate", axis=1)
    df = df.drop("plate", axis=1)
    df.to_csv(f"NSCB_aggregated_features_weak_compound_{channel}_DINO_epoch_{epoch}.csv",index=True)
    return df



def NSC_function(features, channel, epoch):
    df = Aggregate_features_NSC(features, channel, epoch)
    label_df = df[["compound", "moa"]]
    feature_df = df.iloc[: , :-2]
    feature_df = pd.DataFrame(feature_df)
    print(feature_df)
    print(label_df)
    tally = []
    for idx in range(len(label_df)):
        feature = feature_df.iloc[[idx]]
        same_compound_feat = label_df.iloc[[idx]]
        same_compound_val = same_compound_feat[["compound"]]
        same_compound_val = same_compound_val.to_numpy()
        same_compound_val = same_compound_val.item(0)
        drop_index = label_df.loc[label_df['compound'] == same_compound_val]
        drop_index = drop_index.index
        remaining_features1 = feature_df.drop(drop_index)
        remaining_features = remaining_features1#
        remaining_features = remaining_features.reset_index(drop=True)
     
        remaining_features['cos_sim'] = cosine_similarity(remaining_features, feature).reshape(-1)
        nn = remaining_features[['cos_sim']].idxmax()
        label_df_dropped = label_df.drop(drop_index)
        dif_compound_val = label_df_dropped.iloc[nn]
        moa_dif = dif_compound_val[["moa"]]
        moa_dif = moa_dif.to_numpy()
        moa_dif = moa_dif.item(0)

        moa_orig = same_compound_feat[["moa"]]
        moa_orig = moa_orig.to_numpy()
        moa_orig = moa_orig.item(0)
               
        if moa_orig == moa_dif:
            tally.append(1)
        else:
            tally.append(0)
        a_ret = np.mean(tally)
        print(a_ret)
    return a_ret
        
def NSCB_function(features, channel, epoch):
    df = Aggregate_features_NSCB(features, channel, epoch)
    df = pd.DataFrame(df)
    df = df.drop([17,18,19,20,21,22,60,61,62,63,64], axis=0)
    label_df = df[["compound", "moa", "batch"]]
    feature_df = df.iloc[: , :-3]
    feature_df = pd.DataFrame(feature_df)
    tally = []
    for idx in range(len(label_df)):
        feature = feature_df.iloc[[idx]]
        same_compound_feat = label_df.iloc[[idx]]
        same_compound_val = same_compound_feat[["compound"]]
        same_compound_val = same_compound_val.to_numpy()
        same_compound_val = same_compound_val.item(0)
        same_batch_val = same_compound_feat[["batch"]]
        same_batch_val = same_batch_val.to_numpy()
        same_batch_val = same_batch_val.item(0)
        drop_index1 = label_df.loc[label_df['compound'] == same_compound_val]
        remaining_features = feature_df.drop(drop_index1.index)
        label_df_dropped = label_df.drop(drop_index1.index)        
        drop_index2 = label_df_dropped.loc[label_df_dropped['batch'] == same_batch_val]
        label_df_dropped = label_df_dropped.drop(drop_index2.index)
        remaining_features1 = remaining_features.drop(drop_index2.index)
        remaining_features = remaining_features1     
        remaining_features['cos_sim'] = cosine_similarity(remaining_features, feature).reshape(-1)
        nn = remaining_features[['cos_sim']].idxmax()        
        dif_compound_val = label_df_dropped.loc[nn]
        moa_dif = dif_compound_val[["moa"]]
        moa_dif = moa_dif.to_numpy()
        moa_dif = moa_dif.item(0)
        moa_orig = same_compound_feat[["moa"]]
        moa_orig = moa_orig.to_numpy()
        moa_orig = moa_orig.item(0)
               
        if moa_orig == moa_dif:
            tally.append(1)
        else:
            tally.append(0)
        a_ret = np.mean(tally)
        print(a_ret)
    return a_ret

class ReturnIndexDataset(Dataset):
    def __init__(self, path0, channel):
        
        self.X0 = path0[args.channel_headers[channel]]
        self.y_moa = path0['Unique_MoA']
        self.y_compound = path0['Unique_Compounds']
        self.y_replicate = path0['Replicate']
        self.y_treatment = path0['Unique_Treatments']
        self.y_plate = path0['Plate']
        self.y_batch = path0['Batch']
           
        self.aug0 = albumentations.Compose([
        albumentations.Normalize(mean=[0],std=[1],max_pixel_value=10000, always_apply=True),])
        self.aug1 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=64, y_min=32, x_max=320, y_max=256, always_apply=True),])
        self.aug2 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=320, y_min=32, x_max=544, y_max=256, always_apply=True),])
        self.aug3 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=64, y_min=256, x_max=320, y_max=480, always_apply=True),])
        self.aug4 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=320, y_min=256, x_max=544, y_max=480, always_apply=True),])
    
    def __len__(self):
        return (len(self.X0))  
        
    def __getitem__(self,idx):
        Aimage = Image.open(self.X0[idx])
        Aimage = np.array(Aimage)
        Aimage[Aimage > 10000] = 10000
        crops = []
        transformed0 = self.aug0(image=Aimage)
        image = transformed0['image']
        image_0 = image.astype(np.float32)
        
        transformed1 = self.aug1(image=image_0)
        transformed2 = self.aug2(image=image_0)
        transformed3 = self.aug3(image=image_0)
        transformed4 = self.aug4(image=image_0)
        
        image1 = transformed1['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        image1 = transformed2['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        image1 = transformed3['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        image1 = transformed4['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        moa = self.y_moa[idx]
        compound = self.y_compound[idx]
        replicate = self.y_replicate[idx]
        treatment = self.y_treatment[idx]
        plate = self.y_plate[idx]
        batch = self.y_batch[idx]
        
        return crops, idx, compound, moa, replicate, treatment, plate, batch

class ReturnIndexDataset_DMSO(Dataset):
    def __init__(self, path0, channel):
        
        self.X0 = path0[args.channel_headers[channel]]                   
        self.aug0 = albumentations.Compose([
        albumentations.Normalize(mean=[0],std=[1],max_pixel_value=10000, always_apply=True),])
        self.aug1 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=64, y_min=32, x_max=320, y_max=256, always_apply=True),])
        self.aug2 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=320, y_min=32, x_max=544, y_max=256, always_apply=True),])
        self.aug3 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=64, y_min=256, x_max=320, y_max=480, always_apply=True),])
        self.aug4 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=320, y_min=256, x_max=544, y_max=480, always_apply=True),])
    
    def __len__(self):
        return (len(self.X0))  
        
    def __getitem__(self,idx):
        Aimage = Image.open(self.X0[idx])
        Aimage = np.array(Aimage)
        Aimage[Aimage > 10000] = 10000
        crops = []
        transformed0 = self.aug0(image=Aimage)
        image = transformed0['image']
        image_0 = image.astype(np.float32)
        
        transformed1 = self.aug1(image=image_0)
        transformed2 = self.aug2(image=image_0)
        transformed3 = self.aug3(image=image_0)
        transformed4 = self.aug4(image=image_0)
        
        image1 = transformed1['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        image1 = transformed2['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        image1 = transformed3['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        image1 = transformed4['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        return crops, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[1], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.04, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default='features8',
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path_train', default=(f'/BBBC021_annotated.csv'), type=str)
    parser.add_argument('--channel_headers', default= ['Image_FileName_DAPI','Image_FileName_Tubulin', 'Image_FileName_Actin'], type=list)

    args = parser.parse_args()
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    tally_epoch = []
    for channel in range(0,2):
        print(channel)
        for train_epoch in range(0,100,5):
            if channel == 0:
                weights = f'DAPI_weak_compound_DINO_checkpoint00{train_epoch}.pth'
#                weights = f'DAPI_DINO_checkpoint00{train_epoch}.pth'
#                weights = f'pretrain_full_checkpoint.pth'
            else:
                if channel == 1:
                    weights = f'Tubulin_weak_compound_DINO_checkpoint00{train_epoch}.pth'
#                    weights = f'Tubulin_DINO_checkpoint00{train_epoch}.pth'
#                    weights = f'pretrain_full_checkpoint.pth'
                else:
                    weights = f'Actin_weak_compound_DINO_checkpoint00{train_epoch}.pth'
#                    weights = f'Actin_DINO_checkpoint00{train_epoch}.pth'
#                    weights = f'pretrain_full_checkpoint.pth'

            train_features, DMSO_features = extract_feature_pipeline(args,weights,channel)
            if utils.get_rank() == 0:
                if args.use_cuda:
                    train_features = train_features.cuda()
                    DMSO_features = DMSO_features.cuda()
            
            train_features = correct_tvn(DMSO_features, train_features)
            nscb_epoch = NSCB_function(train_features, channel, train_epoch)
            tally_epoch.append(nscb_epoch)
            print(tally_epoch)
    dist.barrier()
    
