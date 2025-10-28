import os
import pickle
import time
import torch
import json
import argparse
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import (
    load_files,
    save_pickle,
    fix_seed,
    print_model,
    CosineAnnealingWarmUpRestarts,
)
from model import HypergraphTransformer
from modules.logger import setup_logger, get_rank
from dataloader import KVQA, FVQA, load_FVQA_data

from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from distributed_utils import init_distributed_mode, dist, reduce_value
import sys

def eval_epoch(model, loader, args):
    model.eval()
    total_right = torch.zeros(1).to(args.device)
    total_right_aset = torch.zeros(1).to(args.device)
    total_num = torch.zeros(1).to(args.device)
    

    
    for b_idx, batch in enumerate(loader):
        batch = [b.cuda()for b in batch]
        labels = batch[-1]

        pred = model(batch)
        pred_score, pred_ans = pred.max(1)

        nz_idxs = labels.nonzero()
        right = labels[nz_idxs] == pred_ans[nz_idxs]
        total_right += right.sum().item()
        total_num += len(labels)


        if "fvqa" in args.data_name:
            _, top3_indices = torch.topk(pred, 3)
            for idx, indices in enumerate(top3_indices):
                if labels[idx] in indices:
                    total_right_aset += 1

       
    if args.device != torch.device("cpu"):
        torch.cuda.synchronize(args.device)

   
    total_right=reduce_value(total_right,average=False).item()
    total_right_aset=reduce_value(total_right_aset,average=False).item()
    total_num=reduce_value(total_num,average=False).item()

    return total_right, total_right_aset, total_num


def inference(model, test_loader, ckpt_path, args, logger,task_idx=-1, res=False):
    last_ckpt = os.path.join(ckpt_path, "ckpt_best.pth.tar")
    checkpoint = torch.load(last_ckpt,encoding="utf8")

    # if list(checkpoint["state_dict"].keys())[0].startswith("module."):
    #     checkpoint["state_dict"] = {
    #         k[7:]: v for k, v in checkpoint["state_dict"].items()
    #     }

    if not list(checkpoint["state_dict"].keys())[0].startswith("module."):
        checkpoint["state_dict"] = {
            "module." + k: v for k, v in checkpoint["state_dict"].items()
        }
    # print("print parameters:",checkpoint["state_dict"].keys())
    model.load_state_dict(checkpoint["state_dict"],strict=False)
    print("load: %s" % (last_ckpt))
    dist.barrier()
    
    total_right, total_right_aset, total_num = eval_epoch(model, test_loader, args)
    accuracy = total_right / total_num

    
    
    if "fvqa" in args.data_name:
        if args.rank==0:
            logger.info("## Test accuracy (@1) : %f" % (accuracy))
    accuracy = total_right_aset / total_num
        
    return accuracy


def main():
    """parse config file"""
    parser = argparse.ArgumentParser(description="experiments")
    parser.add_argument("--data_name", default="fvqa_sp0")
    parser.add_argument("--cfg", default="ht_fvqat")
    parser.add_argument("--exp_name", default="ht_fvqa_sp0")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--per_cate", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--selected", action="store_true")
    parser.add_argument("--abl_only_ga", action="store_true")
    parser.add_argument("--abl_only_sa", action="store_true")
    parser.add_argument("--abl_ans_fc", action="store_true")
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--q_opt", type=str, default="org")
    parser.add_argument("--n_hop", type=int, default=1)
    parser.add_argument("--local_rank",default=-1,type=int)
    parser.add_argument("--syncBN",default=True,type=bool)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=2, type=int, help='number of distributed processes')
    args = parser.parse_args()  


    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)   


    config_file = "configs/%s.yaml" % (args.cfg)
    model_cfg = load_files(config_file)

    fix_seed(model_cfg["MODEL"]["SEED"])

    train_transfomer=transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])
    test_transformer=transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])



    log_path = model_cfg["RES"]["LOG"] + args.exp_name
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ckpt_path = model_cfg["RES"]["CKPT"] + args.exp_name
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    logger = setup_logger(args.exp_name, log_path, get_rank())
    if rank==0:
        logger.info(model_cfg["MODEL"])
        logger.info(args)



    # ------------ Construct Dataset Class ------------------------------------
    datasets = {}
    if args.data_name == "kvqa":
        modes = ["train", "val", "test"]
        n_node_lists = []
        for mode in modes:
            fname = ckpt_path + "/%s_cache.pkl" % (mode)
            if os.path.isfile(fname):
                datasets[mode] = load_files(fname)
            else:
                if mode=="train":
                    transform=train_transfomer
                else:
                    transform=test_transformer
                data = KVQA(model_cfg, args, mode,transform)
                datasets[mode] = data
                save_pickle(data, fname)
            n_node_lists.append(max(datasets[mode].n_node))
        max_n_node = max(n_node_lists)

        for mode in modes:
            datasets[mode].max_n_node = max_n_node

    elif "fvqa" in args.data_name:
        train, test = load_FVQA_data(model_cfg, args)
        datasets["train"] = FVQA(model_cfg, args, train,train_transfomer)
        datasets["test"] = FVQA(model_cfg, args, test,test_transformer)


    train_sampler=DistributedSampler(datasets["train"])
    if "fvqa" in args.data_name:
        val_sampler=DistributedSampler(datasets["test"])
    else:
        val_sampler=DistributedSampler(datasets["val"])

    train_loader = DataLoader(
        datasets["train"],
        sampler=train_sampler,
        batch_size= model_cfg["MODEL"]["BATCH_SIZE"],
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )
    if "fvqa" in args.data_name:
        val_loader = DataLoader(
            datasets["test"],
            sampler=val_sampler,
            batch_size=model_cfg["MODEL"]["BATCH_SIZE"],
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True
        )
    else:
        val_loader = DataLoader(
            datasets["val"],
            sampler=val_sampler,
            batch_size=model_cfg["MODEL"]["BATCH_SIZE"],
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True
        )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=model_cfg["MODEL"]["BATCH_SIZE"],
        num_workers=args.num_workers,
        shuffle=False
    )

    # ------------ Model -----------------------
    
    model = HypergraphTransformer(model_cfg, args).cuda()

    if rank==0:
        torch.save(model.state_dict(),"./ckpt/initial_weights.pt")
    dist.barrier()
    model.load_state_dict(torch.load("./ckpt/initial_weights.pt", map_location=device))

    if args.syncBN:
        model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model=DistributedDataParallel(model,device_ids=[args.gpu])

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, T_0=150, T_mult=1, eta_max=0.001, T_up=10, gamma=0.5
    )
    

    # ------------ Evaluate -----------------------
    if args.inference == True:
        if args.per_cate == False:
            with torch.no_grad():
                test_acc_final = inference(model, test_loader, ckpt_path, args, logger, res=False)
            if rank==0:
                logger.info("test accuracy (final) : %f" % (test_acc_final))

        else:  # analysis on question types (KVQA only)
            if args.data_name == "kvqa":
                cate_accu_test = []
                qtypes = load_files(model_cfg["DATASET"]["IDX2QTYPE"])
                for task_idx in range(10):
                    test = KVQA(model_cfg, args, "test", task_idx)
                    test.max_n_node = max_n_node
                    test_loader = DataLoader(
                        test,
                        batch_size=model_cfg["MODEL"]["BATCH_SIZE"],
                        num_workers=args.num_workers,
                        shuffle=False,
                    )
                    accu = inference(
                        model, test_loader, ckpt_path, args, task_idx=task_idx, res=True
                    )
                    cate_accu_test.append(accu)
                print(qtypes[:10])
                print(cate_accu_test)
            else:
                raise NotImplementedError(
                    "Datasets except KVQA do not have categories for questions. Set per_cate as False."
                )
        return 0

    # ------------ Training -----------------------
    train_loss = []
    best_acc = 0.0

    for e_idx in range(0, args.max_epoch):
        train_sampler.set_epoch(e_idx)
        val_sampler.set_epoch(e_idx)
        model.train()
        total_right = torch.zeros(1).to(device)
        total_num = torch.zeros(1).to(device)
        total_right_aset = torch.zeros(1).to(device)
        

       
        for b_idx, batch in enumerate(train_loader):
            batch = [b.cuda() for b in batch]       
            labels = batch[-1]
            pred = model(batch)   
            pred_score, pred_ans = pred.max(1)
            loss = F.nll_loss(pred, labels)
            train_loss.append(loss.item())

            nz_idxs = labels.nonzero()
            right = labels[nz_idxs] == pred_ans[nz_idxs]
            total_right += right.sum().item()
            total_num += len(labels)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if args.schedule:
            lr_scheduler.step()
            
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        
        total_right=reduce_value(total_right,average=False).item()
        total_num=reduce_value(total_num,average=False).item()
        total_right_aset=reduce_value(total_right_aset,average=False).item()
        if rank == 0:
            if args.debug == False:
                tr_accu = total_right / total_num                              
                logger.info(
                    "epoch %i train accuracy : %f, %i/%i"
                    % (e_idx, tr_accu, total_right, total_num)
                )

        with torch.no_grad():
            total_right_val, total_right_aset_val, total_num_val = eval_epoch(
                model, val_loader, args
            )
        if rank==0:
            if args.debug == False:
                val_acc = total_right_val / total_num_val
                val_acc_aset = total_right_aset_val / total_num_val
               

                if "fvqa" in args.data_name:
                    # summary.add_scalar("accu_aset/val", val_acc_aset, e_idx)
                    logger.info(
                        "epoch %i val accuracy : %f, %i/%i / %f, %i/%i"
                        % (
                            e_idx,
                            val_acc,
                            total_right_val,
                            total_num_val,
                            val_acc_aset,
                            total_right_aset_val,
                            total_num_val,
                        )
                    )
                else:
                    logger.info(
                        "epoch %i val accuracy : %f, %i/%i"
                        % (e_idx, val_acc, total_right_val, total_num_val)
                    )
            
        
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(
                    {
                        "epoch_idx": e_idx,
                        "state_dict": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(ckpt_path, "ckpt_best.pth.tar"),
                )
            logger.info("## Current VAL Best : %f" % (best_acc))

    dist.barrier()
    with torch.no_grad():
        test_acc_final = inference(model, test_loader, ckpt_path, args,logger)
    if rank==0:
        logger.info("## Test accuracy : %f" % (test_acc_final))


if __name__ == "__main__":
    main()
