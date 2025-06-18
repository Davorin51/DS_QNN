#!/usr/bin/env python
"""Prune & fine‑tune CIFAR‑trained checkpoints (32×32 by default).

Changes vs previous version
---------------------------
* **input‑size flag** (`--input-size`, default **32**) so everything (example input, FLOPs) matches CIFAR‑10 resolution.
* Safe if you resized to 224 — just pass `--input-size 224`.

python scripts/prune_and_finetune.py \
    --arch resnet18 \
    --ckpt models/resnet18_baseline.pth \
    --data ./datasets/cifar10 \
    --ratio 0.5 \
    --ft-epochs 5 \
    --batch-size 128

"""
from __future__ import annotations

import argparse, csv, time
from pathlib import Path
from datetime import datetime
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

try:
    import torch_pruning as tp
except ImportError as e:
    raise ImportError("pip install torch_pruning") from e

# ------------- Data ------------------

def get_loaders(root: Path | str, bs: int, img_size: int):
    tf_base = [transforms.RandomHorizontalFlip(), transforms.ToTensor(),
               transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))]
    if img_size != 32:
        tf_base.insert(0, transforms.Resize(img_size))
    train_ds = datasets.CIFAR10(root=str(root), train=True, download=True, transform=transforms.Compose(tf_base))
    test_ds  = datasets.CIFAR10(root=str(root), train=False, download=True, transform=transforms.Compose(tf_base[1:]))
    return (DataLoader(train_ds, bs, True, num_workers=4, pin_memory=True),
            DataLoader(test_ds,  bs, False, num_workers=4, pin_memory=True))

# ------------- Model builder (same) --

def build_model(arch, n_cls=10):
    a=arch.lower()
    if a=="vgg16":
        m=models.vgg16_bn(); m.classifier[6]=nn.Linear(m.classifier[6].in_features,n_cls)
    elif a=="resnet18":
        m=models.resnet18(); m.fc=nn.Linear(m.fc.in_features,n_cls)
    elif a in {"mobilenet_v3","mobilenet_v3_large","mobilenetv3"}:
        m=models.mobilenet_v3_large(); m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,n_cls)
    elif a in {"efficientnet","efficientnet_b0","efficientnetv1"}:
        m=models.efficientnet_b0(); m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,n_cls)
    else: raise ValueError(arch)
    return m

# ---------- Helpers ---------------

def accuracy(o,y): return o.argmax(1).eq(y).float().mean().item()

def finetune(model, loaders, epochs, dev):
    tr, te = loaders; opt=optim.Adam(model.parameters(), lr=1e-3); crit=nn.CrossEntropyLoss()
    best=0.0
    for _ in range(epochs):
        model.train()
        for x,y in tr:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); out=model(x); loss=crit(out,y); loss.backward(); opt.step()
        model.eval(); acc=[]
        with torch.no_grad():
            for x,y in te: acc.append(accuracy(model(x.to(dev)),y.to(dev)))
        best=max(best,sum(acc)/len(acc))
    return best

# -------- Pruning fns ------------

def struct_prune(model, example, ratio, strategy):
    DG=tp.DependencyGraph().build_dependency(model, example_inputs=example)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            idx=strategy()(m.weight, amount=ratio)
            DG.get_pruning_plan(m, tp.prune_conv_out_channel, idx).exec()
    return model

def l1_prune(model, ex, r): return struct_prune(model, ex, r, tp.strategy.L1Strategy)

def rand_prune(model, ex, r): return struct_prune(model, ex, r, tp.strategy.RandomStrategy)

def unstruct_mag(model, ratio):
    to_prune=[(m,'weight') for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
    torch.nn.utils.prune.global_unstructured(to_prune, pruning_method=torch.nn.utils.prune.L1Unstructured, amount=ratio)
    for m,_ in to_prune: torch.nn.utils.prune.remove(m,'weight')
    return model

# -------- Metrics ---------------

def count_params(m): return sum(p.numel() for p in m.parameters())

def count_nz(m): return sum(p.nonzero().size(0) for p in m.parameters())

def flops(m, in_size):
    try: from ptflops import get_model_complexity_info
    except ImportError: return 0
    macs,_=get_model_complexity_info(m,(3,in_size,in_size),as_strings=False,print_per_layer_stat=False)
    return int(macs*2)

# ---------- CSV ---------------

def save(csv_path:Path,row:dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    hdr=not csv_path.exists()
    with open(csv_path,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=row.keys())
        if hdr: w.writeheader(); w.writerow(row)

# ---------- Main --------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arch',required=True); ap.add_argument('--ckpt',required=True)
    ap.add_argument('--data',default='./data'); ap.add_argument('--ratio',type=float,default=0.5)
    ap.add_argument('--ft-epochs',type=int,default=5); ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--input-size',type=int,default=32,help='CIFAR=32, 224 if resized')
    ap.add_argument('--results-root',default='./results'); ap.add_argument('--outdir',default='./models')
    args=ap.parse_args()

    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders=get_loaders(args.data,args.batch_size,args.input_size)
    ex=torch.randn(1,3,args.input_size,args.input_size,device=dev)

    STRATS={'struct_l1':l1_prune,'struct_rand':rand_prune,'unstruct_mag':unstruct_mag}

    for name,fn in STRATS.items():
        model=build_model(args.arch).to(dev)
        model.load_state_dict(torch.load(args.ckpt,map_location=dev))
        model=fn(model,ex,args.ratio) if 'struct' in name else fn(model,args.ratio)
        sparsity=1-count_nz(model)/count_params(model)
        acc=finetune(model,loaders,args.ft_epochs,dev)
        fl=flops(model.cpu(),args.input_size)
        ck=Path(args.outdir)/f"{args.arch}_{name}_{int(args.ratio*100)}.pth"; ck.parent.mkdir(exist_ok=True,parents=True)
        torch.save(model.state_dict(),ck)
        row=dict(timestamp=datetime.now().isoformat(timespec='seconds'),arch=args.arch,method=name,ratio=args.ratio,
                 post_acc=f"{acc:.4f}",sparsity=f"{sparsity:.4f}",flops=fl,model_size=ck.stat().st_size)
        save(Path(args.results_root)/args.arch/'prune_metrics.csv',row)
        print(f"{name}: acc={acc:.4f} sparsity={sparsity:.2%}")

if __name__=='__main__': main()
