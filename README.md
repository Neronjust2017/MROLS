# Meta-Reweighting-with-Offline-Loss-Sequence

This is the code for the paper: 《Learning to Reweight Samples with Offline Loss Sequence》

## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 1.5.0 
- Torchvision 0.6.0


## Running MROLS with data bias (label noise and class imbalance)
Here are some examples with label noise:
```bash
(cifar10 symmetric noise)
cd noise/
python BaseModel.py --dataset cifar10 --corruption_type unif --corruption_prob 0.4 --model WideResNet --epochs 40 --seed 1
python MW-Net.py --dataset cifar10 --corruption_type unif --corruption_prob 0.4 --model WideResNet --epochs 40 --seed 1
python MROLS.py --dataset cifar10 --corruption_type unif --corruption_prob 0.4 --model WideResNet --epochs 40 --seed 1

(cifar10 asymmetric noise)
cd noise/
python BaseModel.py --dataset cifar10 --corruption_type flip2 --corruption_prob 0.4 --model ResNet --epochs 60 --seed 1
python MW-Net.py --dataset cifar10 --corruption_type flip2 --corruption_prob 0.4 --model ResNet --epochs 60 --seed 1
python MROLS.py --dataset cifar10 --corruption_type flip2 --corruption_prob 0.4 --model ResNet --epochs 60 --seed 1

(cifar100 symmetric noise)
cd noise/
python BaseModel.py --dataset cifar100 --corruption_type unif --corruption_prob 0.4 --model WideResNet --epochs 40 --seed 1
python MW-Net.py --dataset cifar100 --corruption_type unif --corruption_prob 0.4 --model WideResNet --epochs 40 --seed 1
python MROLS.py --dataset cifar100 --corruption_type unif --corruption_prob 0.4 --model WideResNet --epochs 40 --seed 1

(cifar100 asymmetric noise)
cd noise/
python BaseModel.py --dataset cifar100 --corruption_type flip2 --corruption_prob 0.4 --model ResNet --epochs 60 --seed 1
python MW-Net.py --dataset cifar100 --corruption_type flip2 --corruption_prob 0.4 --model ResNet --epochs 60 --seed 1
python MROLS.py --dataset cifar100 --corruption_type flip2 --corruption_prob 0.4 --model ResNet --epochs 60 --seed 1

```
Here are some examples with class imbalance:
```bash
(cifar10 class imbalance)
cd imbalance/
python BaseModel.py --dataset cifar10 --num_classes 10 --num_meta 100 --imb_factor 0.005 --seed 1
python MW-Net.py --dataset cifar10 --num_classes 10 --num_meta 100 --imb_factor 0.005 --seed 1
python MROLS.py --dataset cifar10 --num_classes 10 --num_meta 100 --imb_factor 0.005 --seed 1
```

## Acknowledgements
We thank the Pytorch implementation on glc(https://github.com/mmazeika/glc) and meta-weight-net(https://github.com/xjtushujun/meta-weight-net).





