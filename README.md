# A Theoretical Framework for Preventing Class Collapse in Supervised Contrastive Learning


### For training on the CIFAR datasets

```python
CUDA_VISIBLE_DEVICES=0 python cifar/main.py experiment_name \
    --data_dir ./data --save_dir ./result \
    -a resnet18 --data CIFAR10 \
    --lr 0.05 --lr_decay_rate 0.1 --wd 0.0001 --warmup_from 0.01 --warm \
    --dim 128 \
    --batch_size $batch_size --n_augment 2 \
    --alpha 0.5 --temperature 0.1 \
    --seed 0 --gpu 0 --epoch 1000 --balanced_batch True
```


### For training on the ImageNet dataset

```python
for alpha in 0.0 0.5 1.0 0.2 0.8
do
    CUDA_VISIBLE_DEVICES=0,1 imagenet/python main.py supcon/alpha_${alpha} \
    -a resnet50 --lr 0.3 --dim 128 --T 0.1 --alpha ${alpha} \
    --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
    --dir my_dir
done
```