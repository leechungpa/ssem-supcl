# Optimizing Embeddings with Supervised Contrastive Loss for Preventing Class Collapse

```python
python main.py experiment_name \
    --data_dir ./data --save_dir ./result \
    -a resnet18 --data CIFAR10 \
    --lr 0.05 --lr_decay_rate 0.1 --wd 0.0001 --warmup_from 0.01 --warm \
    --dim 128 \
    --batch_size $batch_size --n_augment 2 \
    --alpha 0.5 --temperature 0.1 \
    --seed 0 --gpu 0 --epoch 1000 --balanced_batch True
```