# Model_Pruning

Algorithm reproduction of：Learning Efficient Convolutional Networks through Network Slimming (CVPR2017)


![](./asserts/pruning.pdf)

## Preparation

### Data

You need to place the CIFAR10 dataset in `data` folder.

## Train

First you need to pretrain the VGG model on CIFAR10:

```shell
python pretrain.py --dir_data 'data/cifar-10-python.tar.gz' --save 'parameters/main/'
```

Next you need to prune the model:

```shell
python prune.py --dir_data 'data/cifar-10-python.tar.gz' --model 'parameters/main/model_best.pth.tar' --save 'parameters/prune/' --ratio 0.9
```

You can change `ratio `  to get different pruned model.

Finally you should fine tune the pruned model:

```shell
python finetune.py --dir_data 'data/cifar-10-python.tar.gz' --refine 'parameters/prune/pruned_layer8_0.9.pth.tar' --save 'parameters/finetune/' --ratio 0.9
```

After completing the above steps, we can get the parameters of the pre-trained, pruned and fine-tuned model respectively.

By changing the parameters in the code, you can set different ratio for different convolution layers for testing.

## Test

Comparative tests are performed for the three weights obtained:

```shell
python test.py --dir_data 'data/cifar-10-python.tar.gz' --baseline 'parameters/main/model_best.pth.tar' --pruned 'parameters/prune/pruned_layer8_0.9.pth.tar' --finetune 'parameters/finetune/finetune_model_best_layer8_0.9.pth.tar' --ratio 0.9
```



























