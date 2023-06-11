python pretrain.py --dir_data 'data/cifar-10-python.tar.gz' --save 'parameters/main/'

python vggprune.py --dir_data 'data/cifar-10-python.tar.gz' --model 'parameters/main/model_best.pth.tar' --save 'parameters/prune/' --ratio 0.9

python main_finetune.py --dir_data 'data/cifar-10-python.tar.gz' --refine 'parameters/prune/pruned_layer8_0.9.pth.tar' --save 'parameters/finetune/' --ratio 0.9


python test.py --dir_data 'data/cifar-10-python.tar.gz' --baseline 'parameters/main/model_best.pth.tar' --pruned 'parameters/prune/pruned_layer8_0.9.pth.tar' --finetune 'parameters/finetune/finetune_model_best_layer8_0.9.pth.tar' --ratio 0.9