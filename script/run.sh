CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/train_net.py \
    --config-file ./configs/pcl/bagtricks_R50-ibn.yml  \
    --num-gpus 4 \
    2>&1 | tee logs/exp1-03-05.log
    
