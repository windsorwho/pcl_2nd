CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/train_net.py \
    --config-file ./configs/pcl/bagtricks_R101-ibn.yml  \
    --num-gpus 4 \
    2>&1 | tee logs/exp2-r101-03-06.log
    
