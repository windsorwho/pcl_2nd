CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/feat.py \
    --config-file ./configs/pcl/bagtricks_R50-ibn.yml  \
    --num-gpus 4 \
    --feat_file r50_ibn \
    2>&1 | tee -a logs/infer_r50.log
    
