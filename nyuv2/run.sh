mkdir -p logs

dataroot=PATH_TO_DATA

nohup python -u model_segnet_mt.py --apply_augmentation --dataroot $dataroot --seed 1 --weight equal --method sdmgrad --alpha 0.3 > logs/sdmgrad-3e-1-sd1.log 2>&1 &