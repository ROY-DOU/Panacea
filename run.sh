#! /bin/bash
# dataset=Fdataset 
# python main1.py --dataset ${dataset}

dataset=Fdataset

# num_gpus=2

for i in {0..7}; do
    echo Run $((i + 1)) times 
    # python main1.py --gpu_id $i --dataset ${dataset} &
    taskset -c $i python main1.py --gpu_id $i --dataset ${dataset} &
done

wait

python merge.py