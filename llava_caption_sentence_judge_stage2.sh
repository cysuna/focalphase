#!/bin/bash


max_processes=8  
running_processes=0
declare -A pid_to_gpu  

for j in {0..7}; do
  for i in {0..7}; do

    while [ $running_processes -ge $max_processes ]; do
      for pid in "${!pid_to_gpu[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
          unset pid_to_gpu["$pid"]
          ((running_processes--))
        fi
      done
      sleep 10  
    done

    CUDA_VISIBLE_DEVICES=$i python llava_caption_sentence_judge_stage2.py \
      --answers-file  exp/llava-out2/coco-blip$i-$j.jsonl \ 
      --image-folder train2014 \
      --question-file  exp/llava-out1/coco-blip$i-$j.jsonl \  
      --part $j &
    
    pid=$!
    pid_to_gpu["$pid"]=$i
    ((running_processes++))
    echo "start $pid @ GPU $i (part $j)"
  done
done

wait
echo "done"

