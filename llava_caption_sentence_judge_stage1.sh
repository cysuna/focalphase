#!/bin/bash

for j in {0..7}
do
  for i in {0..7}
  do
    CUDA_VISIBLE_DEVICES=$i python llava_caption_sentence_judge_stage1.py \
    --answers-file exp/llava-out1/coco-llava157$i-$j.jsonl \
    --image-folder coco/train2014 \
    --question-file exp/llava/coco-llava157$i.jsonl \
    --part $j &
  done
done
