import os
import json

def generate_combined_coco_list(train_dir, output_file):
    """
    生成合并的COCO数据集图像列表
    
    参数:
        train_dir: 训练集图像目录
        val_dir: 验证集图像目录
        output_file: 输出文件路径
    """
    # 获取训练集和验证集图像文件
    # train_images = sorted([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
    val_images = sorted([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
    
    # 合并图像列表
    all_images =  val_images
    
    # 生成并写入JSON行
    with open(output_file, 'w') as f:
        for idx, img_file in enumerate(all_images, start=1):
            record = {
                "id": idx,
                "image": img_file,
                "query": "Describe this image in detail.",
                "question_id": idx,
                "text": "Describe this image in detail."
            }
            f.write(json.dumps(record) + '\n')

    print(len(all_images))
coco_train_dir = "data/train2014"
# 使用示例
generate_combined_coco_list(coco_train_dir, 'data/train2014_caption.json')