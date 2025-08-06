
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

current_file_path = __file__
print(f"current_file_path: {current_file_path}")

# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
# 去除文件名后缀
file_name_without_extension = os.path.splitext(os.path.basename(current_file_path))[0]

results_path = os.path.join(current_dir, "../../../", f'output/{file_name_without_extension}')

os.makedirs(results_path, exist_ok=True)


def to_pil(data):
    # Convert tensor to numpy array
    if torch.is_tensor(data):
        data = data.cpu().numpy()

    r = Image.fromarray((data[0] * 255).astype(np.uint8))
    g = Image.fromarray((data[1] * 255).astype(np.uint8))
    b = Image.fromarray((data[2] * 255).astype(np.uint8))
    pil_img = Image.merge('RGB', (r,g,b))
    return pil_img

# 在返回结果之前添加可视化代码
def visualize_dataset(loader, dataset_name, split_name, class_names):
    dataset_path = os.path.join(results_path, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)

    # Check if images already exist
    existing_images = [f for f in os.listdir(dataset_path) if f.startswith(f'{dataset_name}_{split_name}_group')]
    if existing_images:
        print(f"Images already exist for {dataset_name} {split_name}. Skipping visualization.")
        return

    # 收集图片和标签
    images = []
    labels = []
    samples_per_class = {i: 0 for i in range(len(class_names))}
    max_samples = 10  # 每个类别收集10张图片

    for batch_images, batch_labels in loader:
        for img, label in zip(batch_images, batch_labels):
            if samples_per_class[label.item()] < max_samples:
                images.append(img)
                labels.append(label.item())
                samples_per_class[label.item()] += 1

        # 检查是否所有类别都收集够了样本
        if all(count >= max_samples for count in samples_per_class.values()):
            break

    # 可视化并保存图片
    random_visualize(images, labels, class_names, dataset_name, split_name)

def split_labels_into_groups(label_names, group_size=10):
    """Split label names into groups of specified size"""
    return [label_names[i:i + group_size] for i in range(0, len(label_names), group_size)]

def random_visualize(imgs, labels, label_names, dataset_name, split_name):
    # 将标签分组，每组10个类别
    label_groups = split_labels_into_groups(label_names)

    for group_idx, group_labels in enumerate(label_groups):
        figure = plt.figure(figsize=(len(group_labels), 10))

        # 获取当前组的标签索引范围
        label_indices = [label_names.index(label) for label in group_labels]

        # 筛选属于当前组的图片
        valid_indices = [i for i, label in enumerate(labels) if label in label_indices]
        np.random.shuffle(valid_indices)

        count = {label: 0 for label in label_indices}

        for idx in valid_indices:
            label = labels[idx]
            if count[label] >= 10:
                continue
            if all(c >= 10 for c in count.values()):
                break

            img = to_pil(imgs[idx])
            label_name = label_names[label]

            # 调整subplot索引计算
            relative_label_idx = label_indices.index(label)
            subplot_idx = count[label] * len(group_labels) + relative_label_idx + 1

            plt.subplot(10, len(group_labels), subplot_idx)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            if count[label] == 0:
                plt.title(label_name)

            count[label] += 1

        # 保存图片
        save_path = os.path.join(results_path, dataset_name, f'{dataset_name}_{split_name}_group{group_idx+1}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


