import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

def build_dataframe(root_path: str):
    image_paths, labels = [], []
    for label in os.listdir(root_path):
        label_path = os.path.join(root_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(label_path, img_file))
                    labels.append(label)
    df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    return df

def balance_and_merge(df, out_merge_dir='merged_images', resize=(224,224)):
    os.makedirs(out_merge_dir, exist_ok=True)
    max_samples = df['label'].value_counts().max()
    balanced_df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=max_samples, replace=True, random_state=42)
    ).reset_index(drop=True)

    merged_paths, merged_labels = [], []
    groups = balanced_df.groupby('label')
    resize_tf = transforms.Resize(resize)

    for label in balanced_df['label'].unique():
        group = groups.get_group(label)
        for i in range(len(group)//2):
            img1 = Image.open(group.iloc[i]['image_path']).convert('RGB')
            img2 = Image.open(group.iloc[-(i+1)]['image_path']).convert('RGB')
            img1, img2 = resize_tf(img1), resize_tf(img2)
            merged = np.uint8((np.array(img1) + np.array(img2)) / 2)
            merged_path = os.path.join(out_merge_dir, f'{label}_merged_{i}.jpg')
            Image.fromarray(merged).save(merged_path)
            merged_paths.append(merged_path)
            merged_labels.append(label)

    merged_df = pd.DataFrame({'image_path': merged_paths, 'label': merged_labels})
    return pd.concat([balanced_df, merged_df], ignore_index=True)

if __name__ == "__main__":
    root = r'brisc2025/classification_task/train'
    df = build_dataframe(root)
    print("Original counts:\n", df['label'].value_counts())
    df2 = balance_and_merge(df)
    df2.to_csv('dataset_dataframe.csv', index=False)
    print("After balancing + merging:", len(df2))
