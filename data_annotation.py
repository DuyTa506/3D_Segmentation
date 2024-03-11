import os
import json
parent_folder_path = '/app/duy55/Unet-Like-3D-Medical-Image-Segmentation/brats_2021_task1/BraTS2021_Training_Data'
subfolders = [f for f in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, f))]
num_folders = len(subfolders)
print(f"Số lượng mẫu trong '{parent_folder_path}' là: {num_folders}")



folder_data = []

for fold_number in os.listdir(parent_folder_path):
    fold_path = os.path.join(parent_folder_path, fold_number)

    if os.path.isdir(fold_path):
        entry = {"fold": 0, "image": [], "label": ""}

        for file_type in ['flair', 't1ce', 't1', 't2']:
            file_name = f"{fold_number}_{file_type}.nii.gz"
            file_path = os.path.join(fold_path, file_name)

            if os.path.exists(file_path):

                entry["image"].append(os.path.abspath(file_path))

        label_name = f"{fold_number}_seg.nii.gz"
        label_path = os.path.join(fold_path, label_name)
        if os.path.exists(label_path):
            entry["label"] = os.path.abspath(label_path)

        folder_data.append(entry)


json_data = {"training": folder_data}

json_file_path = '/app/duy55/Unet-Like-3D-Medical-Image-Segmentation/info.json'
with open(json_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=2)

print(f"Thông tin đã được ghi vào {json_file_path}")
