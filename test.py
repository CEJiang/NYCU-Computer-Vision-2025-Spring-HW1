import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader


# 1. 測試資料集，支援 TTA（多版本 transform）
class TestDataset(Dataset):
    def __init__(self, image_files, tta_times=4):
        self.image_files = image_files
        self.tta_times = tta_times
        self.loader = default_loader

        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        path = self.image_files[index]
        image = self.loader(path)
        file_name = os.path.splitext(os.path.basename(path))[0]

        # 對單張 image 做多種 transform
        images = [tf(image) for tf in self.tta_transforms[:self.tta_times]]
        return torch.stack(images), file_name  # [TTA, C, H, W], file name


# 2. 載入測試資料，支援 TTA 數量設定
def load_test_data(data_path, batch_size, tta_times=4):
    image_files = sorted(glob.glob(os.path.join(data_path, "test", "*.*")))
    test_dataset = TestDataset(image_files, tta_times=tta_times)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2
    )
    return test_loader


# 3. 推論 + 儲存結果，使用 softmax 平均做最終預測
def inference_and_save(model, device, test_loader,
                       class_names, output_csv="prediction.csv"):
    model.eval()
    predictions = []

    with torch.no_grad():
        for stacked_images, file_names in test_loader:
            # stacked_images: [B, TTA, C, H, W]
            B, T, C, H, W = stacked_images.shape
            stacked_images = stacked_images.view(B * T, C, H, W).to(device)

            outputs = model(stacked_images)                # [B*T, num_classes]
            # [B, TTA, num_classes]
            outputs = outputs.view(B, T, -1)

            probs = torch.softmax(outputs, dim=2)          # 取 softmax 機率
            # TTA 平均後 [B, num_classes]
            avg_probs = probs.mean(dim=1)

            preds = avg_probs.argmax(dim=1).cpu().numpy()  # 最終預測類別

            for fn, pred in zip(file_names, preds):
                folder_name_str = class_names[pred]
                label_int = int(folder_name_str)
                predictions.append((fn, label_int))

    df = pd.DataFrame(predictions, columns=["image_name", "pred_label"])
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")


# 4. 主函式整合測試
def test_model(device, net, train_loader, args):
    net.load_state_dict(
        torch.load(
            "saved_models/best_model.pth",
            map_location=device,
            weights_only=False))
    test_loader = load_test_data(
        args.data_path,
        args.batch_size,
        tta_times=4)  # TTA 數量可調
    class_names = train_loader.dataset.classes

    inference_and_save(
        model=net,
        device=device,
        test_loader=test_loader,
        class_names=class_names,
        output_csv="prediction.csv"
    )
