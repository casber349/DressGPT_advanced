import torch
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from attribute_labeler_dataset import DressAttributeDataset
from attribute_labeler_model import DressAttributeLabelerModel
from torchvision import transforms

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎬 啟動本地 4050 【5-Fold 重心向量】知識庫打包引擎！設備: {device}")
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 288)),
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DressAttributeDataset(transform=val_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 1. 初始化 5 個模型大腦，並分別載入對應的 Fold 最佳權重
    models = []
    for fold in range(1, 6):
        checkpoint_path = f"checkpoints/dress_attribute_labeler_fold{fold}_best.pth"
        if not os.path.exists(checkpoint_path):
            print(f"❌ 找不到第 {fold} 折的權重檔: {checkpoint_path}，請確認是否已從 5070 複製過來！")
            return
        
        model = DressAttributeLabelerModel(num_attributes=38).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        models.append(model)
        
    print(f"🧠 成功解禁 5 顆大腦！準備進行特徵跨時空融融合...")
    
    all_fused_embeddings = []
    all_image_ids = []
    
    # 2. 讓 5 顆大腦聯手榨取 1000 張圖的黃金特徵
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            
            # 用來累加 5 個模型特徵的張量 [Batch_Size, 64]
            batch_embeddings_sum = torch.zeros(images.size(0), 64).to(device)
            
            # 讓 5 個模型輪流看這批照片，把 64維 Embedding 疊加起來
            for model in models:
                _, embedding_64d = model(images)
                batch_embeddings_sum += embedding_64d
                
            # 計算重心向量：除以 5 得到平均特徵
            batch_embeddings_avg = batch_embeddings_sum / 5.0
            
            # 移回 CPU 並轉為 numpy 陣列
            all_fused_embeddings.append(batch_embeddings_avg.cpu().numpy())
            
            # 💡 【完美修正點】從 dataset.data 的字典陣列中，精確解包出當前 Batch 的 image_id
            start_idx = i * 32
            end_idx = start_idx + images.size(0)
            batch_rows = dataset.data[start_idx:end_idx]
            batch_ids = [row['image_id'] for row in batch_rows]
            all_image_ids.extend(batch_ids)
            
            print(f"融合進度: [{end_idx}/{len(dataset)}] 張穿搭重心特徵提煉完畢...")
            
    # 3. 打包固化
    knowledge_base = {
        "image_ids": all_image_ids,
        "embeddings": np.vstack(all_fused_embeddings) 
    }
    
    output_path = "knowledge_base_64d.pt"
    torch.save(knowledge_base, output_path)
    print(f"\n🎉 奇蹟融合成功！DressGPT 5-Fold 究極知識庫已固化至: {output_path}")
    print(f"最終矩陣形狀: {knowledge_base['embeddings'].shape}，穩健度點滿！")

if __name__ == "__main__":
    main()