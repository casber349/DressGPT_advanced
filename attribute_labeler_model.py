import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class DressAttributeLabelerModel(nn.Module):
    def __init__(self, num_attributes=38):
        super().__init__()
        print("正在載入 ConvNeXt-Tiny 預訓練大腦 (64維特徵緩衝版)...")
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = self.backbone.classifier[2].in_features
        
        # 💡 將緩衝架構拆開，以便我們在中間攔截 64 維度的特徵空間
        self.block1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        self.final_layer = nn.Linear(64, num_attributes)
        print(f"🔥 雙結構思考頭升級成功！支援 38 維 Logits 輸出與 64 維特徵過濾。")
        
    def forward(self, x):
        # 1. 骨幹網路提取
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1) # [Batch, 768]
        
        # 2. 通過小神經網路過濾雜訊
        x1 = self.block1(features)
        embedding_64d = self.block2(x1)        # 💡 關鍵特徵攔截點
        
        # 3. 分類器輸出
        logits = self.final_layer(embedding_64d)
        
        # 🔥 同時回傳：Logits (給 Train 計算 Loss) 與 64維 Embedding (給後續推薦/偷分數)
        return logits, embedding_64d

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DressAttributeLabelerModel(num_attributes=38).to(device)
    fake_images = torch.randn(4, 3, 512, 288).to(device)
    
    with torch.no_grad():
        logits, embeddings = model(fake_images)
        
    print("\n--- 5070 測試回傳確認 ---")
    print(f"Logits 形狀 (Train 專用): {logits.shape}")
    print(f"Embedding 形狀 (偷分數專用): {embeddings.shape}")