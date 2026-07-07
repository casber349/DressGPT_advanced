import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class DressStyleClassifierModel(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        print("正在載入 ConvNeXt-Tiny 預訓練大腦 (風格分類器版)...")
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # 拿到 ConvNeXt 原廠分類器的輸入維度 (768)
        in_features = self.backbone.classifier[2].in_features
        
        # 💡 沿用與 Scorer 完全相同的多層非線性特徵緩衝架構，確保模型理解特徵的穩定度
        # 唯一的差別是將最後一層的輸出從 1 改為動態傳入的類別數 (num_classes)
        self.backbone.classifier[2] = nn.Sequential(
            nn.Linear(in_features, 256),      # 第一層：768 降到 256，舒緩資訊瓶頸
            nn.LayerNorm(256),                # 標準化：讓數據更穩定，防止梯度爆炸
            nn.GELU(),                        # 激活函數
            nn.Dropout(0.15),                 # 安全帶：防止過擬合

            nn.Linear(256, 64),               # 第二層：256 降到 64
            nn.LayerNorm(64),                 # 標準化
            nn.GELU(),                        # 激活函數

            nn.Linear(64, num_classes)        # 第三層：不再是 1 個分數，而是輸出各風格的未歸一化預測值 (Logits)
        )
        print(f"🔥 成功將最後一層升級為【多層非線性 Style Classifier 思考頭】！當前設定類別數: {num_classes}")
        
    def forward(self, x):
        return self.backbone(x)

# --- 獨立測試區塊 ---
if __name__ == "__main__":
    # 檢查 CUDA (5070) 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"當前測試使用的設備: {device}")
    
    # 初始化模型（帶入剛才測試出來的 19 類）並送到 5070 顯卡上
    NUM_CLASSES = 19
    model = DressStyleClassifierModel(num_classes=NUM_CLASSES).to(device)
    
    # 模擬一包資料 (Batch Size = 4, 彩色 3 通道, 高 512, 寬 288)
    print(f"\n正在模擬一組 4 張 16:9 的穿搭照片送進 5070...")
    fake_images = torch.randn(4, 3, 512, 288).to(device)
    
    # 讓模型跑一次前向傳播 (Forward Pass)
    with torch.no_grad():
        logits = model(fake_images)
        # 💡 使用 argmax 模擬找出每一張照片機率最高（Logits 最大）的風格 ID
        predicted_ids = torch.argmax(logits, dim=1)
        
    print("\n--- 5070 顯卡測試成功！ ---")
    print(f"模型預測輸出的 Tensor 形狀: {logits.shape} (代表得到了 4 張圖在 {NUM_CLASSES} 個風格類別上的原始 Logits)")
    print(f"模擬預測出的最高機率風格整數 ID 分別為: {predicted_ids.cpu().numpy()}")
    print("--------------------------------")