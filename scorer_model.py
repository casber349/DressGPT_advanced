import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class DressScorerModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("正在載入 ConvNeXt-Tiny 預訓練大腦...")
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # 拿到 ConvNeXt 原廠分類器的輸入維度 (768)
        in_features = self.backbone.classifier[2].in_features
        
        # 💡 將原本的單層 Linear 改造成「多層非線性高級拉桿」
        self.backbone.classifier[2] = nn.Sequential(
            nn.Linear(in_features, 256),      # 第一層：768 降到 256，舒緩資訊瓶頸
            nn.LayerNorm(256),                # 標準化：讓數據更穩定，防止梯度爆炸
            nn.GELU(),                        # 激活函數：現代大腦最愛的非線性激發
            nn.Dropout(0.15),                  # 安全帶：訓練時隨機關掉 15% 神經元，死死壓住過擬合

            nn.Linear(256, 64),               # 第二層：768 降到 256，舒緩資訊瓶頸
            nn.LayerNorm(64),                 # 標準化：讓數據更穩定，防止梯度爆炸
            nn.GELU(),                        # 激活函數：現代大腦最愛的非線性激發

            nn.Linear(64, 1)                  # 第三層：一個整體穿搭分數
        )
        print("🔥 成功將最後一層升級為【多層非線性 Scorer 思考頭】！")
        
    def forward(self, x):
        return self.backbone(x)

# --- 獨立測試區塊 ---
if __name__ == "__main__":
    # 檢查 CUDA (5070) 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"當前測試使用的設備: {device}")
    
    # 初始化模型並送到 5070 顯卡上
    model = DressScorerModel().to(device)
    
    # 模擬一包資料 (Batch Size = 4, 彩色 3 通道, 高 512, 寬 288)
    print("\n正在模擬一組 4 張 16:9 的穿搭照片送進 5070...")
    fake_images = torch.randn(4, 3, 512, 288).to(device)
    
    # 讓模型跑一次前向傳播 (Forward Pass)
    with torch.no_grad():
        predictions = model(fake_images)
        
    print("\n--- 5070 顯卡測試成功！ ---")
    print(f"模型預測輸出的 Tensor 形狀: {predictions.shape} (代表得到了 4 張圖的分數)")
    print(f"模擬預測出的分數分別為: \n{predictions.cpu().numpy().flatten()}")
    print("--------------------------------")