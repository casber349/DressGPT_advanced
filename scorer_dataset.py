import os
import pymysql
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv

# 讀取 .env 設定檔中的資料庫密碼
load_dotenv()

class DressScorerDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = []
        
        # 1. 連線到 5070 本機的 MySQL 資料庫
        connection = pymysql.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
        try:
            with connection.cursor() as cursor:
                # 2. 撈取 Scorer 任務所需的所有資料（1000 筆全部進去）
                sql = "SELECT image_path, overall_score FROM dress_dataset"
                cursor.execute(sql)
                self.data = cursor.fetchall()
                print(f"🎉 成功從資料庫載入 {len(self.data)} 筆 Scorer 訓練資料！")
        finally:
            connection.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        
        # 3. 讀取圖片路徑（結合當前工作目錄與資料庫內的路徑）
        img_path = os.path.join(os.getcwd(), row['image_path'])
        
        # 4. 讀取照片並確保是完整的 RGB 彩色資訊
        image = Image.open(img_path).convert('RGB')
        
        # 5. 取得對應的整體穿搭分數
        score = float(row['overall_score'])
        
        # 6. 執行圖片預處理（縮放與轉成 Tensor）
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(score, dtype=torch.float32)

# --- 獨立測試區塊 ---
if __name__ == "__main__":
    # 定義維持 9:16 比例且保留 RGB 原色的預處理
    # 原圖 576x1024，我們等比例縮小一半到 高 512, 寬 288，讓 5070 跑得又快又穩
    test_transform = transforms.Compose([
        transforms.Resize((512, 288)),  # 嚴格維持 9:16，絕不壓扁走樣
        transforms.ToTensor(),          # 自動保留 RGB 3 通道
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 初始化 Dataset
    print("正在嘗試連線資料庫並讀取資料...")
    dataset = DressScorerDataset(transform=test_transform)
    
    # 抽第一張出來驗證，確保一切符合預期
    if len(dataset) > 0:
        img, label = dataset[0]
        print("\n--- 檢查第一張樣本的輸出結果 ---")
        print(f"圖片 Tensor 形狀 (Shape): {img.shape}")
        print(f"通道數: {img.shape[0]} (3 代表完美的 RGB 彩色資訊！)")
        print(f"解析度: {img.shape[2]} x {img.shape[1]} (完美的直向 9:16 比例！)")
        print(f"穿搭整體分數 (Label): {label.item()}")
        print("--------------------------------")