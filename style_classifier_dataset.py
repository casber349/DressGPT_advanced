import os
import pymysql
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv

# 讀取 .env 設定檔
load_dotenv()

class DressStyleClassifierDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = []
        
        # 用於將文字風格轉換為數字 ID 的對照字典
        self.style_to_id = {}
        self.id_to_style = {}
        
        # 連線到本機的 MySQL 資料庫
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
                # 步驟 A：從儲存風格清單的 label_styles 表撈取 style_name，建立對照字典
                sql_styles = "SELECT style_name FROM label_styles ORDER BY style_name ASC"
                cursor.execute(sql_styles)
                style_rows = cursor.fetchall()
                
                for idx, row in enumerate(style_rows):
                    name = row['style_name']
                    self.style_to_id[name] = idx
                    self.id_to_style[idx] = name
                
                print(f"📊 成功動態建立風格對照表！總共有 {len(self.style_to_id)} 個風格類別。")
                
                # 步驟 B：從主表 dress_dataset 撈取訓練資料
                # 欄位修正為 main_style，並嚴格執行篩選分數 >= 5.00 且風格不為空的黃金樣本
                sql_data = """
                    SELECT image_path, main_style 
                    FROM dress_dataset 
                    WHERE overall_score >= 5.00 
                      AND main_style IS NOT NULL 
                      AND main_style != ''
                """
                cursor.execute(sql_data)
                self.data = cursor.fetchall()
                print(f"🎉 成功從主表過濾出 {len(self.data)} 筆核心風格訓練資料！(門檻值 >= 5.00)")
                
        finally:
            connection.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        
        # 讀取圖片路徑
        img_path = os.path.join(os.getcwd(), row['image_path'])
        
        # 確保是完整的 RGB 彩色資訊
        image = Image.open(img_path).convert('RGB')
        
        # 取得對應的風格文字（從 main_style 欄位），並轉成整數 ID
        style_str = row['main_style']
        style_id = self.style_to_id[style_str]
        
        # 執行圖片預處理
        if self.transform:
            image = self.transform(image)
            
        # 分類任務的標籤必須轉成 torch.long 型態
        return image, torch.tensor(style_id, dtype=torch.long)


# --- 獨立測試區塊 ---
if __name__ == "__main__":
    test_transform = transforms.Compose([
        transforms.Resize((512, 288)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("正在嘗試連線資料庫並初始化風格資料集...")
    dataset = DressStyleClassifierDataset(transform=test_transform)
    
    print("\n=== 檢查對照表映射結果 ===")
    for name, s_id in dataset.style_to_id.items():
        print(f"風格名稱: {name} ➔ 分配到的整數 ID: {s_id}")
        
    if len(dataset) > 0:
        img, label = dataset[0]
        print("\n=== 檢查第一張風格樣本的輸出結果 ===")
        print(f"圖片 Tensor 形狀 (Shape): {img.shape}")
        print(f"風格整數 ID (Label): {label.item()}")
        print(f"還原回資料庫文字名稱: {dataset.id_to_style[label.item()]}")
        print("--------------------------------")