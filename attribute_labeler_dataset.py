import os
import pymysql
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv

# 讀取 .env 設定檔
load_dotenv()

class DressAttributeDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = []
        
        # 建立雙向對照字典（負責 38 種標籤名稱與一維向量 ID 的轉換）
        self.attr_to_id = {}
        self.id_to_attr = {}
        
        # 連線到 5070 的 MySQL 資料庫
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
                # 💡 步驟 A：依名稱排序撈取 38 種標籤，固化成不變的 0 ~ 37 號索引
                sql_attrs = "SELECT attr_name FROM label_attributes ORDER BY attr_name ASC"
                cursor.execute(sql_attrs)
                attr_rows = cursor.fetchall()
                
                for idx, row in enumerate(attr_rows):
                    name = row['attr_name']
                    self.attr_to_id[name] = idx
                    self.id_to_attr[idx] = name
                
                self.num_attributes = len(self.attr_to_id) # 這裡會精確等於 38
                print(f"📦 成功建立屬性對照表！總共有 {self.num_attributes} 個特徵維度。")
                
                # 💡 步驟 B：從主表撈取所有影像的 image_id 與圖片路域
                sql_images = "SELECT image_id, image_path FROM dress_dataset"
                cursor.execute(sql_images)
                self.data = cursor.fetchall()
                
                # 💡 步驟 C：撈取從表紀錄（即對應你 JOIN 查詢的左半部數據來源）
                sql_annotations = "SELECT image_id, attribute_name FROM image_attributes"
                cursor.execute(sql_annotations)
                all_annos = cursor.fetchall()
                
                # 將從表資料打包進記憶體快取 (Dictionary Map)
                self.img_attr_cache = {}
                for anno in all_annos:
                    img_id = anno['image_id']
                    attr = anno['attribute_name']
                    if img_id not in self.img_attr_cache:
                        self.img_attr_cache[img_id] = []
                    self.img_attr_cache[img_id].append(attr)
                    
                print(f"🎉 成功載入 {len(self.data)} 筆影像主表，並完成屬性快取對齊！")
                
        finally:
            connection.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img_id = row['image_id']
        
        # 1. 讀取圖片
        img_path = os.path.join(os.getcwd(), row['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 2. 初始化一個長度為 38 維、全為 0.0 的 Ground Truth 向量
        target_tensor = torch.zeros(self.num_attributes, dtype=torch.float32)
        
        # 3. 查表：如果這張照片有被標註標籤，就把對應的位置從 0.0 改成 1.0
        # 拿 0001 舉例：active_attrs 會拿到 ['乾淨', '簡約']
        active_attrs = self.img_attr_cache.get(img_id, [])
        for attr in active_attrs:
            if attr in self.attr_to_id:
                attr_id = self.attr_to_id[attr]
                target_tensor[attr_id] = 1.0 # 有選中的標籤填 1，其餘 36 個沒選中的會維持 0
                
        return image, target_tensor


# --- 獨立測試區塊 ---
if __name__ == "__main__":
    test_transform = transforms.Compose([
        transforms.Resize((512, 288)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("正在連線 5070 資料庫並驗證 38 維多標籤資料集...")
    dataset = DressAttributeDataset(transform=test_transform)
    
    # 盲測第 1 張照片 (如果是照順序排，應該就是 0001 號穿搭)
    if len(dataset) > 0:
        img, target = dataset[0]
        print("\n=== 🧪 Dataset 結構盲測驗證 ===")
        print(f"圖片 Tensor 形狀: {img.shape}")
        print(f"Ground Truth (GT) 向量維度: {target.shape} (預計要看到 38)")
        print(f"這張照片實際擁有（標註為 1）的特徵總數: {int(target.sum().item())} 筆")
        
        # 還原文字標籤，看看是不是印出 乾淨、簡約
        active_indices = (target == 1.0).nonzero(as_tuple=True)[0]
        active_names = [dataset.id_to_attr[idx.item()] for idx in active_indices]
        print(f"🔍 解析還原後的文字屬性為: {active_names}")
        print("--------------------------------")