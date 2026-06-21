import pymysql
import os
from dotenv import load_dotenv

load_dotenv()  # 載入 .env 檔案中的環境變數

# 1. 建立連線
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
        # 2. 指定你的圖片資料夾路徑
        image_dir = 'static/dataset_images'
        
        # 檢查資料夾是否存在
        if not os.path.exists(image_dir):
            print(f"錯誤：找不到資料夾 {image_dir}")
            exit()
            
        # 撈出資料夾內所有的檔案名稱
        all_files = os.listdir(image_dir)
        
        # 過濾副檔名，確保只抓取 .jpg 檔案
        image_files = [f for f in all_files if f.lower().endswith('.jpg')]
        
        print(f"正在掃描資料夾... 偵測到 {len(image_files)} 張 JPG 照片。")
        
        # 3. 核心 SQL 語句：使用 INSERT IGNORE
        # 【全新修正】完全對齊新版 Schema：移除 is_pass，補上 gender, season, formality 欄位與預設初始值
        insert_sql = """
        INSERT IGNORE INTO dress_dataset (
            image_id, 
            gender, 
            season, 
            formality, 
            image_path, 
            overall_score, 
            main_style
        ) VALUES (
            %s, 
            'male', 
            'spring_autumn', 
            'casual', 
            %s, 
            0.00, 
            NULL
        );
        """
        
        new_items_count = 0
        
        # 4. 排序檔名，依序（0001, 0002...）檢查並寫入
        for file_name in sorted(image_files):
            # 檔名是 '0001.jpg' -> 切出 '0001' 當作 image_id
            image_id = os.path.splitext(file_name)[0]
            
            # 依據你的需求，路徑存入 'static/dataset_images/0001.jpg'
            relative_path = f"{image_dir}/{file_name}"
            
            # 執行插入（這裡的變數剛好依序對應 SQL 語句中的兩個 %s占位符：image_id 與 relative_path）
            cursor.execute(insert_sql, (image_id, relative_path))
            
            # cursor.rowcount 會回傳實際受影響的列數
            # 如果被 IGNORE 跳過，rowcount 會是 0；如果是新填入的空位，會是 1
            if cursor.rowcount > 0:
                new_items_count += 1
                print(f"成功新增未標註空位：{image_id}")
                
        # 5. 提交交易
        connection.commit()
        print(f"\n增量匯入完成！成功新增了 {new_items_count} 筆新圖片空位，其餘既有標註資料完好如初。")

finally:
    connection.close()