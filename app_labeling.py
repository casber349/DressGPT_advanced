import os
import pymysql
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()  # 載入 .env 檔案中的環境變數
# 資料庫連線配置
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

@app.route("/")
def index():
    return render_template("labeling.html")

# 安全的 API 裝飾器：確保任何後端報錯都只回傳 JSON，絕不吐出 HTML 網頁
@app.errorhandler(Exception)
def handle_exception(e):
    print(f"❌ 後端捕獲嚴重錯誤: {str(e)}")
    return jsonify({
        "status": "fail", 
        "message": f"伺服器內部出錯: {str(e)}"
    }), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204  # 204 代表 No Content，優雅地叫瀏覽器閉嘴

# API: 讀取指定 ID 的標註資料（新增 gender, season, formality 讀取）
@app.route("/api/get_image", methods=["POST"])
def get_image():
    data = request.json or {}
    image_id = data.get("image_id", "0001")
    
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 1. 撈取主資料（移除 is_pass，加入三個新維度）
            cursor.execute(
                """
                SELECT image_id, gender, season, formality, image_path, overall_score, main_style 
                FROM dress_dataset 
                WHERE image_id = %s;
                """, 
                (image_id,)
            )
            main_row = cursor.fetchone()
            
            if not main_row:
                return jsonify({
                    "status": "fail", 
                    "message": f"找不到圖片 ID: {image_id}"
                })
            
            # 2. 撈取特徵屬性（此處維持原樣一對多讀取即可）
            cursor.execute(
                "SELECT attribute_name, attribute_value FROM image_attributes WHERE image_id = %s;", 
                (image_id,)
            )
            attr_rows = cursor.fetchall()
            
            image_data = {
                "image_id": main_row["image_id"],
                "gender": main_row["gender"],
                "season": main_row["season"],
                "formality": main_row["formality"],
                "image_path": main_row["image_path"] if main_row["image_path"] else f"static/dataset_images/{image_id}.jpg",
                "overall_score": float(main_row["overall_score"]),
                "main_style": main_row["main_style"] if main_row["main_style"] else "",
                "attributes": {row["attribute_name"]: float(row["attribute_value"]) for row in attr_rows}
            }
            return jsonify({"status": "success", "data": image_data})
    finally:
        connection.close()

# API: 儲存當前標註結果（新增 gender, season, formality 寫入與更新）
@app.route("/api/save_label", methods=["POST"])
def save_label():
    data = request.json or {}
    image_id = data.get("image_id")
    if not image_id:
        return jsonify({"status": "fail", "message": "遺失關鍵變數 image_id"}), 400
        
    # 接收前端新維度資料，給予安全預設值
    gender = data.get("gender", "male")
    season = data.get("season", "spring_autumn")
    formality = data.get("formality", "casual")
    overall_score = data.get("overall_score", 5.0)  # 新算法完美平衡點預設為 5.0
    main_style = data.get("main_style", "")
    attributes = data.get("attributes", {})
    
    image_path = f"static/dataset_images/{image_id}.jpg"
    
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 1. 寫入或更新主表（欄位全面對應新一對多主表）
            upsert_main_sql = """
                INSERT INTO dress_dataset (image_id, gender, season, formality, image_path, overall_score, main_style)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    gender = VALUES(gender),
                    season = VALUES(season),
                    formality = VALUES(formality),
                    image_path = VALUES(image_path),
                    overall_score = VALUES(overall_score), 
                    main_style = VALUES(main_style);
            """
            cursor.execute(upsert_main_sql, (
                image_id, 
                gender,
                season,
                formality,
                image_path, 
                overall_score, 
                main_style if main_style else None
            ))
            
            # 2. 清理舊特徵並寫入新特徵（維持原有關聯表操作，安全乾淨）
            cursor.execute("DELETE FROM image_attributes WHERE image_id = %s;", (image_id,))
            if attributes:
                insert_attr_sql = "INSERT INTO image_attributes (image_id, attribute_name, attribute_value) VALUES (%s, %s, %s);"
                for name, val in attributes.items():
                    cursor.execute(insert_attr_sql, (image_id, name, val))
                    
            connection.commit()
            return jsonify({"status": "success", "message": f"ID {image_id} 標註保存完畢"})
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        connection.close()

# 載入現有所有風格與屬性選項（將 is_pass 換成 is_positive，回傳欄位改為正負向名稱）
@app.route("/api/get_options", methods=["GET"])
def get_options():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 撈取主風格
            cursor.execute("SELECT style_name FROM label_styles;")
            styles = [row["style_name"] for row in cursor.fetchall()]
            
            # 撈取屬性（將舊 extends 的 is_pass 換成新結構的 is_positive）
            cursor.execute("SELECT attr_name, is_positive FROM label_attributes;")
            attr_rows = cursor.fetchall()
            
            positive_attrs = [r["attr_name"] for r in attr_rows if r["is_positive"] == 1]
            negative_attrs = [r["attr_name"] for r in attr_rows if r["is_positive"] == 0]
            
            return jsonify({
                "status": "success",
                "styles": styles,
                "positive_options": positive_attrs,
                "negative_options": negative_attrs
            })
    finally:
        connection.close()

# 動態新增自訂風格或屬性到資料庫中
@app.route("/api/add_custom_option", methods=["POST"])
def add_custom_option():
    data = request.json or {}
    opt_type = data.get("type") # "style", "positive_attr", "negative_attr"
    value = data.get("value", "").strip()
    
    if not value:
        return jsonify({"status": "fail", "message": "內容不能為空白"}), 400
        
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            if opt_type == "style":
                sql = "INSERT IGNORE INTO label_styles (style_name) VALUES (%s);"
                cursor.execute(sql, (value,))
            elif opt_type in ["positive_attr", "pass_attr"]: # 兼容舊前端寫法安全過渡
                sql = "INSERT IGNORE INTO label_attributes (attr_name, is_positive) VALUES (%s, 1);"
                cursor.execute(sql, (value,))
            elif opt_type in ["negative_attr", "fail_attr"]:
                sql = "INSERT IGNORE INTO label_attributes (attr_name, is_positive) VALUES (%s, 0);"
                cursor.execute(sql, (value,))
            else:
                return jsonify({"status": "fail", "message": "未知的類型"}), 400
                
            connection.commit()
            return jsonify({"status": "success", "message": f"成功新增自訂標籤: {value}"})
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        connection.close()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9527, debug=True, use_reloader=False) # 保持你原本的連線設定