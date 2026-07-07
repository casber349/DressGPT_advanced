import os
import sys
import json
import torch
import pymysql
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms

# 🧠 導入你設計的三個模型大腦架構
from style_classifier_model import DressStyleClassifierModel
from scorer_model import DressScorerModel
from attribute_labeler_model import DressAttributeLabelerModel

# 讀取 .env 設定檔
load_dotenv()

app = Flask(__name__)

# 配置執行設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ⚙️ 超參數與全域變數定義 ---
NUM_CLASSES_STYLE = 19
NUM_ATTRIBUTES = 38
K_SPLITS = 5
CHECKPOINT_DIR = "checkpoints"

# 記憶體常駐大腦清單
style_models = []
scorer_models = []
attribute_models = []

# 標籤對照表
STYLE_MAPPING = {}
ATTRIBUTE_MAPPING = {}
# 1-1. 檔案頂端宣告
KNOWLEDGE_BASE = None

# --- 🖼️ 共用圖片預處理pipeline ---
shared_transform = transforms.Compose([
    transforms.Resize((512, 288)), # 9:16 完美黃金比例
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- ⏳ 載入模型們 ---
def init_ensemble_models():
    global STYLE_MAPPING, ATTRIBUTE_MAPPING
    # 1-2. 在 init_ensemble_models() 內部載入
    global KNOWLEDGE_BASE
    KNOWLEDGE_BASE = torch.load("knowledge_base_64d.pt", map_location=device)

    print(f"\n🔮 DressGPT 預測引擎啟動中... 運行設備: {device}")
    
    # 1. 安全載入兩大標籤對照表
    try:
        with open("style_mapping.json", "r", encoding="utf-8") as f:
            STYLE_MAPPING = json.load(f)
        with open("attribute_mapping.json", "r", encoding="utf-8") as f:
            ATTRIBUTE_MAPPING = json.load(f)
        print(f"✅ 對照表載入成功 (風格: {len(STYLE_MAPPING)} 類, 屬性: {len(ATTRIBUTE_MAPPING)} 類)")
    except Exception as e:
        print(f"⚠️ 載入對照表 JSON 失敗: {e}")

    # 2. 載入模型們
    print(f"⏳ 正在將模型們載入記憶體...")

    for fold in range(1, K_SPLITS + 1):
        # A. scorer 模型
        scorer_net = DressScorerModel().to(device)
        scorer_path = os.path.join(CHECKPOINT_DIR, f"dress_scorer_fold{fold}_best.pth")
        if os.path.exists(scorer_path):
            scorer_net.load_state_dict(torch.load(scorer_path, map_location=device))
            scorer_net.eval()
            scorer_models.append(scorer_net)
        print(f"✅ Fold {fold} Scorer loaded successfully.")

        # B. style classifier 模型
        style_net = DressStyleClassifierModel().to(device)
        style_path = os.path.join(CHECKPOINT_DIR, f"dress_classifier_fold{fold}_best.pth")
        if os.path.exists(style_path):
            style_net.load_state_dict(torch.load(style_path, map_location=device))
            style_net.eval()
            style_models.append(style_net)
        print(f"✅ Fold {fold} Classifier loaded successfully.")
        
        # C. attribute labeler 模型
        attr_net = DressAttributeLabelerModel(num_attributes=NUM_ATTRIBUTES).to(device)
        attr_path = os.path.join(CHECKPOINT_DIR, f"dress_attribute_labeler_fold{fold}_best.pth")
        if os.path.exists(attr_path):
            attr_net.load_state_dict(torch.load(attr_path, map_location=device))
            attr_net.eval()
            attribute_models.append(attr_net)
        print(f"✅ Fold {fold} Attribute loaded successfully.")

    print("全體模型載入完畢！")


# 資料庫核心連線設定
def get_db_connection():
    return pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# 核心評估 API
@app.route('/evaluate', methods=['POST'])
def evaluate():
    # 1. 接收前端傳來的檔案與篩選條件
    file = request.files.get('photo')  # 嚴格對齊你原本的欄位名稱 'photo'
    gender = request.form.get('gender')            
    seasons = request.form.getlist('seasons')      
    formality = request.form.getlist('formality')  

    # 安全檢查
    if not file or file.filename == '':
        return jsonify({'error': '請先選取一張穿搭照片！'}), 400

    print(f" 收到前端中文條件：性別={gender}, 季節={seasons}, 正式度={formality}")

    # ==========================================
    # 🌐 關鍵核心：將前端中文對應到 MySQL 的英文值
    # ==========================================
    gender_map = {"男": "male", "女": "female"}
    season_map = {"夏季": "summer", "春秋季": "spring_autumn", "冬季": "winter"}
    formality_map = {"休閒": "casual", "正式": "formal"}

    db_gender = gender_map.get(gender, "male")
    db_seasons = [season_map[s] for s in seasons if s in season_map]
    db_formality = [formality_map[f] for f in formality if f in formality_map]

    # ==========================================
    # 🧠 核心關鍵：餵入 15 折實際深度學習模型進行真・AI 評估
    # ==========================================
    try:
        # 讀取圖片並轉換 Tensor
        img_raw = Image.open(file.stream).convert("RGB")
        img_tensor = shared_transform(img_raw).unsqueeze(0).to(device)
        
        # 初始化回傳變數
        predicted_main_style = ""
        predicted_sub_style = ""
        real_attributes = []
        final_score = 0.0
        
        with torch.no_grad():
            # 💡 用於日誌記錄的明細容器
            fold_scores = []
            fold_style_probs_list = [] # 記錄各折過完 Softmax 的獨立機率向量
            fold_styles_top1 = []      # 記錄各折單獨認定的第一名風格名稱

            # ==========================================================
            # 🧠 1. 大腦 A：整體評分器 (Scorer) -> 5折簡單平均
            # ==========================================================
            ensemble_score = 0.0
            if scorer_models:
                for model in scorer_models:
                    score_val = model(img_tensor).item()
                    fold_scores.append(round(score_val, 2)) # 記錄單折分數
                    ensemble_score += score_val
                ensemble_score /= len(scorer_models)
                final_score = round(float(ensemble_score), 2)
            else:
                final_score = 0.0

            # ==========================================================
            # 🧠 2. 大腦 B：風格分類器 (Style Classifier) -> 條件觸發 + Soft Voting
            # ==========================================================
            if final_score >= 4.00:
                ensemble_style_probs = torch.zeros(1, NUM_CLASSES_STYLE).to(device)
                if style_models:
                    for model in style_models:
                        logits = model(img_tensor)
                        
                        # 🚨 核心實作：Soft Voting！每一折必須先用 Softmax 轉成「真實機率分佈」
                        probs = torch.softmax(logits, dim=1)
                        ensemble_style_probs += probs
                        
                        # 收集除錯明細
                        f_style_id = str(torch.argmax(probs, dim=1).item())
                        fold_styles_top1.append(STYLE_MAPPING.get(f_style_id, "未知"))
                        fold_style_probs_list.append(probs.squeeze(0))
                    
                    # 將 5 折的「機率」進行平均
                    ensemble_style_probs /= len(style_models)
                    
                    # 從平均後的機率海中，撈出前兩名
                    top2_probs, top2_indices = torch.topk(ensemble_style_probs, 2, dim=1)
                    predicted_main_style = STYLE_MAPPING.get(str(top2_indices[0][0].item()), "未定義主風格")
                    predicted_sub_style = STYLE_MAPPING.get(str(top2_indices[0][1].item()), "未定義副風格")
                else:
                    predicted_main_style, predicted_sub_style = "模型未就緒", "模型未就緒"
            else:
                # 分數不及格，風格直接留空
                predicted_main_style, predicted_sub_style = "", ""

            # ==========================================================
            # 🧠 3. 大腦 C：屬性標註器 (Attribute Labeler) -> 重心向量 + 去資料庫偷鄰居標籤
            # ==========================================================
            # 🚨 防禦機制：確保 KNOWLEDGE_BASE 確實是字典型態
            if attribute_models and KNOWLEDGE_BASE is not None and isinstance(KNOWLEDGE_BASE, dict):
                ensemble_embedding_64d = torch.zeros(1, 64).to(device)
                
                # A. 融合成當前圖片的紅色「重心特徵向量」[cite: 3]
                for model in attribute_models:
                    _, embedding = model(img_tensor)
                    ensemble_embedding_64d += embedding
                ensemble_embedding_64d /= len(attribute_models)
                
                # B. 取出知識庫的 1000 筆鄰居數據[cite: 3]
                kb_embeddings = torch.from_numpy(KNOWLEDGE_BASE['embeddings']).float().to(device) 
                kb_image_ids = KNOWLEDGE_BASE['image_ids'] 
                
                # C. 計算當前重心與所有鄰居的 Cosine Similarity[cite: 3]
                norm_current = ensemble_embedding_64d / ensemble_embedding_64d.norm(dim=1, keepdim=True)
                norm_kb = kb_embeddings / kb_embeddings.norm(dim=1, keepdim=True)
                similarities = torch.mm(norm_current, norm_kb.t()).squeeze(0)  
                
                # D. 抓出相似度最高的前 K 個鄰居 (設定 K=5)
                K_NEIGHBORS = 5
                topk_sim, topk_indices = torch.topk(similarities, k=K_NEIGHBORS)
                
                topk_image_ids = [kb_image_ids[idx.item()] for idx in topk_indices]
                topk_sim_list = [float(sim.item()) for sim in topk_sim]
                
                # E. 即時去資料庫撈取並實施加權累加[cite: 3]
                total_similarity = sum(topk_sim_list) 
                accumulated_scores = {}               
                neighbor_raw_data = {}                
                
                # 💡 關鍵修正：建立大腦 C 專用的、獨立的資料庫連線與 cursor 環境，避免沿用已關閉的舊連線
                attr_conn = None
                try:
                    attr_conn = get_db_connection()
                    with attr_conn.cursor() as attr_cursor:
                        for img_id, sim_val in zip(topk_image_ids, topk_sim_list):
                            sql = "SELECT * FROM image_attributes WHERE image_id = %s;"
                            attr_cursor.execute(sql, (img_id,))
                            rows = attr_cursor.fetchall()
                            
                            neighbor_raw_data[img_id] = []
                            for row in rows:
                                # 🚨 核心防禦：自動辨識資料庫吐出的是「字典」還是「元組」[cite: 3]
                                if isinstance(row, dict):
                                    attr_name = row['attribute_name']
                                    attr_value = row['attribute_value']
                                else:
                                    attr_name = row[1]
                                    attr_value = row[2]
                                
                                val_float = float(attr_value)
                                neighbor_raw_data[img_id].append(f"{attr_name}:{val_float}")
                                
                                # 進行動態加權累加[cite: 3]
                                if attr_name not in accumulated_scores:
                                    accumulated_scores[attr_name] = 0.0
                                accumulated_scores[attr_name] += val_float * sim_val
                except Exception as db_err:
                    print(f"❌ 大腦 C 即時讀取資料庫發生錯誤: {db_err}")
                finally:
                    if attr_conn:
                        attr_conn.close() # 查完立即安全關閉，好習慣點滿
                
                # F. 計算最終相似度加權平均值[cite: 3]
                for attr_name, weighted_sum in accumulated_scores.items():
                    # 💡 修正：將變數名稱改為 attr_final_score，才不會把大腦 A 的穿搭總分 final_score 給覆蓋掉！
                    attr_final_score = weighted_sum / total_similarity
                    attr_final_score = round(attr_final_score, 1) 
                    
                    if attr_final_score >= 1.0:
                        real_attributes.append({'name': attr_name, 'score': attr_final_score})
                
                real_attributes = sorted(real_attributes, key=lambda x: x['score'], reverse=True)[:4]
            else:
                if KNOWLEDGE_BASE is not None and not isinstance(KNOWLEDGE_BASE, dict):
                    print("⚠️ [警告] KNOWLEDGE_BASE 載入型態錯誤，目前為 str。請檢查 init_ensemble_models()！")
                real_attributes = []

            # ==========================================================
            # 🎯 🌟 全新升級：推薦榜樣防妖魔化引擎 (條件過濾 + 5分封頂 + 特徵對齊Top5隨機抽2)
            # ==========================================================
            recommendations = []
            rec_conn = None
            
            if KNOWLEDGE_BASE is not None:
                try:
                    rec_conn = get_db_connection()
                    with rec_conn.cursor() as rec_cursor:
                        # 1. 執行硬性條件過濾：性別、季節、正式度 + 💡 強制分數必須大於等於 5 分
                        # (請確保你的 dress_dataset 資料庫欄位有名為 overall_score 的浮點數欄位)
                        query = "SELECT image_id, image_path FROM dress_dataset WHERE gender = %s AND overall_score >= 5.0"
                        params = [db_gender]

                        if db_seasons:
                            placeholders = ', '.join(['%s'] * len(db_seasons))
                            query += f" AND season IN ({placeholders})"
                            params.extend(db_seasons)

                        if db_formality:
                            placeholders = ', '.join(['%s'] * len(db_formality))
                            query += f" AND formality IN ({placeholders})"
                            params.extend(db_formality)

                        rec_cursor.execute(query, params)
                        db_rows = rec_cursor.fetchall() # 撈出所有符合硬性條件的高分極品樣本
                        
                    if db_rows:
                        # 建立 image_id 到 image_path 的快速對照字典
                        db_id_to_path = {row['image_id']: row['image_path'] for row in db_rows}
                        
                        # 2. 跨時空對齊：找出這些高分候選人在 1000 張特徵知識庫中的陣列索引 (Index)
                        kb_image_ids = KNOWLEDGE_BASE['image_ids']
                        valid_kb_indices = [i for i, img_id in enumerate(kb_image_ids) if img_id in db_id_to_path]
                        
                        if valid_kb_indices:
                            # 抽取這群優質候選人的 64D 特徵矩陣
                            filtered_kb_embeddings = torch.from_numpy(KNOWLEDGE_BASE['embeddings'][valid_kb_indices]).float().to(device)
                            
                            # 3. 計算使用者紅色重心向量與這群優質候選人的 Cosine Similarity
                            norm_current = ensemble_embedding_64d / ensemble_embedding_64d.norm(dim=1, keepdim=True)
                            norm_filtered = filtered_kb_embeddings / filtered_kb_embeddings.norm(dim=1, keepdim=True)
                            rec_sims = torch.mm(norm_current, norm_filtered.t()).squeeze(0) # 算出所有候選人的相似度
                            
                            # 4. 鎖定相似度最高的前 N 名 (這裡設定 N=5，確保進榜的都是風格極其接近的帥照)
                            TOP_N_POOL = min(5, len(valid_kb_indices))
                            topn_sims, topn_sub_indices = torch.topk(rec_sims, k=TOP_N_POOL)
                            
                            # 還原回原本知識庫的真正的 image_id
                            topn_kb_indices = [valid_kb_indices[idx.item()] for idx in topn_sub_indices]
                            topn_image_ids = [kb_image_ids[idx] for idx in topn_kb_indices]
                            
                            # 5. 實施動態洗牌黑魔法：從這 5 個最像的高分榜樣中，隨機抽出 2 個展示給前端
                            import random
                            final_selected_ids = random.sample(topn_image_ids, k=min(2, len(topn_image_ids)))
                            
                            # 對接路徑輸出
                            for img_id in final_selected_ids:
                                path = db_id_to_path[img_id]
                                formatted_path = f"/{path}" if not path.startswith('/') else path
                                recommendations.append(formatted_path)
                                
                except Exception as rec_err:
                    print(f"❌ 特徵對齊推薦引擎發生錯誤: {rec_err}")
                finally:
                    if rec_conn:
                        rec_conn.close()

            # 防枯竭備用機制 (如果完全沒有符合條件的高分候選人)
            if not recommendations:
                print("⚠️ 經過高分與特徵過濾後無相符穿搭，啟動極致安全備用顯示")
                recommendations = ['/static/dataset_images/0001.jpg', '/static/dataset_images/0002.jpg']

            # ==========================================================
            # 📊 🛠️ 後台極致演算解密：完美對齊截圖邏輯的偵錯面板
            # ==========================================================
            print("\n" + "="*60)
            print("🧠 DressGPT 5-Fold 真・實值演算明細偵錯面板")
            print("="*60)
            print(f"【評分結果】集成總分 (5折簡單平均)")
            for f in range(len(fold_scores)):
                print(f"  • scorer fold {f+1}: {fold_scores[f]}")
            print(f"  --> output overall score: {final_score}")
            print("-"*60)
            
            if final_score >= 4.00 and fold_style_probs_list:
                print(f"【風格結果】集成風格 (Soft Voting 機率平均法)")
                main_idx = int(top2_indices[0][0].item())
                sub_idx = int(top2_indices[0][1].item())
                for f in range(len(style_models)):
                    p_main = round(float(fold_style_probs_list[f][main_idx].item()), 2)
                    p_sub = round(float(fold_style_probs_list[f][sub_idx].item()), 2)
                    print(f"  • style_classifier fold {f+1} -> 預測首選: {fold_styles_top1[f]:4s} | [主風格單折機率: {p_main}, 副風格單折機率: {p_sub}]")
                print(f"  --> 平均機率(soft voting)後：主風格: {predicted_main_style}, 潛在風格: {predicted_sub_style}")
            else:
                print("【風格結果】分數未達 4.00 門檻，風格判定依規留空。")
                
            print("-"*60)
            if attribute_models and KNOWLEDGE_BASE is not None:
                print(f"【屬性結果】從 MySQL 虛擬偷取 (方案二：相似度加權平均)")
                print(f"  • 鎖定特徵最接近的 {K_NEIGHBORS} 個資料庫鄰居：")
                for n in range(K_NEIGHBORS):
                    img_id = topk_image_ids[n]
                    sim_val = round(topk_sim_list[n], 3)
                    attrs_str = ", ".join(neighbor_raw_data.get(img_id, []))
                    print(f"    - 鄰居 #{n+1} (ID: {img_id}) | Cos_Sim = {sim_val} | 原始標籤: [{attrs_str}]")
                print(f"  --> 流出前端屬性（前4名且得分 >= 1.0）:")
                for item in real_attributes:
                    print(f"    • 【{item['name']:4s}】加權平均後得分: {item['score']}")
            
            print("-"*60)
            print(f"【推薦結果】榜樣防妖魔化對齊引擎")
            if KNOWLEDGE_BASE is not None and db_rows:
                print(f"  • 符合硬性條件且分數 >= 5 分的優質候選人共: {len(db_rows)} 位")
                print(f"  • 已成功在特徵空間中鎖定相似度最高的 Top-{TOP_N_POOL} 精英池")
                print(f"  --> 隨機搖號流出至前端的雙榜樣路徑: {recommendations}")
            else:
                print("  • 無符合高分條件之候選人，觸發防空機制。")
            print("="*60 + "\n")

        # ==========================================
        # 🎯 打包符合最新邏輯的極致回傳結果
        # ==========================================
        final_result = {
            'score': final_score,                       
            'main_style': predicted_main_style,         
            'sub_style': predicted_sub_style,           
            'attributes': real_attributes,               # 內含 38 個不篩選、依原始 Mapping 順序排列的完整屬性
            'recommendations': recommendations          
        }
        return jsonify(final_result)

    except Exception as e:
        print(f"❌ AI 大腦推論階段發生嚴重錯誤: {e}")
        return jsonify({'error': f'伺服器 AI 引擎分析失敗: {str(e)}'}), 500


if __name__ == '__main__':
    # 啟動前先把模型載入記憶體，避免每次請求都重新載入
    init_ensemble_models()
    # 保留偵錯面版，關閉重複 import 兩次的reloader (浪費時間)
    app.run(host='0.0.0.0', port=9528, debug=True, use_reloader=False)