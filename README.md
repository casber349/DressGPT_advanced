# DressGPT_advanced (v2) 穿搭 AI 智慧評估與推薦系統

DressGPT_advanced v2 是一款將數據工程、深度學習與網頁應用完美閉環的端到端（End-to-End）穿搭智慧評估系統。系統以 **ConvNeXt** 卷積神經網路為特徵提取骨幹（Backbone），向上建構了**三大 AI 推論大腦**（評分、風格分類、多標籤屬性識別）。專案從最底層的 MySQL 數據清洗標註出發，經由 5-Fold 集成學習（Ensemble Learning）與軟性投票（Soft Voting）穩固推論精準度，最終將高維特徵解耦，於前端實現即時的穿搭品質評估、多維屬性分析與風格推薦榜樣。

---

## 🏗️ 系統三層式架構

本專案由下至上劃分為三個核心層次，確保數據流轉的流暢與解耦：

### 🛠️ 第 1 層：數據清洗與專業標註（Data Labeling Pipeline）
負責建立服飾影像的結構化監督訊號（Ground Truth），提供高品質的機器學習訓練數據。
* `construct_db.sql`: 負責在 MySQL 中初始化名為 `dress_gpt_advanced` 的資料庫，並建構所有關聯式表格結構（包含主風格清單、屬性清單、照片標註主表等）。
* `app_labeling.py` / `templates/labeling.html`: 評審專用標註網頁主控台。基於 Flask 框架開發（運行於 `Port 9527`），提供圖形化介面供專業人員快速對影像進行風格歸類、多維度屬性滑桿強度微調、正負向缺陷定義與綜合評分。
* `extend_dataset.py`: 數據集自動擴充腳本。用於將本地新增的待標註圖片路徑與基礎預設值批量 `INSERT` 到 MySQL 表格中，此腳本具備數據保護機制，**絕不覆蓋或更動已標註完成的珍貴歷史資料**。
* `dress_gpt_advanced_backup_1000.sql`: 資料庫備份檔。可用於開發環境的隨時復原。
    * **資料庫備份指令：**
        ```bash
        mysqldump -h 127.0.0.1 -u root -p dress_gpt_advanced > dress_gpt_advanced_backup_1000.sql
        ```
    * **資料庫復原指令：**
        ```bash
        mysql -h 127.0.0.1 -u root -p dress_gpt_advanced < dress_gpt_advanced_backup_1000.sql
        ```

### 🧠 第 2 層：深度學習核心模型（Deep Learning Core）
以強大的 **ConvNeXt (CNN)** 架構為 Backbone，針對各應用場景下游任務接上獨立的小型多層感知機（MLP）神經網路。所有大腦均採用 **5-Fold 交叉驗證（Cross Validation）** 來抵抗過擬合並提升泛化表現。

1.  **評分模型 (Scorer Brain)** `[5-Fold Ensemble]`
    * **代碼組件**：`scorer_dataset.py` / `scorer_model.py` / `train_scorer.py`
    * **數據與損失**：以資料庫全體圖片（1000張）為訓練集，採用 **MSE Loss** 進行連續數值回歸訓練。
    * **推論機制**：每個 Fold 模型獨立輸出 1 個 0.00~10.00 之間的實數。最終回傳的總體評分（`overall_score`）為 5 個 Fold 模型的預測平均值。
2.  **風格分類模型 (Style Classifier Brain)** `[5-Fold Soft Voting]`
    * **代碼組件**：`style_classifier_dataset.py` / `style_classifier_model.py` / `train_style_classifier.py`
    * **數據與損失**：過濾掉未達及格線的雜亂雜訊，僅選取資料庫中評分在 **5 分以上且擁有明確主風格** 的精華圖片（551張），採用 **Cross Entropy Loss** 進行多分類（19種風格）訓練。
    * **推論機制**：為了最佳化體驗，系統**僅在使用者穿搭獲得 4 分以上時觸發此大腦**。每個 Fold 模型輸出 19 維的 Logits，並在應用端透過 **Soft Voting（軟性投票，將各 Fold 機率取平均）** 決定結果。**平均機率最高者為「主風格」**，**次高者為「潛在風格」**。
3.  **屬性標註模型 (Attribute Labeler Brain)** `[5-Fold Logistic Regression]`
    * **代碼組件**：`attribute_labeler_dataset.py` / `attribute_labeler_model.py` / `train_attribute_labeler.py`
    * **數據與損失**：以全體圖片（1000張）為訓練集。針對目前主流的 38 種服飾屬性與特徵進行並行多標籤的 Logistic Regression（對抗非排他性標籤），並以 **Soft IoU Loss** 作為優化指標。
    * **關鍵雙輸出**：
        * **輸出 A**：照片所擁有的所有屬性預測機率。
        * **輸出 B**：提取出能代表該照片穿搭語意空間的 **64維重心特徵向量 (Embedding)**，用於後續去中心化即時推薦。

### 🌐 第 3 層：整合應用程式介面（Application API Layer）
實現用戶端的前後端完美閉環，提供直覺且高質感的互動介面。
* `generate_feature_database.py`: 離線知識庫特徵提煉腳本。將 MySQL 內的所有照片輸入 5-Fold 屬性標註模型中，將其輸出的 64 維特徵向量進行平均，最後打包匯出成一檔常駐記憶體的向量矩陣 `knowledge_base_64d.pt`，作為應用端計算餘弦相似度（Cosine Similarity）的**特徵知識庫**。
* `style_mapping.json` / `attribute_labeling.json`: 索引對照表。將模型預測輸出的數值 ID 瞬間轉譯為人類可讀的實際風格名稱與屬性標籤。
* `app.py` / `templates/index.html`: 使用者應用程式介面。基於 Flask 框架開發（運行於 `Port 9528`），使用者點擊 `evaluate` 即可一鍵觸發後台三大模型協同推論。

---

## 📊 深度學習大腦實驗與全域交叉驗證報告

以下為系統三大核心模型經過 5-Fold 交叉驗證後的最終真實訓練指標數據統計：

### 1. 評分模型 (Scorer)
* **各 Fold 最佳 RMSE**：`['1.2118', '1.2178', '1.1377', '1.1874', '1.1479']`
* **各 Fold 最佳 R² 分數**：`['0.5958', '0.6052', '0.7115', '0.6338', '0.6932']`
* **📊 5-Fold 平均驗證集 RMSE**：**`1.1805`**
* **📊 5-Fold 平均驗證集 R² 分數**：**`0.6479`**

### 2. 風格分類模型 (Style Classifier)
* **各 Fold 最佳 Loss**：`['1.3893', '1.1885', '1.1231', '1.0106', '1.3444']`
* **各 Fold 最佳 準確率(Accuracy)**：`['64.86%', '71.82%', '70.91%', '74.55%', '67.27%']`
* **📊 5-Fold 平均驗證集 Loss**：**`1.2112`**
* **📊 5-Fold 平均驗證集 準確率(Accuracy)**：**`69.88%`**

### 3. 屬性標註模型 (Attribute Labeler)
* **各 Fold 最佳 Loss (1 - Soft_IoU)**：`['0.6972', '0.7075', '0.6673', '0.6861', '0.7022']`
* **各 Fold 最佳 Mean IoU**：`['32.13%', '30.27%', '35.24%', '34.46%', '31.85%']`
* **📊 5-Fold 平均驗證集 Loss**：**`0.6921`**
* **📊 5-Fold 平均驗證集 屬性重疊度(Mean IoU)**：**`32.79%`**

---

## ⚡ 快速啟動指南

### 1. 環境配置
請確保本地或虛擬環境中已安裝 PyTorch、Flask 以及相關資料庫連接庫：
```bash
pip install -r reequirements.txt