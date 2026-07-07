import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import KFold

# 載入你剛才驗證成功的分類器元件
from style_classifier_dataset import DressStyleClassifierDataset
from style_classifier_model import DressStyleClassifierModel

def main():
    # --- 1. 基礎超參數設定 ---
    BATCH_SIZE = 32
    BACKBONE_LR = 1e-5     # 解凍後大腦特徵層的微弱學習率
    HEAD_LR = 1e-3         # 風格分類頭的標準學習率
    WEIGHT_DECAY = 1e-4    # 正則化防止過擬合
    
    # 💡 雙階段動態早停超參數 (a, b, c, d)
    PATIENCE_A = 7      # ⬅️ 參數 a: 凍結階段連續幾輪沒進步就停下來
    MAX_EPOCH_B = 40    # ⬅️ 參數 b: 凍結階段最多跑幾輪
    
    PATIENCE_C = 10     # ⬅️ 參數 c: 不凍結階段連續幾輪沒進步就停下來
    MAX_EPOCH_D = 40    # ⬅️ 參數 d: 不凍結階段最多跑幾輪
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 雙階段動態早停 Style Classifier K-fold 訓練啟動！設備: {device}")
    
    # --- LOG 檔案初始化 ---
    log_filename = "kfold_classifier_log.txt"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write("==================================================\n")
        log_file.write("   DressGPT Classifier 雙階段動態早停 K-Fold 紀錄  \n")
        log_file.write("==================================================\n")
        log_file.write(f"超參數設定: \n")
        log_file.write(f"- BATCH_SIZE: {BATCH_SIZE}\n")
        log_file.write(f"- Stage 1 (Lock): 容忍 {PATIENCE_A} 輪 (a), 上限 {MAX_EPOCH_B} 輪 (b)\n")
        log_file.write(f"- Stage 2 (Unlk): 容忍 {PATIENCE_C} 輪 (c), 上限 {MAX_EPOCH_D} 輪 (d)\n")
        log_file.write(f"- Backbone LR: {BACKBONE_LR}, Head LR: {HEAD_LR}\n\n")

    # --- 2. 準備數據供應鏈 ---
    train_transform = transforms.Compose([
        transforms.Resize((512, 288)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), 
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 288)),
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 初始化分類資料集（只會載入 551 筆黃金穿搭樣本）
    full_dataset = DressStyleClassifierDataset(transform=train_transform)
    total_count = len(full_dataset)
    num_classes = len(full_dataset.style_to_id)
    
    # 💡 關鍵安全步驟：將動態生成的 19 類對照表存為 JSON，供未來 Web 推論查表使用
    mapping_filename = "style_mapping.json"
    with open(mapping_filename, "w", encoding="utf-8") as f:
        json.dump(full_dataset.id_to_style, f, ensure_ascii=False, indent=4)
    print(f"💾 風格文字對照表已安全固化至: {mapping_filename}")
    
    K_SPLITS = 5
    kf = KFold(n_splits=K_SPLITS, shuffle=True, random_state=42)
    all_fold_val_loss = []
    all_fold_val_acc = []  
    
    print(f"🌀 開始執行 {K_SPLITS}-Fold 交叉驗證，紀錄將寫入 {log_filename}...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(total_count))):
        print(f"\n==================== 🔥 Fold [{fold + 1}/{K_SPLITS}] 開跑 ====================")
        
        with open(log_filename, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n--- Fold [{fold + 1}/{K_SPLITS}] 詳細風格訓練歷程 ---\n")
            log_file.write("Epoch │ Stage Tag      │ Train Loss (Acc)   │ Val Loss (Acc)\n")
            log_file.write("──────────────────────────────────────────────────────────────────────────\n")

        train_sub_dataset = Subset(full_dataset, train_idx)
        val_sub_dataset = Subset(full_dataset, val_idx)
        val_sub_dataset.dataset.transform = val_transform 
        
        train_loader = DataLoader(train_sub_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_sub_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        # 建立指定類別數的模型
        model = DressStyleClassifierModel(num_classes=num_classes).to(device)
        
        # 🚀 初始狀態：進入 Stage 1，鎖死大腦特徵層，只開分類頭
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.backbone.classifier[2].parameters():
            param.requires_grad = True
            
        optimizer = optim.AdamW(model.backbone.classifier[2].parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
        
        # 💡 切換為分類核心：交叉熵損失函數
        criterion = nn.CrossEntropyLoss()
        
        # ⚙️ 雙階段動態控制器狀態初始化
        stage = 1
        current_stage_epoch = 0
        global_epoch = 0
        best_val_loss = float('inf')  # 依舊看 Loss 最低點決定早停與進步與否
        patience_counter = 0
        
        # 用於終盤結帳的指標紀錄（對應到 Stage 2 交叉熵最低的瞬間）
        best_fold_loss = float('inf')
        best_fold_acc = 0.0
        
        # 第一階段最佳權重臨時暫存路徑（用於回滾）
        tmp_checkpoint_path = f"checkpoints/tmp_best_classifier_stage1_fold{fold+1}.pth"
        
        while True:
            global_epoch += 1
            current_stage_epoch += 1
                
            # --- 訓練階段 ---
            model.train()
            train_loss = 0.0
            train_corrects = 0
            
            for images, labels in train_loader:
                # 💡 注意：labels 不需要 .unsqueeze(1)，保持一維張量
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                
                # 計算答對的樣本數
                preds = torch.argmax(outputs, dim=1)
                train_corrects += torch.sum(preds == labels).item()
                
            epoch_train_loss = train_loss / len(train_loader.dataset)
            epoch_train_acc = train_corrects / len(train_loader.dataset)

            # --- 驗證階段 ---
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    
                    preds = torch.argmax(outputs, dim=1)
                    val_corrects += torch.sum(preds == labels).item()
                    
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = val_corrects / len(val_loader.dataset)
            
            stage_tag = f"[S1:Lock]" if stage == 1 else f"[S2:Unlk]"
            
            # 即時輸出列印（將比例轉成百分比顯示，更直覺）
            print(f"Epoch {global_epoch:02d} {stage_tag} │ Train Loss: {epoch_train_loss:.4f} (Acc: {epoch_train_acc*100:.2f}%) │ Val Loss: {epoch_val_loss:.4f} (Acc: {epoch_val_acc*100:.2f}%)")
            
            # 同步寫入日誌檔
            with open(log_filename, "a", encoding="utf-8") as log_file:
                log_file.write(f"Epoch {global_epoch:02d} │ {stage_tag:14s} │ Train: {epoch_train_loss:.4f} ({epoch_train_acc*100:.2f}%) │ Val Loss: {epoch_val_loss:.4f} ({epoch_val_acc*100:.2f}%)\n")
            
            # ==========================================
            # 📉 雙階段動態早停核心邏輯控制器
            # ==========================================
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                
                os.makedirs("checkpoints", exist_ok=True)
                if stage == 1:
                    # 階段一創紀錄：暫存臨時檔
                    torch.save(model.state_dict(), tmp_checkpoint_path)
                else:
                    # 階段二創紀錄：全域最終最佳風格分類器模型
                    torch.save(model.state_dict(), f"checkpoints/dress_classifier_fold{fold+1}_best.pth")
                    best_fold_loss = epoch_val_loss
                    best_fold_acc = epoch_val_acc
            else:
                patience_counter += 1
                
            # ---- 🛑 檢查 Stage 1 是否觸發跳轉條件 ----
            if stage == 1:
                if patience_counter >= PATIENCE_A or current_stage_epoch >= MAX_EPOCH_B:
                    print(f"➔ [系統提示] Stage 1 飽和 (連續 {patience_counter} 輪未進步或達上限 {MAX_EPOCH_B} 輪)。")
                    
                    if os.path.exists(tmp_checkpoint_path):
                        print("🔄 正在從暫存檔回滾至 Stage 1 最佳權重...")
                        model.load_state_dict(torch.load(tmp_checkpoint_path))
                    
                    print("🔓 已全面解凍大腦特徵層，重新宣告雙 LR 優化器，正式挺進 Stage 2 微調！")
                    
                    # 全面解凍大腦
                    for param in model.parameters():
                        param.requires_grad = True
                        
                    # 重新配置雙學習率優化器
                    optimizer = optim.AdamW([
                        {'params': model.backbone.features.parameters(), 'lr': BACKBONE_LR},
                        {'params': model.backbone.classifier[2].parameters(), 'lr': HEAD_LR}
                    ], weight_decay=WEIGHT_DECAY)
                    
                    stage = 2
                    current_stage_epoch = 0
                    best_val_loss = float('inf')
                    patience_counter = 0
                    
                    with open(log_filename, "a", encoding="utf-8") as log_file:
                        log_file.write(f"➔ [系統狀態切換] Stage 1 結束。載入最佳分類頭，全面解凍進入 Stage 2。\n")
                    continue

            # ---- 🏁 檢查 Stage 2 是否滿足最終終止條件 ----
            elif stage == 2:
                if patience_counter >= PATIENCE_C or current_stage_epoch >= MAX_EPOCH_D:
                    print(f"🏁 [系統提示] Stage 2 觸發最終早停 (連續 {patience_counter} 輪未進步或達上限 {MAX_EPOCH_D} 輪)。")
                    
                    if os.path.exists(tmp_checkpoint_path):
                        os.remove(tmp_checkpoint_path)
                        
                    with open(log_filename, "a", encoding="utf-8") as log_file:
                        log_file.write(f"➔ [系統訓練終止] Stage 2 觸發早停，本折（Fold）實驗在此安全收尾。\n")
                    break

        print(f"✨ Fold [{fold + 1}] 結束！最佳驗證集歸檔成績 -> Loss: {best_fold_loss:.4f}, Accuracy: {best_fold_acc*100:.2f}%")
        all_fold_val_loss.append(best_fold_loss)
        all_fold_val_acc.append(best_fold_acc)
        
        with open(log_filename, "a", encoding="utf-8") as log_file:
            log_file.write(f"➔ Fold [{fold + 1}] 最佳盲測表現總結 -> Val Loss 最低點對應之 Loss: {best_fold_loss:.4f} │ Accuracy: {best_fold_acc*100:.2f}%\n")
            log_file.write("──────────────────────────────────────────────────────────────────────────\n")

    # --- 4. 算總帳：寫入最終全域平均成績 ---
    mean_loss = np.mean(all_fold_val_loss)
    mean_acc = np.mean(all_fold_val_acc)
    
    summary_text = (
        f"\n==================== 📊 全域交叉驗證總結 ====================\n"
        f"各 Fold 最佳 Loss: {[f'{r:.4f}' for r in all_fold_val_loss]}\n"
        f"各 Fold 最佳 Accuracy: {[f'{r*100:.2f}%' for r in all_fold_val_acc]}\n"
        f"----------------------------------------------------------\n"
        f"🔥 5-Fold 平均驗證集 Loss: 【{mean_loss:.4f}】\n"
        f"🚀 5-Fold 平均驗證集 準確率(Accuracy): 【{mean_acc*100:.2f}%】\n"
        f"==========================================================\n"
    )
    
    print(summary_text)
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(summary_text)
        
    print(f"💾 完整分類器自動控制實驗數據已成功保存至 {log_filename}")

if __name__ == "__main__":
    main()