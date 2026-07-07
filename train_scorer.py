import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from scorer_dataset import DressScorerDataset
from scorer_model import DressScorerModel

def main():
    # --- 1. 基礎超參數設定 ---
    BATCH_SIZE = 32
    BACKBONE_LR = 1e-5     
    HEAD_LR = 1e-3         
    WEIGHT_DECAY = 1e-4    
    
    # 💡 雙階段動態早停超參數 (a, b, c, d)
    PATIENCE_A = 7      # ⬅️ 參數 a: 凍結階段連續幾輪沒進步就停下來
    MAX_EPOCH_B = 40    # ⬅️ 參數 b: 凍結階段最多跑幾輪
    
    PATIENCE_C = 10     # ⬅️ 參數 c: 不凍結階段連續幾輪沒進步就停下來
    MAX_EPOCH_D = 40    # ⬅️ 參數 d: 不凍結階段最多跑幾輪
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 雙階段動態早停 K-fold 訓練啟動！設備: {device}")

    # --- LOG 檔案初始化 ---
    log_filename = "kfold_scorer_log.txt"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write("==================================================\n")
        log_file.write("      DressGPT Scorer 雙階段動態早停 K-Fold 紀錄    \n")
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
    
    full_dataset = DressScorerDataset(transform=train_transform)
    total_count = len(full_dataset)
    
    K_SPLITS = 5
    kf = KFold(n_splits=K_SPLITS, shuffle=True, random_state=42)
    all_fold_val_r2 = []
    all_fold_val_rmse = []  
    
    print(f"🌀 開始執行 {K_SPLITS}-Fold 交叉驗證，紀錄將寫入 {log_filename}...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(total_count))):
        print(f"\n==================== 🔥 Fold [{fold + 1}/{K_SPLITS}] 開跑 ====================")
        
        with open(log_filename, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n--- Fold [{fold + 1}/{K_SPLITS}] 詳細訓練歷程 ---\n")
            log_file.write("Epoch │ Stage Tag      │ Train RMSE (R²) │ Val Loss (RMSE, R²)\n")
            log_file.write("──────────────────────────────────────────────────────────────────────────\n")

        train_sub_dataset = Subset(full_dataset, train_idx)
        val_sub_dataset = Subset(full_dataset, val_idx)
        val_sub_dataset.dataset.transform = val_transform 
        
        train_loader = DataLoader(train_sub_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_sub_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        # 建立模型
        model = DressScorerModel().to(device)
        
        # 🚀 初始狀態：進入 Stage 1，鎖死大腦特徵層，只開拉桿
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.backbone.classifier[2].parameters():
            param.requires_grad = True
            
        optimizer = optim.AdamW(model.backbone.classifier[2].parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()
        
        # ⚙️ 雙階段動態控制器狀態初始化
        stage = 1
        current_stage_epoch = 0
        global_epoch = 0
        best_val_loss = float('inf')  # 依據你的指令：看 loss function 決定進步與否
        patience_counter = 0
        
        # 用於終盤總結的指標紀錄（會對應到 Stage 2 的最低 Loss 瞬間）
        best_fold_r2 = -float('inf')
        best_fold_rmse = float('inf')
        
        # 第一階段最佳權重臨時暫存路徑（用於回滾）
        tmp_checkpoint_path = f"checkpoints/tmp_best_stage1_fold{fold+1}.pth"
        
        # 使用 while True 控制動態彈性的 Epoch 總量
        while True:
            global_epoch += 1
            current_stage_epoch += 1
                
            # --- 訓練階段 ---
            model.train()
            train_loss = 0.0
            all_train_preds, all_train_labels = [], []
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                all_train_preds.extend(outputs.detach().cpu().numpy().flatten())
                all_train_labels.extend(labels.cpu().numpy().flatten())
                
            epoch_train_mse = train_loss / len(train_loader.dataset)
            epoch_train_rmse = np.sqrt(epoch_train_mse)
            epoch_train_r2 = r2_score(all_train_labels, all_train_preds)

            # --- 驗證階段 ---
            model.eval()
            val_loss = 0.0
            all_val_preds, all_val_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).unsqueeze(1)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    all_val_preds.extend(outputs.cpu().numpy().flatten())
                    all_val_labels.extend(labels.cpu().numpy().flatten())
                    
            # 💡 核心指標：驗證集 Loss (MSELoss)
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_rmse = np.sqrt(epoch_val_loss)
            epoch_val_r2 = r2_score(all_val_labels, all_val_preds)
            
            # 格式化標籤方便排版
            stage_tag = f"[S1:Lock]" if stage == 1 else f"[S2:Unlk]"
            
            # 終端機即時列印
            print(f"Epoch {global_epoch:02d} {stage_tag} │ Train RMSE: {epoch_train_rmse:.4f} (R²: {epoch_train_r2:+.4f}) │ Val Loss: {epoch_val_loss:.4f} (RMSE: {epoch_val_rmse:.4f}, R²: {epoch_val_r2:+.4f})")
            
            # 同步寫入檔案
            with open(log_filename, "a", encoding="utf-8") as log_file:
                log_file.write(f"Epoch {global_epoch:02d} │ {stage_tag:14s} │ Train: {epoch_train_rmse:.4f} ({epoch_train_r2:+.4f}) │ Val Loss: {epoch_val_loss:.4f} ({epoch_val_rmse:.4f}, {epoch_val_r2:+.4f})\n")
            
            # ==========================================
            # 📉 雙階段動態早停核心邏輯控制器
            # ==========================================
            if epoch_val_loss < best_val_loss:
                # 創紀錄，重置耐性
                best_val_loss = epoch_val_loss
                patience_counter = 0
                
                os.makedirs("checkpoints", exist_ok=True)
                if stage == 1:
                    # 階段一創紀錄：暫存至臨時權重檔，準備未來回滾
                    torch.save(model.state_dict(), tmp_checkpoint_path)
                else:
                    # 階段二創紀錄：全域最終最佳模型，儲存正式檢查點並更新大考結帳分數
                    torch.save(model.state_dict(), f"checkpoints/dress_scorer_fold{fold+1}_best.pth")
                    best_fold_rmse = epoch_val_rmse
                    best_fold_r2 = epoch_val_r2
            else:
                # 沒進步，耐性計數器累加
                patience_counter += 1
                
            # ---- 🛑 檢查 Stage 1 是否觸發跳轉條件 ----
            if stage == 1:
                if patience_counter >= PATIENCE_A or current_stage_epoch >= MAX_EPOCH_B:
                    print(f"➔ [系統提示] Stage 1 飽和 (連續 {patience_counter} 輪未進步或達上限 {MAX_EPOCH_B} 輪)。")
                    
                    # 🚨 關鍵回滾：強制將大腦回復到第一階段表現最好的那一個 Epoch 狀態
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
                    
                    # 狀態機環境重置，讓第二階段重新算帳
                    stage = 2
                    current_stage_epoch = 0
                    best_val_loss = float('inf')  # 重置為無限大，讓 Stage 2 的第一輪強制作為起點基礎
                    patience_counter = 0
                    
                    with open(log_filename, "a", encoding="utf-8") as log_file:
                        log_file.write(f"➔ [系統狀態切換] Stage 1 結束。載入最佳拉桿，全面解凍進入 Stage 2。\n")
                    continue  # 直接進入下一個迴圈，不執行下面的 Stage 2 檢查

            # ---- 🏁 檢查 Stage 2 是否滿足最終終止條件 ----
            elif stage == 2:
                if patience_counter >= PATIENCE_C or current_stage_epoch >= MAX_EPOCH_D:
                    print(f"🏁 [系統提示] Stage 2 觸發最終早停 (連續 {patience_counter} 輪未進步或達上限 {MAX_EPOCH_D} 輪)。")
                    
                    # 清除暫存檔案，保持目錄乾淨
                    if os.path.exists(tmp_checkpoint_path):
                        os.remove(tmp_checkpoint_path)
                        
                    with open(log_filename, "a", encoding="utf-8") as log_file:
                        log_file.write(f"➔ [系統訓練終止] Stage 2 觸發早停，本折（Fold）實驗在此安全收尾。\n")
                    break  # 徹底破出 while 迴圈，結束當前 Fold

        print(f"✨ Fold [{fold + 1}] 結束！最佳驗證集歸檔成績 -> R²: {best_fold_r2:.4f}, RMSE: {best_fold_rmse:.4f}")
        all_fold_val_r2.append(best_fold_r2)
        all_fold_val_rmse.append(best_fold_rmse)
        
        with open(log_filename, "a", encoding="utf-8") as log_file:
            log_file.write(f"➔ Fold [{fold + 1}] 最佳盲測表現總結 -> Val Loss(MSE)最低點對應之 RMSE: {best_fold_rmse:.4f} │ R²: {best_fold_r2:.4f}\n")
            log_file.write("──────────────────────────────────────────────────────────────────────────\n")

    # --- 4. 算總帳：寫入最終全域平均成績 ---
    mean_r2 = np.mean(all_fold_val_r2)
    mean_rmse = np.mean(all_fold_val_rmse)
    
    summary_text = (
        f"\n==================== 📊 全域交叉驗證總結 ====================\n"
        f"各 Fold 最佳 RMSE: {[f'{r:.4f}' for r in all_fold_val_rmse]}\n"
        f"各 Fold 最佳 R² 分數: {[f'{r:.4f}' for r in all_fold_val_r2]}\n"
        f"----------------------------------------------------------\n"
        f"🔥 5-Fold 平均驗證集 RMSE: 【{mean_rmse:.4f}】\n"
        f"🚀 5-Fold 平均驗證集 R² 分數: 【{mean_r2:.4f}】\n"
        f"==========================================================\n"
    )
    
    print(summary_text)
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(summary_text)
        
    print(f"💾 完整自動控制實驗數據已成功保存至 {log_filename}")

if __name__ == "__main__":
    main()