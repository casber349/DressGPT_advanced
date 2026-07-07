import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import KFold

# 載入你剛才驗證成功的屬性組件
from attribute_labeler_dataset import DressAttributeDataset
from attribute_labeler_model import DressAttributeLabelerModel

# 💡 1. 實作你手繪確認過、完全不帶 Mask 的純粹型 Soft IoU Loss 引擎
class SoftIoULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        # 透過 Sigmoid 轉成連續機率 (0.0 ~ 1.0) 也就是你圖中的藍色數字
        probs = torch.sigmoid(logits)
        
        # 聯立矩陣乘法：交集 (Intersection) 與聯集 (Union)
        intersection = torch.sum(probs * targets, dim=1)
        union = torch.sum(probs, dim=1) + torch.sum(targets, dim=1) - intersection
        
        # 計算連續空間下的 Soft IoU (加入 eps 防止分母為 0 導致 5070 噴 NaN)
        iou = (intersection + self.eps) / (union + self.eps)
        
        # Loss = 1 - Soft IoU，目標是讓 IoU 直奔 1.0
        return 1.0 - torch.mean(iou)

def main():
    # --- 1. 基礎超參數設定 ---
    BATCH_SIZE = 32
    BACKBONE_LR = 1e-5     # 解凍後大腦特徵層的微弱學習率
    HEAD_LR = 1e-3         # 屬性思考頭的標準學習率
    WEIGHT_DECAY = 1e-4    # 正則化防止過擬合
    
    # 💡 雙階段動態早停超參數 (a, b, c, d)
    PATIENCE_A = 7      # ⬅️ 參數 a: 凍結階段連續幾輪沒進步就停下來
    MAX_EPOCH_B = 40    # ⬅️ 參數 b: 凍結階段最多跑幾輪
    
    PATIENCE_C = 10     # ⬅️ 參數 c: 不凍結階段連續幾輪沒進步就停下來
    MAX_EPOCH_D = 40    # ⬅️ 參數 d: 不凍結階段最多跑幾輪
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 雙階段動態早停 Attribute Labeler K-fold 訓練啟動！設備: {device}")
    
    # --- LOG 檔案初始化 ---
    log_filename = "kfold_attribute_log.txt"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write("==================================================\n")
        log_file.write("   DressGPT Attribute Labeler 雙階段動態早停 K-Fold 紀錄  \n")
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
    
    # 初始化屬性資料集
    full_dataset = DressAttributeDataset(transform=train_transform)
    total_count = len(full_dataset)
    num_attributes = full_dataset.num_attributes # 這裡會自動抓到 38
    
    # 💡 關鍵安全步驟：將動態生成的 38 類屬性對照表存為 JSON，供未來 Web 推論或鄰居分贓時查表
    mapping_filename = "attribute_mapping.json"
    with open(mapping_filename, "w", encoding="utf-8") as f:
        json.dump(full_dataset.id_to_attr, f, ensure_ascii=False, indent=4)
    print(f"💾 屬性標籤文字對照表已安全固化至: {mapping_filename}")
    
    K_SPLITS = 5
    kf = KFold(n_splits=K_SPLITS, shuffle=True, random_state=42)
    all_fold_val_loss = []
    all_fold_val_iou = []  
    
    print(f"🌀 開始執行 {K_SPLITS}-Fold 交叉驗證，紀錄將寫入 {log_filename}...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(total_count))):
        print(f"\n==================== 🔥 Fold [{fold + 1}/{K_SPLITS}] 開跑 ====================")
        
        with open(log_filename, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n--- Fold [{fold + 1}/{K_SPLITS}] 詳細屬性訓練歷程 ---\n")
            log_file.write("Epoch │ Stage Tag      │ Train Loss (IoU)   │ Val Loss (IoU)\n")
            log_file.write("──────────────────────────────────────────────────────────────────────────\n")

        train_sub_dataset = Subset(full_dataset, train_idx)
        val_sub_dataset = Subset(full_dataset, val_idx)
        val_sub_dataset.dataset.transform = val_transform 
        
        train_loader = DataLoader(train_sub_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_sub_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        # 建立指定屬性數的模型
        model = DressAttributeLabelerModel(num_attributes=num_attributes).to(device)
        
        # 🚀 初始狀態：進入 Stage 1，鎖死大腦特徵層，只開自訂的 64 維多層緩衝思考頭
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.block1.parameters():
            param.requires_grad = True
        for param in model.block2.parameters():
            param.requires_grad = True
        for param in model.final_layer.parameters():
            param.requires_grad = True
            
        head_params = list(model.block1.parameters()) + list(model.block2.parameters()) + list(model.final_layer.parameters())
        optimizer = optim.AdamW(head_params, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
        
        # ⚙️ 丟入你設計的專屬 Loss 引擎
        criterion = SoftIoULoss()
        
        # ⚙️ 雙階段動態控制器狀態初始化
        stage = 1
        current_stage_epoch = 0
        global_epoch = 0
        best_val_loss = float('inf')  
        patience_counter = 0
        
        best_fold_loss = float('inf')
        best_fold_iou = 0.0
        
        tmp_checkpoint_path = f"checkpoints/tmp_best_attribute_stage1_fold{fold+1}.pth"
        
        while True:
            global_epoch += 1
            current_stage_epoch += 1
                
            # --- 訓練階段 ---
            model.train()
            train_loss = 0.0
            train_iou_sum = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device) # labels 預設就是多標籤的 0/1 浮點向量
                optimizer.zero_grad()
                
                # 💡 注意模型現在是雙輸出，我們取第一個 Logits 計算 Loss 即可
                outputs, _ = model(images)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                
                # 線上計算 Hard IoU (閥值選定 0.25) 供你開根號級別的直覺觀測
                preds = (torch.sigmoid(outputs) > 0.25).float()
                inter = torch.sum(preds * labels, dim=1)
                union = torch.sum(preds, dim=1) + torch.sum(labels, dim=1) - inter
                batch_iou = torch.sum((inter + 1e-7) / (union + 1e-7))
                train_iou_sum += batch_iou.item()
                
            epoch_train_loss = train_loss / len(train_loader.dataset)
            epoch_train_iou = train_iou_sum / len(train_loader.dataset)

            # --- 驗證階段 ---
            model.eval()
            val_loss = 0.0
            val_iou_sum = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs, _ = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    inter = torch.sum(preds * labels, dim=1)
                    union = torch.sum(preds, dim=1) + torch.sum(labels, dim=1) - inter
                    batch_iou = torch.sum((inter + 1e-7) / (union + 1e-7))
                    val_iou_sum += batch_iou.item()
                    
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_iou = val_iou_sum / len(val_loader.dataset)
            
            stage_tag = f"[S1:Lock]" if stage == 1 else f"[S2:Unlk]"
            
            # 即時輸出列印（將 IoU 轉成百分比顯示，非常完美）
            print(f"Epoch {global_epoch:02d} {stage_tag} │ Train Loss: {epoch_train_loss:.4f} (IoU: {epoch_train_iou*100:.2f}%) │ Val Loss: {epoch_val_loss:.4f} (IoU: {epoch_val_iou*100:.2f}%)")
            
            with open(log_filename, "a", encoding="utf-8") as log_file:
                log_file.write(f"Epoch {global_epoch:02d} │ {stage_tag:14s} │ Train: {epoch_train_loss:.4f} ({epoch_train_iou*100:.2f}%) │ Val Loss: {epoch_val_loss:.4f} ({epoch_val_iou*100:.2f}%)\n")
            
            # ==========================================
            # 📉 雙階段動態早停核心邏輯控制器
            # ==========================================
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                
                os.makedirs("checkpoints", exist_ok=True)
                if stage == 1:
                    torch.save(model.state_dict(), tmp_checkpoint_path)
                else:
                    torch.save(model.state_dict(), f"checkpoints/dress_attribute_labeler_fold{fold+1}_best.pth")
                    best_fold_loss = epoch_val_loss
                    best_fold_iou = epoch_val_iou
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
                        {'params': model.backbone.parameters(), 'lr': BACKBONE_LR},
                        {'params': head_params, 'lr': HEAD_LR}
                    ], weight_decay=WEIGHT_DECAY)
                    
                    stage = 2
                    current_stage_epoch = 0
                    best_val_loss = float('inf')
                    patience_counter = 0
                    
                    with open(log_filename, "a", encoding="utf-8") as log_file:
                        log_file.write(f"➔ [系統狀態切換] Stage 1 結束。載入最佳屬性頭，全面解凍進入 Stage 2。\n")
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

        print(f"✨ Fold [{fold + 1}] 結束！最佳驗證集歸檔成績 -> Loss: {best_fold_loss:.4f}, Mean IoU: {best_fold_iou*100:.2f}%")
        all_fold_val_loss.append(best_fold_loss)
        all_fold_val_iou.append(best_fold_iou)
        
        with open(log_filename, "a", encoding="utf-8") as log_file:
            log_file.write(f"➔ Fold [{fold + 1}] 最佳盲測表現總結 -> Val Loss 最低點對應之 Loss: {best_fold_loss:.4f} │ Mean IoU: {best_fold_iou*100:.2f}%\n")
            log_file.write("──────────────────────────────────────────────────────────────────────────\n")

    # --- 4. 算總帳：寫入最終全域平均成績 ---
    mean_loss = np.mean(all_fold_val_loss)
    mean_iou = np.mean(all_fold_val_iou)
    
    summary_text = (
        f"\n==================== 📊 全域交叉驗證總結 ====================\n"
        f"各 Fold 最佳 Loss (1 - Soft_IoU): {[f'{r:.4f}' for r in all_fold_val_loss]}\n"
        f"各 Fold 最佳 Mean IoU: {[f'{r*100:.2f}%' for r in all_fold_val_iou]}\n"
        f"----------------------------------------------------------\n"
        f"🔥 5-Fold 平均驗證集 Loss: 【{mean_loss:.4f}】\n"
        f"🚀 5-Fold 平均驗證集 屬性重疊度(Mean IoU): 【{mean_iou*100:.2f}%】\n"
        f"==========================================================\n"
    )
    
    print(summary_text)
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(summary_text)
        
    print(f"💾 完整屬性標註器自動控制實驗數據已成功保存至 {log_filename}")

if __name__ == "__main__":
    main()