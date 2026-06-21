-- 1. 確保資料庫存在（若不存在才建立）
CREATE DATABASE IF NOT EXISTS dress_gpt_advanced
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

-- 2. 切換使用該資料庫
USE dress_gpt_advanced;

-- =======================================================
-- 第一部分：選單元資料管理表
-- =======================================================

-- 建立主風格清單表
CREATE TABLE IF NOT EXISTS label_styles (
    style_name VARCHAR(50) PRIMARY KEY
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 建立屬性標籤清單表（將 is_pass 改為 is_positive 區分正負向）
CREATE TABLE IF NOT EXISTS label_attributes (
    attr_name VARCHAR(50) PRIMARY KEY,
    is_positive TINYINT(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- =======================================================
-- 第二部分：照片標註核心資料表
-- =======================================================

-- 主表：存放照片的核心基本資料（刪除 is_pass，新增精確的性別、季節、正式度）
CREATE TABLE IF NOT EXISTS dress_dataset (
    image_id VARCHAR(50) NOT NULL,
    gender VARCHAR(20) NOT NULL DEFAULT 'male',          -- 預設為男裝
    season VARCHAR(20) NOT NULL DEFAULT 'spring_autumn', -- 預設為最常見的春秋季
    formality VARCHAR(20) NOT NULL DEFAULT 'casual',     -- 預設為休閒
    image_path VARCHAR(255) NOT NULL,
    overall_score DECIMAL(4, 2) NOT NULL,
    main_style VARCHAR(50) DEFAULT NULL,
    PRIMARY KEY (image_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 從表：專門處理多值屬性標籤
CREATE TABLE IF NOT EXISTS image_attributes (
    image_id VARCHAR(50) NOT NULL,
    attribute_name VARCHAR(50) NOT NULL,
    attribute_value DECIMAL(4, 2) NOT NULL,
    PRIMARY KEY (image_id, attribute_name),
    FOREIGN KEY (image_id) REFERENCES dress_dataset(image_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;