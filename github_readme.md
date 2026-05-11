# GitHub 上傳規則與流程

**Remote：** `https://github.com/denghongli1021/I2AI_FinalProject_AutoML`
**專案根目錄：** `d:\UserData\claude_project\人工智慧project\`
**注意：** `.git` 資料夾在 `code/` 子目錄內，git 指令須在 `code/` 下執行

---

## 一、禁止上傳

| 資料夾 | 原因 |
|---|---|
| `openml_cc18_data/` | 大型資料集 |
| `ucr_ts_80(時序資料)/` | 大型資料集 |
| `autogluon_models/` | 模型快取，可重新生成 |
| `test/` | 本地測試 |
| `doc/` | 內部文件 |
| `.claude/` | Claude Code 個人設定 |
| `.github_readme.md` | Github上傳設定 |
| `CLAUDE.md` | Claude 內容 |

---

## 二、可上傳

- `automl_platform/`（排除 `__pycache__/`）
- `run_automl.py`、`run_baseline.py`、`run_comparison.py`、`run_pipeline.py`、`data_colloect_time.py`、`data_collect.py`
- `模型訓練架構規格書.md`、`模型訓練架構規格書_v2.md`
- `.gitignore`、`README.md`、`requirements.txt`
- `run.md`

---

## 三、上傳流程

```powershell
# 1. 進入 git 工作目錄
cd "d:\UserData\claude_project\人工智慧project\code"

# 2. 建立或切換分支（首次建立用 -b）
git checkout -b <分支名稱>
# 已存在的分支：git checkout <分支名稱>

# 3. 從上層目錄複製最新檔案（排除 __pycache__）
robocopy "..\automl_platform" ".\automl_platform" /E /XD __pycache__
copy "..\run_pipeline.py" ".\run_pipeline.py"
copy "..\run_baseline.py" ".\run_baseline.py"
copy "..\模型訓練架構規格書.md" ".\模型訓練架構規格書.md"
copy "..\模型訓練架構規格書_v2.md" ".\模型訓練架構規格書_v2.md"

# 4. 確認沒有禁止的資料夾被加入
git status

# 5. 加入並 commit
git add .gitignore automl_platform/ run_automl.py run_baseline.py "模型訓練架構規格書.md" "模型訓練架構規格書_v2.md"
git commit -m "說明這次更新了什麼"

# 6. 推送
git push -u origin <分支名稱>
```

---

## 四、快速確認清單

- [ ] 在 `code/` 目錄下執行（不是專案根目錄）
- [ ] `git status` 無禁止資料夾
- [ ] 無 `*.pyc`、`.env`、大型二進位檔
