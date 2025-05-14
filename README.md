# 自動回覆系統

本專案使用 Streamlit 建立網頁介面，可輸入來文內容，並透過向量庫相似度搜尋自動生成回覆。

## 安裝與執行

1. 在專案根目錄建立並啟用 Python 虛擬環境後，安裝相依套件：
```
pip install -r requirements.txt
```
2. 執行：
```
streamlit run app.py
```

## 部署至 Streamlit Cloud

1. 將本資料夾推送至 GitHub，並在 Streamlit Cloud 中，將專案來源指向此儲存庫的 `streamlit_app` 資料夾。
2. Streamlit Cloud 會自動偵測 `requirements.txt`，並執行預設命令 `streamlit run app.py`。
