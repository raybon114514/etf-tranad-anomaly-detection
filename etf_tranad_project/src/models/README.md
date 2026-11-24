# 深度學習異常偵測模型 (`etf_tranad_project/src/models/`)

本套件實作兩款多變量時間序列重建／對抗異常偵測模型：

| 檔案 | 主要類別 | 說明 |
| :--- | :--- | :--- |
| `__init__.py` | — | 套件初始化檔 |
| [`tranad.py`](tranad.py) | `PositionalEncoding`、`TranAD` | 多變量 Transformer 雙階段自條件對抗架構，考量金融報酬無界特性將解碼器輸出改為 Linear Decoder |
| [`vtt.py`](vtt.py) | `DilatedCausalConv`、`VTT` | 結合膨脹因果卷積（參考 He & Zhao 2019 TCN）與 Transformer 注意力機制之混合架構 |
