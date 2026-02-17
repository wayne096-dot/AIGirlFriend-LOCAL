## Local AI Assistant(Girlfriend) System

這是一個基於 **全本地運行 (Offline-only)** 的 AI 聊天助手（女友）系統。它結合了大型語言模型 (LLM) 的邏輯推理與 VITS 的高音質語音合成，並搭配 Live2D 模型進行實時嘴型同步與角色互動。

所有對話數據與 AI 推理均在本地完成，保障您的隱私安全。



## 主要功能

-   **完全本地化**: 無需連網，不調用外部 API，隱私安全無憂。
-   **日語語音輸出**: 系統接收中文輸入，透過本地 LLM 生成俏皮的日文回答，並以日語 TTS 播放。
-   **實時互動**: 搭配 Live2D 模型，嘴型根據語音能量實時同步。

## 技術棧 (Tech Stack)

* **LLM 推理**: `llama-cpp-python` (基於 GGUF 格式模型)
* **TTS 語音**: `onnxruntime` + VITS (Tamamo Cross)
* **前端 UI**: `tkinter`
* **Live2D 同步**: `Live2dTK`
* **語言處理**: `pykakasi` (中文處理與日文轉換)

## 安裝與運行

### 1. 環境需求
- Python 3.10+
- 安裝必要庫:
  ```pip install llama-cpp-python numpy sounddevice onnxruntime pykakasi Live2dTK live2d-py```

## 依賴套件版權聲明 (Dependency Licenses)

本專案依賴於以下第三方軟體庫，其版權歸原作者所有：

* **llama-cpp-python**: [MIT License](https://github.com/abetlen/llama-cpp-python/blob/main/LICENSE)
* **onnxruntime**: [MIT License](https://github.com/microsoft/onnxruntime/blob/main/LICENSE)
* **numpy**: [BSD 3-Clause License](https://github.com/numpy/numpy/blob/main/LICENSE.txt)
* **sounddevice**: [MIT License](https://github.com/spatialaudio/python-sounddevice/blob/master/LICENSE)
* **pykakasi**: [MIT License](https://github.com/miurahr/pykakasi/blob/master/LICENSE)
* **Live2dTK**: [MIT License/GPL］



### 重要提示 (Models & Assets)
* **Tamamo Cross TTS 模型**: 基於 VITS 架構，通常屬於開源訓練權重，請遵循原訓練者的授權說明。
* **Live2D 模型**: 本專案僅提供程式架構。**若您使用特定角色模型，請務必確認該模型檔案的著作權授權** (例如是否禁止商用、是否需署名等)。本專案作者不對用戶自行使用的模型版權負責。
