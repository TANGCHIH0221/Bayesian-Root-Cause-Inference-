# Bayesian-Root-Cause-Inference-
利用 **Noisy-OR 模型結合 EM 演算法**，進行 **不確定性分析與根因推斷**。   透過模擬資料，我們嘗試辨識「最容易造成失敗的節點」，並提供決策依據。
---

## 方法 (Method)
- **資料生成**：用 Poisson / Normal / Gamma 三種分布產生節點失敗機率 \\(q_j\\)，再由隨機路由矩陣 \\(A\\) 生成觀測結果 \\(y\\)。  
- **模型假設**：  
  \[
  P(y_i=1) = 1 - \prod_j (1 - A_{ij} q_j) \quad \text{(Noisy-OR)}
  \]
- **推斷方法**：  
  - 使用 **EM (Expectation-Maximization)** 演算法估計 \\(q_j\\)  
  - **E-step**：將失敗樣本的「責任」分配到經過的節點  
  - **M-step**：根據責任更新節點失敗率（帶 Beta 先驗，避免估計過度膨脹）

---
