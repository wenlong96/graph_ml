# CPL Optimisation on Campaign Graphs

## 📖 Overview
This project investigates **Cost-Per-Lead (CPL) optimisation** for paid advertising campaigns. We frame the problem as a node-level prediction task, classifying campaigns into "Low-CPL" (efficient) vs. "High-CPL" (inefficient) regimes.

The core research question is: **Can Graph Neural Networks (GNNs) leverage relationships between campaigns (structural similarity) to outperform traditional tabular baselines?**

We benchmark tabular models against various Graph architectures, ranging from local message-passing networks (GCN/GAT) to global attention mechanisms (Graph Transformers).

---

## 🛠 Data & Preprocessing
The dataset consists of ad campaign configurations at the **Platform × Device** granularity.

**Feature Engineering:**
1.  **Efficiency Ratios:** CTR, Lead Rate, Conversion Rate, Spend-Per-Impression.
2.  **Time Features:** Seasonality (Month, Quarter) to capture market drift.
3.  **Text Embeddings:** Headline embeddings generated using `Qwen/Qwen3-Embedding-0.6B`.
4.  **Target Variable:** `CPL_eff` (Effective Cost Per Lead).

**Graph Construction:**
To test graph learning, we constructed two types of campaign graphs:
* **k-NN Graph:** Connects campaigns that are close in the engineered feature space (Euclidean distance).
* **Shared Metadata Graph:** Connects campaigns that share explicit business logic (e.g., same Platform + Objective).

---

## 🔬 Methodology & Models
We implemented a progressive series of experiments to isolate the impact of graph structure:

### 1. Non-Graph Baselines
* **K-Means Clustering:** To test for natural segmentation without labels.
* **Logistic Regression:** Optimistic upper-bound baseline.
* **Deep Neural Network (DNN):** Standard tabular classification.

### 2. Local Message Passing GNNs
* **GCN (Graph Convolutional Network):** Standard aggregation.
* **Anisotropic GCN:** Using GAT-style attention to weight neighbors.
* **Hybrid GCN:** Decoupled MLP encoder + GCN head to preserve feature signals.

### 3. Global Graph Learning
* **Graph Transformer (GT):** Uses Multi-Head Attention to allow global node communication, bypassing local neighborhoods.
* **GT + LapPE:** Graph Transformer with Laplacian Positional Encodings.

---

## 📊 Results Summary

Our experiments revealed a clear failure mode of local GNNs on this dataset, which was rectified by Graph Transformers.

| Model Architecture | Mechanism | Result (F1) | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Tabular | **High** | Strong predictive signal exists in node features. |
| **DNN** | Tabular | **High** | Non-linear feature interactions are predictive. |
| **GCN / GAT** | Local Aggregation | ~0.0 (Fail) | **Over-smoothing.** Local neighbors are noisy/homogenous. |
| **Hybrid GCN** | MLP + Local MP | ~0.0 (Fail) | Local message passing actively destroys the signal. |
| **Graph Transformer** | **Global Attention** | **High** | **Success.** Bypassed local noise via global context. |
| **GT + LapPE** | Global + Positional | High (Degraded) | Over-expressivity led to slight overfitting. |

### 🔑 Key Findings
1.  **Topology is Noise:** Local structures (kNN or Shared Metadata) introduced noise. Aggregating neighbors caused **over-smoothing**, washing out the high-fidelity signals of individual campaigns.
2.  **Context is Key:** The **Graph Transformer** succeeded where GCNs failed because it utilized **Global Attention**. It allowed campaigns to attend to the entire ecosystem rather than being forced to average with their immediate, noisy neighbors.
**References:**
* *Switch Transformers / Mesh TensorFlow (Concepts applied in architectural logic)*
* *A Benchmarking Study of Graph Neural Networks (Dwivedi et al.)*
