Proposal: Generalizable AI-Generated Image Detection Using Multi-View Forensic Features
1. Project Title
Generalizable AI-Generated Image Detection: Combining Semantic, Frequency, and Reconstruction-Based Forensic Cues
本研究旨在建立一個能辨識影像真實性（Real vs AI-generated）的影像鑑識系統。不同於單純將任務視為二元分類，本專題將問題重新定義為「在生成模型、壓縮方式、解析度與後處理條件改變時，模型是否仍能穩定辨識 AI 生成影像」。

2. Motivation and Problem Background
隨著 Stable Diffusion、DALL·E、Midjourney、Imagen 等生成模型快速進步，AI 生成影像已能在社群媒體、新聞圖片、廣告素材與藝術創作中產生高度擬真的視覺內容。這使得「影像是否真實」不再只是人眼判斷問題，而成為一個需要自動化影像鑑識方法支援的 computer vision task。
原始 proposal 將任務定義為：輸入 32×32 RGB image，輸出 Real / AI-generated label。這樣的定義雖然適合建立 baseline，但過於封閉。因為模型可能只學到 CIFAKE dataset 中 Stable Diffusion 1.4 與 CIFAR-10 之間的資料差異，而不是真正學到 AI 生成影像的通用 forensic cues。
CIFAKE 論文指出，該資料集包含 60,000 張 CIFAR-10 真實影像與 60,000 張 Stable Diffusion 生成影像，最佳 CNN 模型達到 92.98% accuracy；但 Grad-CAM 分析也顯示，模型關注的可能不是主要物體本身，而是背景中的細微不自然痕跡。這表示單純高 accuracy 並不代表模型真正理解「AI 生成」的本質。(arXiv)
因此，本研究的核心問題不只是「能不能在 CIFAKE 上分類成功」，而是：
模型是否能學到可泛化的 AI 影像鑑識特徵，並在不同生成器、不同壓縮條件與不同影像退化情境下維持穩定表現？

3. Task Definition
3.1 Main Task: Binary Authenticity Classification
Input：一張 RGB image。
Output：影像為真實照片或 AI 生成影像的機率。
在 CIFAKE setting 中，影像解析度為 32×32；在延伸實驗中，會使用較高解析度資料集或將影像 resize 至模型輸入大小，例如 224×224。
3.2 Extended Task: Open-Set Generalization
除了 in-distribution 分類，本研究加入 open-set / cross-generator evaluation：
Train：CIFAKE 或部分 generator 產生的 fake images。
Test：未出現在訓練階段的 generator，例如其他 diffusion models 或 GAN-based models。
這個設計是因為現有研究指出，許多 detector 在訓練資料內表現良好，但遇到未看過的生成模型時會明顯退化。Ojha 指出，傳統 real-vs-fake classifier 容易只學到已知 fake pattern，導致 unseen fake images 被吸收到 real class 中。(arXiv)
3.3 Optional Auxiliary Task: Generator-Family Attribution
若資料允許，本研究也會加入輔助任務：
Output 1：Real / AI-generated。
Output 2：若為 AI-generated，預測可能的 generator family，例如 Stable Diffusion、GAN、Midjourney-like、DALL·E-like。
此任務不作為主評分指標，而是幫助模型學習不同生成器的 artifact pattern，並輔助 error analysis。

4. Dataset Design
4.1 Primary Dataset: CIFAKE
CIFAKE 將作為主要 controlled benchmark。它包含：
Real images：60,000 張，來自 CIFAR-10。
Fake images：60,000 張，由 Stable Diffusion 1.4 生成。
Resolution：32×32 RGB。
Classes：對應 CIFAR-10 的 10 個語意類別。
Task：Binary classification, Real vs AI-generated。
本研究不直接採用任意 80/10/10 切分，而是保留 CIFAKE 官方 train/test 分布，再從 training set 中切出 validation set。這樣能避免實驗設定與原始 benchmark 不一致。
4.2 Generalization Dataset: GenImage Subset
CIFAKE 只有單一 fake generator，無法充分測試泛化能力。因此本研究會加入 GenImage 的子集作為 cross-generator testing。GenImage 是一個 million-scale benchmark，包含超過一百萬組 real/fake image pairs，並涵蓋多種 diffusion models 與 GANs。該 benchmark 特別設計了 cross-generator image classification 與 degraded image classification 兩種任務，用來測試模型在未見生成器與影像退化下的穩健性。(arXiv)
若計算資源有限，本專題會從 GenImage 中選取代表性 generator subset，例如：
Training generators：Stable Diffusion / BigGAN subset。
Unseen testing generators：Midjourney-like / ADM / GLIDE / VQDM subset。
Robustness testing：JPEG compression、blur、resize、noise。
4.3 Hard-Case Dataset: Chameleon or Similar Benchmark
若資料取得可行，本研究會額外使用 Chameleon 或類似高難度 AI image detection benchmark 作為 hard test set。A Sanity Check for AI-generated Image Detection 指出，許多現有 detector 在一般 benchmark 上表現良好，但在 Chameleon 這種更接近人類難以分辨的資料上，會大量將 AI-generated images 判成 real images。(arXiv)

5. Data Validation and Bias Control
本研究會特別處理 dataset bias，因為 AI image detection 很容易受到非語意因素干擾，例如 JPEG 壓縮程度、解析度差異、resize pipeline、檔案格式與來源平台。
5.1 Label and Distribution Check
資料驗證包含：
確認 Real / Fake label 對應正確。
確認類別分布是否平衡。
檢查不同 semantic class 是否均衡，例如 airplane、cat、dog、truck 等。
檢查 train / validation / test 是否有重複影像或近似重複影像。
檢查 real 與 fake 是否存在解析度、壓縮率、檔案格式差異。
5.2 Compression and Resolution Bias Control
Fake or JPEG? 一文指出，AI-generated image detection datasets 常含 JPEG compression 與 image size bias，導致 detector 學到的不是生成痕跡，而是資料處理流程差異；移除這些 bias 後，cross-generator performance 會明顯改變。(arXiv)
因此本研究會加入 matched preprocessing：
所有影像統一轉為 RGB。
訓練與測試資料使用一致 resize pipeline。
Real / Fake 影像套用相同 JPEG quality range。
若做 compression robustness test，必須對 real 與 fake 同時處理。
避免模型只靠圖片清晰度、壓縮痕跡或尺寸差異分類。

6. Preprocessing and Data Augmentation
6.1 Basic Preprocessing
CIFAKE setting：
Resize：保留 32×32，或另行 upscale 至 224×224 供 pretrained backbone 使用。
Normalization：使用 CIFAR-10 mean/std 或 ImageNet mean/std，依模型而定。
Label encoding：Real = 0, AI-generated = 1。
Extended setting：
Resize to 224×224 for ResNet / CLIP / DINO-style models。
Maintain original aspect ratio when possible。
Use center crop or random resized crop only in training。
6.2 Robustness-Oriented Augmentation
原始 proposal 中的 flip、rotation、noise 太一般化，與 AI-generated image detection 的問題不完全對應。本研究改用 forensic-oriented augmentation：
JPEG compression：quality = 30, 50, 70, 90。
Gaussian blur：模擬社群平台壓縮或低品質圖片。
Resize down-up：模擬截圖、轉傳、壓縮。
Color jitter：避免模型過度依賴色彩分布。
Gaussian noise：測試低階 artifact 是否穩定。
Random crop：避免模型只學固定位置 artifact。
Wang 在 CNN-generated image detection 中顯示，augmentation 對於跨生成模型泛化非常重要，尤其是 blur 與 JPEG 類型的後處理。(arXiv)

7. Proposed Method
本研究不只比較 Simple CNN vs ResNet，而是設計一個逐層加強的 model ladder，最後提出 multi-view forensic detector。
7.1 Baseline 1: Simple CNN
第一個 baseline 使用小型 CNN，對應 CIFAKE 原始任務。
Architecture：
Conv-BN-ReLU blocks。
Max pooling。
Global average pooling。
Fully connected binary classifier。
Loss：Binary cross entropy。
Purpose：
確認資料 pipeline 正確。
建立 CIFAKE in-distribution baseline。
作為後續模型比較基準。
Expected limitation：
可能只在 CIFAKE 上表現良好。
容易學到 background artifacts。
對 unseen generator 泛化能力弱。
7.2 Baseline 2: ResNet-Based Classifier
第二個 baseline 使用 ResNet-18 或 ResNet-50。若輸入為 32×32，需調整第一層 convolution 或將影像 resize 至 224×224。
Purpose：
比較 handcrafted small CNN 與標準 deep CNN。
測試更深模型是否提升 CIFAKE accuracy。
觀察是否出現 overfitting 或 shortcut learning。
7.3 Baseline 3: CLIP Feature Linear Probe
第三個 baseline 使用 frozen CLIP image encoder，抽取 image embedding 後接 linear classifier。這是因為研究顯示，使用未專門為 fake detection 訓練的大型視覺語言模型特徵，對 unseen generator 有較好泛化能力。Ojha 使用大型 pretrained vision-language model feature space，發現簡單 nearest-neighbor 或 linear probing 就能提升 unseen generative model detection 表現。(arXiv)
Purpose：
測試 pretrained semantic representation 是否比純 CNN 更穩定。
作為 open-set generalization baseline。
避免模型完全依賴 CIFAKE 的低階 artifact。
7.4 Proposed Model: Multi-View Forensic Detector
本研究主要模型為三分支架構：
Branch A: Semantic-Spatial Branch
使用 ResNet / CLIP / DINO-style backbone 抽取高階語意與空間特徵。此分支負責偵測物體結構、語意一致性、背景邏輯與整體視覺不自然性。
Branch B: Frequency-Artifact Branch
將影像轉換至頻率或殘差空間，例如：
FFT spectrum。
DCT coefficients。
High-pass residual。
SRM-like filters。
此分支負責偵測生成模型可能留下的 spectral artifact、upsampling trace、noise inconsistency。相關研究顯示，生成影像常在高頻或頻譜分布上與真實影像不同，這類特徵對 AI-generated image detection 很重要。(arXiv)
Branch C: Patch-Level Forensic Branch
將影像切成 patches，選取高頻與低頻 patch，分別抽取局部特徵。這個設計參考 AIDE 的概念：同時利用 high-level semantic embedding 與 low-level patchwise features，捕捉 noise pattern、anti-aliasing、local artifact 等線索。(arXiv)
最後將三個分支的特徵 concat，輸入 fusion classifier：

7.5 Optional Stretch Module: Reconstruction Error Branch
若時間與 GPU 資源允許，加入 diffusion reconstruction error。DIRE 指出，diffusion-generated images 通常能被 diffusion model 較好重建，而 real images 的 reconstruction error 較大，因此 reconstruction error 可作為辨識 diffusion-generated images 的線索。(arXiv)
但此分支計算成本較高，因此放在 stretch goal。可行替代方案是使用 latent reconstruction error，例如 LaRE²，降低計算成本。(arXiv)

8. Training Strategy
8.1 Loss Function
Main loss：
Binary cross entropy。
若加入 generator attribution：

其中 L_generator 為 multi-class cross entropy，只對 fake images 計算。
8.2 Calibration
本研究不只追求 accuracy，也會觀察模型 confidence 是否可信。若模型對錯誤預測仍高度自信，實際應用價值會下降。因此會加入：
Temperature scaling。
Expected Calibration Error (ECE)。
Negative Log-Likelihood (NLL)。
8.3 Training Protocol
Stage 1：在 CIFAKE 上訓練 Simple CNN、ResNet、CLIP linear probe。
Stage 2：訓練 multi-view model。
Stage 3：使用 GenImage subset 測試 unseen generator。
Stage 4：加入 JPEG、blur、resize、noise 測試 robustness。
Stage 5：進行 explainability 與 error analysis。

9. Evaluation Design
9.1 In-Distribution Evaluation
在 CIFAKE test set 上評估：
Accuracy。
Precision。
Recall。
F1-score。
AUROC。
Average Precision。
Accuracy 用於與原始 CIFAKE 結果比較，但不作為唯一重點。
9.2 Cross-Generator Evaluation
此為本研究的主要貢獻之一。
設定：
Train on CIFAKE / Stable Diffusion subset。
Test on unseen generators from GenImage subset。
Metrics：
AUROC。
Average Precision。
Balanced Accuracy。
F1-score。
False Positive Rate at fixed True Positive Rate。
目的：
判斷模型是否只學到 Stable Diffusion 1.4 的 artifact，還是真的能泛化到其他生成模型。
9.3 Robustness Evaluation
對 test images 套用後處理：
JPEG quality = 90 / 70 / 50 / 30。
Gaussian blur。
Downsample then upsample。
Gaussian noise。
Screenshot-like compression。
每種 perturbation 分別報告 AUROC、F1-score、Accuracy drop。
GenImage benchmark 也將 degraded image classification 作為重要任務，因為真實網路圖片經常經過壓縮、縮放或模糊處理。(arXiv)
9.4 Calibration Evaluation
報告：
ECE。
NLL。
Reliability diagram。
目的：
確認模型輸出的 fake probability 是否可解釋，而不只是分類結果正確。

10. Explainability and Error Analysis
10.1 Grad-CAM / Attention Map
對 CNN / ResNet 使用 Grad-CAM。
對 CLIP / transformer-based model 使用 attention rollout 或 patch importance map。
觀察模型是否關注主要物體、背景、邊界、紋理或局部噪聲。
CIFAKE 原始研究發現，模型常關注背景細節而非物體本身，因此 explainability 是必要分析，而不是額外加分項。(arXiv)
10.2 Frequency Visualization
對 real 與 fake images 計算：
Average FFT spectrum。
High-frequency energy distribution。
DCT coefficient heatmap。
Residual image visualization。
目的：
分析模型是否利用頻率差異進行判斷，並檢查這些差異是否在 JPEG compression 後仍存在。
10.3 Error Case Taxonomy
錯誤案例分為：
False Positive：真實影像被判為 AI-generated。
可能原因：低解析度、過度壓縮、背景平滑、照片本身有奇怪噪聲。
False Negative：AI-generated 被判為 real。
可能原因：生成器品質高、低階 artifact 被壓縮消除、影像語意合理、背景自然。
Cross-generator failure：訓練模型未見過的 generator 造成 detector 失效。
可能原因：模型學到 generator-specific artifact，而非 universal fake cue。
Compression failure：JPEG 或 resize 後判斷錯誤。
可能原因：壓縮破壞原本的 spectral artifact，或模型本身依賴壓縮差異。

11. Expected Results
預期 Simple CNN 在 CIFAKE 上可達到不錯 accuracy，但在 unseen generator 與 degraded images 上表現下降。ResNet 可能提升 in-distribution performance，但仍可能依賴 dataset-specific shortcuts。CLIP linear probe 預期在 cross-generator generalization 上較穩定，但可能對 32×32 低解析度影像不夠敏感。
本研究提出的 multi-view forensic detector 預期能在以下方面優於 baseline：
在 CIFAKE 上維持接近或高於 ResNet 的分類表現。
在 GenImage unseen generator subset 上有更穩定 AUROC。
在 JPEG compression、blur、resize 後 performance drop 較小。
透過 Grad-CAM 與 frequency visualization 提供較清楚的模型解釋。
透過 calibration metrics 顯示模型信心是否可靠。

12. Expected Contributions
本研究的貢獻不是單純「比較 CNN、VGG、ResNet」，而是建立一個更完整的 AI image forensics proposal：
第一，將任務從封閉式 binary classification 擴展為 generalizable authenticity detection，考慮 unseen generator 與 post-processing robustness。
第二，加入 dataset bias control，避免模型學到 JPEG、resolution 或 preprocessing shortcut，而非真正的 AI-generated artifact。
第三，提出 multi-view detector，結合 semantic-spatial features、frequency-domain artifacts 與 patch-level forensic cues。
第四，設計完整評估流程，包括 in-distribution、cross-generator、robustness、calibration 與 explainability。
第五，透過 error analysis 分析模型失敗原因，而不是只報告 accuracy。
14. Key References
Bird and Lotfi, CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images, 2023.
Wang et al., CNN-Generated Images Are Surprisingly Easy to Spot... for Now, CVPR 2020.
Ojha et al., Towards Universal Fake Image Detectors That Generalize Across Generative Models, CVPR 2023.
Zhu et al., GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image, 2023.
Wang et al., DIRE for Diffusion-Generated Image Detection, ICCV 2023.
Grommelt et al., Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets, 2024.
Yan et al., A Sanity Check for AI-generated Image Detection, ICLR 2025.
Luo et al., LaRE²: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection, CVPR 2024.

