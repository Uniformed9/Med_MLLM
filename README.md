# Med_MLLM
# 模态

 Text 

Radiology

MRI

Pathology

Mammography

Chest X-ray 

Dermatology

Genomics(Variant Calling)

# **功能场景**

1 医疗问答、诊断支持和医学知识等信息服务

dataset:MedQA 、MedMCQA、PubMedQA

2 **各种医疗报告生成,总结**

从胸片解析报告所见，再生成相应胸片报告结论，完成从读片到写报告的整个流程。

![img](https://exijcfggl6p.feishu.cn/space/api/box/stream/download/asynccode/?code=Y2E5M2JjYTEwZDg5NzgyMmYzM2M3NmE2MThjZTI2M2JfNHNuRWhFOG9iZThxb2NTSVA0RW5MNHNmSG5jZkJNeHJfVG9rZW46RDR0emIxM0Eyb3dxWnh4VW5GQmNJd1BTbmxkXzE3MDA3MjAxNjY6MTcwMDcyMzc2Nl9WNA)

3 **药品助手**

基于药品知识库进行药品知识问答，进一步解决大模型幻觉问题，产生更精确答案。

![img](https://exijcfggl6p.feishu.cn/space/api/box/stream/download/asynccode/?code=ODRkNDFkNDBjNWVhM2EwMjMwMDlhYmU4NzExNGFhNjNfa1JUQ05DcHRMVUQzbjVSQ0xMS2JXZlBCRW90MFJpTFBfVG9rZW46WFpUOGJ1RldXb2RKeFZ4SzRtbGNkbmtZbkdNXzE3MDA3MjAxNjY6MTcwMDcyMzc2Nl9WNA)

4 **结构化输出**

根据用户需求进行推理回答并按结构化 json 格式返回结果

![img](https://exijcfggl6p.feishu.cn/space/api/box/stream/download/asynccode/?code=MjVlYTQ4YzljYWVlMmQ5MzNlZjU2NTE2NTg4NzNjZjJfYlpwZ3pXYTk5aUZFenliVzVDbmhuS2ZDNnR6TUlDamhfVG9rZW46VVd3a2JjTHo0b1psZ0h4eWtKYmNGN3lzbkNnXzE3MDA3MjAxNjY6MTcwMDcyMzc2Nl9WNA)

## **MultiModel**

5 生物医学图像开放性研究问题的视觉-语言对话助手

![img](https://exijcfggl6p.feishu.cn/space/api/box/stream/download/asynccode/?code=OWQ4YjEwM2RmZTViZjExZjU5MTE3MzJkODMxNTg3MGZfZFl3QlBobVN3d1lFNWNFT3Q2S1JMTFVjVEtwYlFtcTdfVG9rZW46UTlHcWJYNzdhb3NrSm14MHlHbGNUNGRFbm9jXzE3MDA3MjAxNjY6MTcwMDcyMzc2Nl9WNA)

6 图像分类

![img](https://exijcfggl6p.feishu.cn/space/api/box/stream/download/asynccode/?code=MTU0MGE2MWY5NGFkMDIyNDg0YmUwNDYyYTExM2E0ODRfeXljM01hdXpNWE95WHhaVnlhanRXWG44ZzB2djhibFBfVG9rZW46SVphS2JwUHo0b3F3TFl4Wjl5b2M2TndQbklmXzE3MDA3MjAxNjY6MTcwMDcyMzc2Nl9WNA)

6 医学图像分割

## 场景实现模型

**[WiNGPT2](https://github.com/winninghealth/WiNGPT2)**

**[明医 (MING)](https://github.com/MediaBrain-SJTU/MING)**

[LLaVA-Med](https://github.com/microsoft/LLaVA-Med)

## Downstream tasks

1 Medical Question Answering  

2 Medical Visual Question Answering

3 Medical Image Classification 

4 Radiology Report Summarization

5 Radiology Report Generation

6 Genomic Variant Calling

7 Breast Cancer Detection in Mammography

8 Neurological Disorder Identification in Medical Image

9 Lung Disease Classification in Chest X-rays

# 数据

### 通用数据

***text***

1 首选GPT4产生的数据，相对质量高，目前开源的数据主要有 sharegpt_zh、alpaca-zh、wizard-70k等 2 其他一些开源数据，有人工整理和gpt3.5产生的一些数据

***image***

imagenet

### 医疗数据

![img](https://exijcfggl6p.feishu.cn/space/api/box/stream/download/asynccode/?code=ZGJjMWIzYjEzMGQ0ZmM2ZmY4NmU5MTIzMThhNmNmM2FfT3VGeVZLalhYOGdTUkxya29tc1FudWl1Nk9tR2VNbFlfVG9rZW46SzFaemJycm45b2VkenR4eFRMSGNxUFRibjBmXzE3MDA3MjAxNjY6MTcwMDcyMzc2Nl9WNA)

# **Dataset**

## **[明医 (MING)](https://github.com/MediaBrain-SJTU/MING)**

[model](https://huggingface.co/BlueZeros/MING-7B)

![img](https://exijcfggl6p.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjhjN2VjMjkxOWJkNWRjYzA3NTI5Zjg3NWViOTczMmJfUFNHNG9qMXdhV0tVZFFncDhXbUhVQWlCWGU0NGhHekFfVG9rZW46V0ZtY2JnbkFPb0haVXV4MklkeGNIek1VbjU3XzE3MDA3MjAxNjY6MTcwMDcyMzc2Nl9WNA)

## 中文医疗问答数据集

[Data](https://github.com/Toyhom/Chinese-medical-dialogue-data)(open source)

## 中文医学知识库

[Data](https://github.com/king-yyf/CMeKG_tools)(open source),[related_model](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)

## **ChatDoctor Dataset**

[Data and Model](https://github.com/donote/llm-medical-data)

## Med-Palm m**(****[web](https://github.com/kyegomez/Med-PaLM)****)**

![img](https://exijcfggl6p.feishu.cn/space/api/box/stream/download/asynccode/?code=OGU4OTAyNDk4ZTEzNjAwY2IwM2YwNTEzMjI1YTUwMDdfVlc2b1lUWG16OTZsNTdSS0R0TlZpV2IyUHZpd1VYeXBfVG9rZW46TjZMVGJKMHlpb1dIdXB4aUNGM2NCcmRubnBoXzE3MDA3MjAxNjY6MTcwMDcyMzc2Nl9WNA)

### MIMIC-III  

该数据集是一个包含在重症监护病房中的患者医疗记录的大型公开数据库。包括与生命体征，药物，实验室测量值，医嘱，手术代码，诊断代码，影像报告，住院时间，生存数据等。

[Data](https://physionet.org/content/mimic3-carevue/1.4/)(Open source),[description](https://blog.csdn.net/qq_43787862/article/details/105028846)，

### MIMIC-CXR

主要是胸部X光片，包含了227835次影像学检查获得的377110张图像（总大小4.6T）

[Data](https://physionet.org/content/cxr-pro/1.0.0/)(Open source)

**PhysioNet（****[web](https://physionet.org/)****）**

1 临床数据库

临床数据库除了MIMIC-III外，还有eICU Collaborative Research Database, Paediatric Intersive Care Database以及过去的MIMIC数据。和MIMIC不同，eICU是一个多中心的ICU数据集，数据采集自飞利浦eICU系统。包含超过200000次入院病例，数据内容包括生理体征测量、护理计划文档、疾病严重程度评估、诊断信息和治疗信息。

2 波形数据库

波形数据库由七部分组成，包括多参数数据库（Mutli-Parameter Databases）、ECG数据集、心博间隔数据集（Interbeat (RR) Interval Databases）、其它心血管数据库、步态和平衡数据库（Gait and Balance Databases）、神经电和肌电数据库（Neuroelectric and Myoelectric Databases）、综合数据库（Synthetic Databases）。多参数数据库就包括了十多个数据集，主要是体征检测数据，有ECG、血压、呼吸等。

3 图像数据库

图像数据库包括MIMIC-CXR Database和Samples of MR Images两个数据集，前一个数据集主要是胸部X光片，包含了227835次影像学检查获得的377110张图像（总大小4.6T），关于该数据集的详细描述最近在Scientific Data上发表。后一个数据集是磁共振血管造影图像，这个数据集发布于2001年，数据量也只有70M。

### **PAD-UFES-20**

内容：由 2,298 张皮肤病变的临床图像组成，包括六种不同类型的皮肤病变。（大小3.35GB）

任务：进行 6 类别的分类任务，使用皮肤病变图像和关联的临床文本特征作为多模态输入。

[Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1)(Open source),[Usage](https://github.com/labcin-ufes/PAD-UFES-20/tree/master)

### VinDr-Mammo

内容： 包括 5,000 例乳腺X射线成像研究，总共有 20,000 张灰度图像，具有广泛的乳腺水平评估和病变级别注释。(大小：337.8GB)

任务： 进行乳腺水平的 5 类别 BI-RADS 分类任务。

[Data](https://www.nature.com/articles/s41597-023-02100-7)(Open source), 

### CBIS-DDSM

内容： 是数字数据库的筛选乳腺影像子集，包含 2,620 个病例。

任务： 进行乳腺癌分类，包括异常病变的 3 类别分类任务。

[Data](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)(Open source)

### PrecisionFDA Truth Challenge V2

内容： 用于基因组变异检测的挑战，使用了 NIST 提供的 HG002 样本。

任务： 进行基因组变异的三类别分类任务

[Data](https://precision.fda.gov/challenges/10)(Open source)

### **MedQA**

[Data](https://github.com/jind11/MedQA)(Open source ) [Paper](https://arxiv.org/abs/2009.13081)

### **MedMCQA**

[Data](https://github.com/medmcqa/medmcqa)

### VQA-RAD

内容：VQA-RAD 放射学图像的视觉问答数据集，包括放射学图像,临床医生创建和验证的问题和回答

任务：

[Data](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)(Open source),

### Path-VQA

视觉问答(VQA)数据集

[Data](https://github.com/UCSD-AI4H/PathVQA/tree/master/data)(Open source),[Paper](https://arxiv.org/abs/2003.10286)

### Slake-VQA

[Data](https://github.com/UCSD-AI4H/PathVQA/tree/master/data)(Open source),[Paper](https://arxiv.org/abs/2003.10286)

## **[LLaVA-Med](https://github.com/microsoft/LLaVA-Med)****(****[paper](https://arxiv.org/abs/2306.00890)****)**

在通用领域的多模态大模型LLaVA下通过指令微调扩展到生物医学领域得到

![img](https://exijcfggl6p.feishu.cn/space/api/box/stream/download/asynccode/?code=NjZiYWExMDY1ZDk5ZjU1MjJhZjRlNjg0OGRiY2IzM2NfZDFKbmFNejQwYXlpbnBkSmxIUFI1MnE4d3g2R0dtMEJfVG9rZW46UEUzeGJnbXBvb1lKOEh4RlBMV2NGdGlPbnJkXzE3MDA3MjAxNjY6MTcwMDcyMzc2Nl9WNA)

*The data statistics of biomedical multimodal instruction-following data: (a,b) The root verb-noun pairs of instruction and responses, where the inner circle of the plot represents the root verb of the output response, and the outer circle represents the direct nouns. (c) The distribution of images and* *QA* *pairs on the five domains, one image is shown per domain.*

[Data and Model](https://github.com/microsoft/LLaVA-Med)

### PMC-15M(not open source)

PubMed Central 的生物医学研究文章中提取的1500万对图标题

### **Instruct Data Generation**

Use openAI

## biomedclip（[paper](https://arxiv.org/pdf/2303.00915.pdf)）

提出一种针对生物医学领域的大规模领域特定预训练方法，名为BiomedCLIP。该方法使用了比现有生物医学图像-文本数据集大两个数量级的数据集进行训练，并进行了领域特定的自适应。

## PubMedBERT（[paper](https://arxiv.org/abs/1901.08746)）

a pre-trained biomedical language representation model for biomedical text mining

[model](https://github.com/dmis-lab/biobert)

## Multimodal LLMs for health grounded in individual-specific data([paper](https://arxiv.org/abs/2307.09018))

[Data](https://www.ukbiobank.ac.uk/)

## Openmed

### Endo-Fm

**Foundation Model for** **Endoscopy** **Video Analysis**

- Colonoscopic [[original paper\]](https://ieeexplore.ieee.org/abstract/document/7442848) [[original dataset\]](http://www.depeca.uah.es/colonoscopy_dataset/) [[our preprocessed dataset\]](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155167044_link_cuhk_edu_hk/EjX1xmuzLxhDgC2XFOuQm6YBymcSx0kcKRK0WJ5aLeZkZg?e=eaWcGW)
- SUN-SEG [[original paper\]](https://link.springer.com/article/10.1007/s11633-022-1371-y) [[original dataset\]](https://github.com/GewelsJI/VPS/blob/main/docs/DATA_PREPARATION.md)
- LPPolypVideo [[original paper\]](https://link.springer.com/chapter/10.1007/978-3-030-87240-3_37) [[original dataset\]](https://github.com/dashishi/LDPolypVideo-Benchmark) [[our preprocessed dataset\]](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155167044_link_cuhk_edu_hk/EqyUhxD1a_JEmkJBzY3axqkBYRRQsJqgmF5p-pgh0LUQSg?e=vi9FU0)
- Hyper-Kvasir [[original paper\]](https://www.nature.com/articles/s41597-020-00622-y) [[original dataset\]](https://datasets.simula.no/hyper-kvasir/) [[our preprocessed dataset\]](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155167044_link_cuhk_edu_hk/EoO0sysq_URMq_cm5P-R-B4BqBDoXIsfL3NlupsBZyfW3A?e=VBDcwc)
- Kvasir-Capsule [[original paper\]](https://www.nature.com/articles/s41597-021-00920-z) [[original dataset\]](https://datasets.simula.no/kvasir-capsule/) [[our preprocessed dataset\]](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155167044_link_cuhk_edu_hk/EuqOhvyl9O5OggzsMIh3Xq4B1YIUZFNe25MnWNp3WCk1KQ?e=QApSVj)
- CholecTriplet [[original paper\]](https://www.sciencedirect.com/science/article/pii/S1361841522000846) [[original dataset\]](https://cholectriplet2021.grand-challenge.org/) [[our preprocessed dataset\]](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155167044_link_cuhk_edu_hk/EgELubRL21ZMlthpwHIJyZgB7cx9yTbjJoWuZ14gyhK0Qw?e=ahZAcI)
- Our Private [[our preprocessed dataset\]](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155167044_link_cuhk_edu_hk/EmJYfUuzesNFjenQwnZe3osB2FSdKnvGSLlp87uhDTt1Ow?e=xoCEbi)

### MIS-FM

Medical Image Segmentation Foundation Model

code：https://github.com/openmedlab/MIS-FM

paper：https://arxiv.org/pdf/2306.16925.pdf

model：https://drive.google.com/file/d/1jQc-2hhsp3EyZj54_KEJte85diUtW8Fg/view?usp=sharing

### STU-NET

用于医学图像分割的，Scalable and Transferable

### **[Pre-training](https://github.com/openmedlab/STU-Net#pre-training)**

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

### **[Fine-tuning](https://github.com/openmedlab/STU-Net#fine-tuning)**

- [FLARE22](https://flare22.grand-challenge.org/)（这个被锁了）
- [AMOS22](https://amos22.grand-challenge.org/Home/) （这个也被锁了）
- [AutoPET22](https://autopet.grand-challenge.org/)

## Other Data

### Pubmed

[Data](https://pubmed.ncbi.nlm.nih.gov/?term=medical)

# Data Community

1 **[和鲸](https://www.heywhale.com/home/global?search=医学)**

2 **[天池](https://tianchi.aliyun.com/cblue)**

[推荐数据集介绍](https://zhuanlan.zhihu.com/p/418116865)

3 **[Data Science Central](https://www.datasciencecentral.com/?s=medical+dataset)**

4 **[Data Tau](https://dzone.com/search?page=1)**

5 **[KDnuggets](https://www.kdnuggets.com/?s=medical)**

6 **[Medium](https://medium.com/search?q=medical+dataset)**

7 **[Cross Validated](https://stats.stackexchange.com/search?q=medical+dataset)**
