# Med_MLLM
### 数据

##### 模态

###### Text 

###### Radiology

X-ray

 使用X射线穿透物体并记录通过的射线的强度，通过计算机处理得到图像。X射线成像适用于显示骨骼结构等。

CT

通过旋转的X射线源和探测器阵列，以不同角度获取大量的X射线投影数据，再通过计算机处理形成三维图像。CT扫描适用于显示软组织和骨骼结构。

MRI

利用强大的磁场和无害的无线电波，测量人体内水分子的信号，通过计算机处理形成图像。MRI适用于显示软组织结构，如脑、肌肉和关节。

Mammography

乳腺X射线摄影是一种专门用于检测和筛查乳腺疾病的X射线成像技术

Chest X-ray 

胸部X射线是通过用X射线照射胸部来生成影像，用于诊断和评估肺部和心脏的健康状况。

###### Pathology

病理学是研究疾病的科学，通过对组织和细胞的研究来理解疾病的发生和发展。病理学家通过显微镜等工具对生物组织进行分析。

###### Dermatology

皮肤科学是研究皮肤及其疾病的医学专业。皮肤科医生使用临床检查和显微镜来诊断和治疗皮肤问题。

###### Genomics(Variant Calling)

基因组学是研究生物体基因组结构和功能的科学。变异调用是基因组学中的一项技术，用于识别和分析基因组中的变异

###### OCT

光学相干层析成像,是一种医学成像技术，利用光学原理来获取高分辨率的体内组织结构图像。它特别适用于眼科领域，用于检查眼睛内部的微观结构。





##### 数据

| 数据集                           | 输入模态                   | 输出模态 | 类型                                   | 领域               | 规模    | 语言   | 数据来源                                                     | 可用性 | 备注                                                       |
| -------------------------------- | -------------------------- | -------- | -------------------------------------- | ------------------ | ------- | ------ | ------------------------------------------------------------ | ------ | ---------------------------------------------------------- |
| MIMIC-III                        | Text                       | Text     | 重症患者临床信息                       | 医疗问答           | 10GB    | EN     | [Data](https://physionet.org/content/mimic3-carevue/1.4/)    | ✓      |                                                            |
| CXR-PRO                          | Text,X-ray                 | Text     | 胸部X射线报告                          | 医疗报告总结       | 4.6T    | EN     | [Data](https://physionet.org/content/cxr-pro/1.0.0/)         | ✓      |                                                            |
| PAD-UFES-20                      | Text,Pathology             | Text     | 皮肤病变图像分类                       | 图像文本问答       | 3.35GB  | EN     | [Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1)      | ✓      |                                                            |
| VinDr-Mammo                      | Text,X-ray                 | Text     | 乳腺水平类                             | 图像文本问答       | 337.8GB | EN     | [Data](https://www.nature.com/articles/s41597-023-02100-7)   | ✓      |                                                            |
| CBIS-DDSM                        | Text,X-ray                 | Text     | 乳腺X射线和病理信息                    | 图像文本问答       | 5GB     | EN     | [Data](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) | ✓      |                                                            |
| PrecisionFDA Truth Challenge V2  | Genomics                   | VCF      | 基因组测序数据变异调用                 | 医学入学考试问题。 |         | EN     | [Data](https://precision.fda.gov/challenges/10)              | 授权   |                                                            |
| MedQA                            | Text                       | Text     | 医疗问答                               | 文本问答           | 126MB   | EN,CN  | [Data](https://github.com/jind11/MedQA)                      | ✓      |                                                            |
| MedMCQA                          | Text                       | Text     | 医学考试问题                           | 文本问答           | 53M     | EN     | [Data](https://github.com/medmcqa/medmcqa)                   | ✓      |                                                            |
| VQA-RAD                          | Text,X-ray，MRI,CT         | Text     | 放射学图像问题-答案对的数据集          | 图像文本问答       |         | EN     | [Data](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | ✓      |                                                            |
| Path-VQA                         | Text,Pathology             | Text     | 病理图像问答                           | 图像文本问答       | 1.68GB  | EN     | [Data](https://github.com/UCSD-AI4H/PathVQA/tree/master/data) | ✓      |                                                            |
| Slake-VQA                        | CT, MRI, X-Ray             | Text     | 放射学图像问题-答案对的数据集          | 图像文本问答       | 203MB   | EN，CN | [Data](https://www.med-vqa.com/slake/)                       | ✓      |                                                            |
| autoPET                          | CT                         |          | 图像自动分割肿瘤病灶                   | 图像分割           | 420GB   | EN     | [Data](https://autopet.grand-challenge.org/Description/)     | ✓      |                                                            |
| NIH Pancreas-CT Dataset          | CT                         |          | 胰腺数据集                             | 图像分割与问答     | 4.86GB  | EN     | [Data](https://academictorrents.com/details/80ecfefcabede760cdbdf63e38986501f7becd49) | ✓      |                                                            |
| MICCAI                           | MRI,CT                     | Image    | 十个医学分割任务（包括脑、心脏）       | 图像分割与问答     | 50GB    | EN     | [Data](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) | ✓      | [论文](https://www.nature.com/articles/s41467-022-30695-9) |
| Chinese medical dialogue         | Text                       | Text     | 中文医疗问答数据集                     | 医疗问答           | 358MB   | CN     | [Data](https://github.com/Toyhom/Chinese-medical-dialogue-data) | ✓      | [简介](https://github.com/beamandrew/medical-data)         |
| ChatDoctor                       | Text                       | Text     | 医疗问答数据集                         | 医疗问答           |         | EN     | [Data](https://github.com/Kent0n-Li/ChatDoctor)              | ✓      |                                                            |
| OpenGPT                          | Text                       | Text     | OpenGPT生成医疗对话                    | 医疗问答           |         | EN     | [Data](https://github.com/CogStack/opengpt)                  | ✓      |                                                            |
| Segmentation of OCT images (AMD) | OCT                        |          | 年龄相关得黄斑变性数据集               | 图像文本问答       | 18.08GB | EN     | [Data](https://www.kaggle.com/datasets/paultimothymooney/farsiu-2014) | ✓      |                                                            |
| Retinal OCT Images               | OCT                        |          | 糖尿病黄斑水肿                         | 图像文本问答       | 5.81GB  | EN     | [Data](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) | ✓      |                                                            |
| Segmentation of OCT images       | OCT                        |          | OCT图像分割                            | 图像分割           | 203.6MB | EN     | [Data](https://www.kaggle.com/datasets/paultimothymooney/chiu-2015) | ✓      |                                                            |
| BraTS                            | MRI                        |          | 脑肿瘤分割                             | 图像分割与问答     |         | EN     | [Data](http://braintumorsegmentation.org/)                   | 授权   | [论文](https://arxiv.org/pdf/2107.02314.pdf)               |
| TCGA-LUAD                        | CT                         |          | 肺癌CT数据集                           | 图像分割、目标识别 | 18.3GB  | EN     | [Data](http://dataju.cn/Dataju/web/datasetInstanceDetail/291) | 授权   |                                                            |
| 医学影像数据集集合               | MRI，CT etc.               |          | 人体各部位医学影像数据                 | 图像文本问答       |         | EN、CN | [Data](https://github.com/linhandev/dataset)                 | ✓      |                                                            |
| MICCAI CHALLENAGE                | MRI,CT,X-RAY,Pathology     |          | MICCAI CHALLENGES国际竞赛              |                    |         | EN     | [Data](https://conferences.miccai.org/2023/en/MICCAI2023-CHALLENGES.html) | 授权   |                                                            |
| ISBI                             | Text,MRI,CT,X-ray etc.     |          | 医学图像分析挑战                       | 图像文本问答       |         | EN     | [Data](https://grand-challenge.org/challenges/)              | 授权   |                                                            |
| MedMNIST v2                      | MRI,CI,X-ray Pathologyetc. |          | 物医学图像分类的综合性大规模基准数据集 |                    |         |        | [Data](https://medmnist.com/)                                | ✓      |                                                            |
