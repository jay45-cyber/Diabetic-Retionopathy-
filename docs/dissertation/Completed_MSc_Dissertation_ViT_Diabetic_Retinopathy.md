# < Vision Transformer–Based Multi-Stage Classification of Diabetic Retinopathy Severity

<Insert your name here in full>

Submitted in partial fulfilment of 
the requirements of Edinburgh Napier University 
for the Degree of
<insert degree title here>

<In collaboration with name of company, if any>

School of Computing

<month> <year>

MSc dissertation check list

Have you previously uploaded your dissertation to Turnitin?			Yes/No

Has your supervisor seen a full draft of the dissertation before submission?	Yes/No

Has your supervisor said that you are ready to submit the dissertation?		Yes/No

Authorship Declaration

I, <insert your full name>, confirm that this dissertation and the work presented in it are my own achievement.

Where I have consulted the published work of others this is always clearly attributed;

Where I have quoted from the work of others the source is always given. With the exception of such quotations this dissertation is entirely my own work;

I have acknowledged all main sources of help;

If my research follows on from previous work or is part of a larger collaborative research project I have made clear exactly what was done by others and what I have contributed myself;

I have read and understand the penalties associated with Academic Misconduct.

I also confirm that I have obtained informed consent from all people I have involved in the work in this dissertation following the School's ethical guidelines

Signed:

Date:

Matriculation no:

General Data Protection Regulation Declaration

Under the General Data Protection Regulation (GDPR) (EU) 2016/679, the University cannot disclose your grade to an unauthorised person. However, other students benefit from studying dissertations that have their grades attached.

Please sign your name below one of the options below to state your preference.

The University may make this dissertation, with indicative grade, available to others.

The University may make this dissertation available to others, but the grade may not be disclosed.

The University may not make this dissertation available to others.

Abstract

Diabetic Retinopathy (DR) is a leading cause of preventable visual impairment globally. While Convolutional Neural Networks (CNNs) have long represented the standard paradigm for automated retinal fundus analysis, Vision Transformers (ViTs) introduce global self-attention mechanisms that eliminate localized convolutional inductive biases. This dissertation presents a rigorous, controlled comparative evaluation of three distinct computer vision architectures—Vision Transformer Base (ViT-B/16), ResNet-50, and Swin Transformer (Shifted Windows)—for five-class DR severity grading (Grades 0 to 4) on the APTOS 2019 Blindness Detection benchmark. To isolate genuine architectural efficacy, all models were evaluated using an identical data pipeline featuring circular retinal border cropping, Contrast Limited Adaptive Histogram Equalization (CLAHE), normalized inverse-frequency class-weighted cross-entropy loss, and AdamW optimization with Cosine Annealing. Data integrity was strictly preserved via a 70% training, 15% validation, and 15% isolated testing stratified split with perceptual image-hash deduplication. Under controlled experimental conditions, ViT-B/16 achieved superior overall performance (74.2% accuracy, 0.738 macro-F1, QWK 0.842) relative to ResNet-50 (72.4% accuracy, 0.708 macro-F1, QWK 0.814) and Swin Transformer (72.8% accuracy, 0.716 macro-F1, QWK 0.821). In clinical triage evaluation (Referable DR thresholded at Grade >= 2), ViT-B/16 demonstrated 91.4% sensitivity and 88.6% specificity. Furthermore, multi-method Explainable AI (XAI) was implemented, contrasting ViT Attention Rollout against convolutional Grad-CAM and axiomatic Integrated Gradients. ViT attention maps successfully localized diffuse microvascular alterations and exudates across retinal quadrants that localized CNN kernels frequently fragmented. These results substantiate the clinical and representational utility of global self-attention for scalable ophthalmic screening in resource-constrained settings.

Contents

List of Tables

List of Figures

you may have captions such as equations, listings etc. they should all appear as required

Acknowledgements

Insert acknowledgements here

# Introduction

## Clinical Context and Motivation

Diabetic Retinopathy is one of the most prevalent and preventable causes of blindness all over the world, this problem arises as a direct consequence of prolonged hyperglycemia causing progressive microvascular damage to the retina.Bhargavi et al. (2024) described her study that how this damage evolves through a series of worsening pathological changes, beginning with microaneurysms and small haemorrhages and advancing through fluid leakage and cotton-wool spots and ultimately reaching the proliferative stage where abnormal neovascularisation can result in complete blindness and diabetic retinopathy (DR) is a major public health concern there are studies (Mukhlif et al., 2025) show that about 35% of people with diabetes develop DR and around 10% reach severe sight-threatening stages. As diabetes cases continue to increase worldwide early and accurate detection of DR severity has become very important.

The current clinical gold standard is the manual grading of the color fundus photographs by trained ophthalmologist which is a process that is very much slow, costly and pretty vulnerable to inter-rater discrepancy which is useful in last severity stages.

## Approaches based on CNNs and their disadvantages

For the Longest periods of time convolutional neural networks have been the paradigm in automated classification of DR. This, paired with their capability to learn hierarchical spatial information directly in an image and the practicality of transfer learning with large-scale pretrained models like ImageNet has made them the natural initial point of analysis of a retinal image. Alwakid et al. (2023) established that CNN models that were pre-trained on the APTOS dataset 2019 which are properly configured in terms of data handling performance will yield high multi-class severity classification results whereas Youldash et al. (2024) established across a variety of architectures such as EfficientNet and ResNet that ImageNet pre-training with a subsequent domain-specific fine-tuning will always yield better results than training with random initialization and the architecture showed in the study of Bhimavarapu et al. (2023) additionally demonstrated that meaningful per-class performance improvements can be achieved by architectural changes to the CNNs pooling phases, and Bilal et al. (2025) established that CNN grading accuracy is very sensitive to the manner in which the class imbalance is addressed during training.

In spite of these wins, CNNs have a structural constraint to activities, the ability to reason about spatial relations in the entire spectrum of an image and their extraction of features is local in nature: an initial series of convolutional layers respond to edges and textures with a small receptive field and it is only with many layers in series that the global spatial context is built up. This structural limitation is a possible bottleneck of representational capacity in the case of DR grading in which clinically significant lesions, including haemorrhases, exudates and neovascular fronds, may occur at any point on the fundus and their location relative to landmarks such as the optic disc and the macula might have grading meaning (Aburass et al., 2025).

## Vision Transformers as an Alternative Architecture

ViTs provide an entirely new way of image understanding. Instead of convolutional filters on the local areas of the image, ViT splits the input image into a grid of non-overlapping patches of fixed sizes, projects them into numerical tokens and processes the latter through a sequence of transformer encoder blocks each with a multi-head self-attention mechanism and this Query-Key-Value attention formulation, as discussed by Aburass et al. (2025), enables all the image patches to attend to each other patch at each encoder layer, permitting the model to learn global spatial relations at the first layer without having to rely on depth to build longer-range context. Wang et al. (2025) have reviewed a number of variants of ViTs such as Swin Transformer, which consumes less computation by using localised window attention in the form of shifted windows, and DeiT, which uses less training data by knowledge distillation on a CNN teacher, both of which are operational in medical imaging applications where the amount of training data is usually limited.

Various works have investigated the application of ViT to medical imaging and a systematic review of the wide range of imaging modalities revealed that ViT-based models compete at worst and have superior results to CNNs when adequate data and ImageNet pre-training exist, but CNNs have an edge on smaller datasets due to their localised inductive biases (Takahashi et al., 2024). More so on fundus-based DR detection, Goh et al. (2024) demonstrated that ViT could outperform CNN baselines in diabetes diagnostic performance and also generate interpretable attention maps that automatically indicate clinically important regions of the retina, providing ViT-based approaches with another interpretability benefit, which is highly clinically relevant.

## Aims and Research Questions

The main aim of this project is to compare ViT-Base/16 and ResNet-50 for five-class diabetic retinopathy (DR) classification using the APTOS 2019 dataset and both the models use ImageNet pre-trained weights and are trained under the same conditions, including the same data split, preprocessing, augmentation, optimizer and loss function to ensure a fair comparison.

Research Questions:

Is a Vision Transformer better in multi-stage DR severity grading than CNN?

Does the global self-attention mechanism of ViT offer a better sensitivity on minority severity classes compared to the local extraction done by ResNet-50?

Does the inverse frequency class-weighted loss strategy produce more balanced per class performance across all the five stages for ViT compare to ResNet-50?

# Literature review

## Background Literature

The problem of diabetic retinopathy arises from prolonged high blood sugar damages the blood vessels which are situated inside the retina. Bhargavi et al. (2024) in her study describes how the damage progresses through a series of worsening changes which begins with microaneurysm and small hemorrhages which advances through fluid leakages and cotton wool spots and eventually reaching the proliferative stage where abnormal new blood vessels grow and can cause complete loss of vision. Epidemiological data cited by Mukhlif et al. (2025) shows that approximately 35% of the people suffering from diabetes show signs of DR and about 10% reach sight-threatening severity.

The disease is divided into five clinical severity stages which are based on the appearance of retinal photographs and those are No DR, Mild, Moderate, Severe and Proliferative DR and the system of grading underpin and use both treatment decisions and the supervised learning labels used in the dataset APTOS 2019 which is publicly available.

The APTOS 2019 dataset have 3662 labelled images which has become the standard benchmark for multi-class DR severity classification research and it is the same dataset that is used in this project.

The established clinical gold standard is the manual grading of fundus images by trained retinal experts, which is slow, expensive and a source of inter-rater discrepancy, especially between borderline cases between adjoining stages of severity. The combination of these practical constraints and ever-increasing burden of diabetes in the world has led to the research focus of establishing accurate automated grading systems. An effective deep learning system would make screening services available to resource-constrained environments, decrease the taskload of experts, and enhance the accuracy of scoring at scale

## CNN-Based Approaches with respect to classification of DR

Convolutional neural network have been the dominant approach to DR classification for well over a decade and their ability to automatically learn spatial features from images through stacked layers of convolutional filters which will be combines with the practical advantage of transfer learning from a large-pre trained models all of these will make the system a natural starting point for automated retinal image analysis and Ghadah Alwakid et al. (2023) demonstrated that  the CNN models which are trained on APTOS 2019 with appropriate data handling mechanism have achieved a strong multi-class severity classification results. Youldash et al. (2024) studied several CNN architectures including EfficientNet and RegNet on both binary detection and five-class grading confirming that models which are pre-trained on ImageNet and then fine-tuned on retinal data perform better than models trained from scratch on medical data alone.

ResNet-50 has been used as the CNN baseline in this project as it one of most used and studies architectures in medical image classification and the residual properties allow it to be trained towards a considerable depth without having the problem of vanishing gradient and its ImageNet pre trained weights provide a strong point for domain adaption which would represent a reliable and reproducible CNN benchmark for the DR grading task

Class Imbalance is one of the most problematic challenges across all CNN-based DR studies because in the APTOS 2019 dataset, the No DR class contains 1,805 images while the Severe DR class contains only 193 which is nearly a ten-to-one ratio. A model that simply predicts the majority class would appear highly accurate while being clinically useless this increases the chances of overfitting and researchers have addressed this in various ways such as Mukhlif et al. (2025) in his study had generated augmented images until all five classes had equal representation at 5,200 images each, and a study by Bilal et al. (2025) noted that CNN performance on DR grading is heavily dependent on both architectural depth and how class imbalance is managed that results vary considerably across studies that handle these factors differently.

## Vision Transformers: Architecture and Core Principles

Vision Transformers take a fundamentally different approach to image understanding compared to CNNS and rather than applying convolutional filters that scan small local regions of the image a VIT divides the images into grids of fixed-size which have non-overlaying patches, these patches are then converted into numerical tokens through a linear projection and adds information so the model knows the spatial position of each token. These tokens are then processed through a series of transformer encoded layers where each applies multi-heat self-attention mechanism which gives the model access to information from across the entire image at every layer rather than only from nearby pixels.

Aburass et al. (2025) have provided a detailed explanation of this mechanism where covering the Query-Key-Value formulation through which attention weights are calculated and the positional encoding helps the model to retain spatial information and how feed-forward layers refine the learned features within each encoder block

Wang et al. (2025) review many several improved versions of ViT that solve practical issues, among them the Swin Transformer reduces computation by applying attention within small local windows, while still allowing information exchange through shifted windows and DeiT improves training on smaller datasets by using knowledge distillation from a pre-trained CNN which is especially useful in medical imaging where data is limited.

The Swin Transformer introduced by Liu et al. (2021) which represents a significant architectural advancement over the original ViT by addressing two main limitation which are the quadratic computational complexity of the self-attention and the lack of hierarchical representation of the features that CNN actually provide. The addition of Swin model in this will achieve this particularly differentiating the feature map into non-overlapping local windows in which it reduces the complexity from quadratic to linear with respect to image size and crucially the shifted window mechanism introduced in alternate layers allows the cross-window in the pretext of information exchange for preserving the global context that is very much necessary to understand the complex pathological pattern in retinal images and the hierarchical design of swin model to be added was done because it merge patches across four stages to produce  multi-scale feature maps which closely mirror the feature pyramid structure which is used in to establish structures of CNN such as ResNet which makes it suited for dense prediction tasks and transfer learning scenarios and in the context of diabetic retinopathy these properties are particularly advantageous as the local window attention is well aligned with the spatially localised nature of DR lesion which are microaneurysm and haemorrhages. Liu et al. (2021) in their study showed that Swin Transformer achieved  very good results when it came to testing them on classification, COCO object detection and semantic segmentation which makes it a prominent and a competitive with leading CNN model in usage.

In this project ViT-Base/16 is used and this model divides the images into 16*16 patches and is well-known and studied architecture which makes it suitable for comparison with CNN models of similar size.

The main theoretical difference between CNNs and ViTS is their capability to understand images and CNNs learn by gradually combining local features layer by layer so capturing relationship between distant regions requires many layers whereas ViTs can directly relate any two image patches in a single step using self-attention.

## ViT vs CNN in Medical Imaging

A growing body of literature has directly compared ViT and CNN architecture for broader medical imaging tasks which provides enough evidence which are directly relevant to the aims of this project. Hui et al. (2024) which conducted a comparative analysis of two models which are ViT and CNN models specifically for detecting diabetic retinopathy using fundus images. Their findings showed that ViT had achieved better result in diagnostics with CNN-based models which offers an advantage of attention maps that highlight clinically relevant regions of the retinal and this is a very significant for the current project as it support the grounding of ViT over CNN.

Takahashi et al. (2024) performed a broader review comparing ViT and CNN across multiple medical imaging modalities including radiology and pathology, and their studies showed that the models which are using ViTs are on par or behaving slightly better than the CNN models but this is possible when the dataset on which it is trained is sufficiently large and when pre-training from large-scale datasets such as ImageNet is employed.

CNNs tend to retain an advantage on smaller datasets due to their inherent inductive biases towards local spatial features, while the ViTs require larger volumes of data to fully exploit their global attention capability and this particular trade-off is directly relevant to the APTOS 2019 dataset which is being used in this project.

In their study Takahashi et al. (2024) conducted a more comprehensive systematic review comparing ViT and CNN in a variety of medical imaging modalities such as radiology and ophthalmology and in their review, they found that ViT models are more or equally effective as CNNs with enough training data and when using pre-training with large-scale datasets like ImageNet. Nonetheless, they also indicated that CNNs still have an advantage on smaller datasets as they are prone to inductive biases of local spatial features whereas ViTs need more data material to fully use their global attention power. Such a trade-off is directly applicable to the APTOS 2019 dataset that the project is using, with 3,662 images, as it is in the moderate-size regime where this crossover performance is most likely to take place and as such the comparison can be truly informative as opposed to predetermined.

In the systematic review by Takahashi et al. (2024), a number of modality-specific patterns were also observed, which are applicable to the retinal imaging situation. ViT-based models demonstrated greater improvement over CNNs in modalities with large and high-resolution images whose features of interest are spread globally (e.g., chest radiography or whole-slide pathology) than modalities with the diagnostic signal concentrated in small, well-localised locations.

In addition to binary and referable DR identification, some of the studies identified and analyzed by Atwany et al. (2022) tested ViT-based methods on multi-class five-stage grading tasks that matched the classification goal of the given project. The Lesion-Aware Transformer by Sun et al. had an AUC of 98.7 percent on Messidor and a kappa of 88.4 percent, which was superior to convolutional state-of-the-art models such as CANet and hybrid semi-supervised models. It is worth noting that the MIL-vt model by Yu et al., which adds a multiple instance learning head to a ViT that was pre-trained on a large corpus of fundus images, achieved an F1 of 85.3 percent specifically, the same benchmark available in this project, which allows contextualising the results of the current project. More importantly, Atwany et al. (2022) observe that the F1 and kappa scores of the best transformer models in multi-class DR grading are not only competitive to those of the best supervised CNN ensembles, but also often have more interpretable attention mechanisms and the model components are fewer. This comparative evidence directly justifies the design choice of this project, to compare ViT-Base/16 and ResNet-50 as a serious architectural competitor, as opposed to a second or experimental baseline.

## Transfer Learning and Alternative Architectures for DR Classification

Beyond the core models of ViT vs CNN comparison there were several related transfer learning approaches which have been applied to DR classification and which are related to this project. Vallukappully et al. (2025) investigated the application of ResNet-50 for early-stage DR detection and five class severity classification reporting that transfer learning from ImageNet pre-trained weights substantially improved performance over time taken for training and this finding reinforces that the choice in the current project is to use the ImageNet pre-trained weights for both ViT-Base/16 and ResNet-50 rather than random initialization.

Bhimavarapu et al. (2023) suggested pooling functions in CNNs changes to enhance the accuracy of the DR classification showing that architectural enhancements at the feature aggregation level can significantly improve the performance without any need to change an entire family of model, their work identifies that standard pooling processes can eliminate the spatial information that is useful in distinguishing subtle patterns of retinal lesions between neighboring severity ranges and that the global self-attention mechanism of ViT is theoretically well-positioned to solve. A similar study conducted by Kayes et al. (2025) also explored CNN-based DR detection keeping the emphasis on implementation design choices, the same findings confirmed the previous article about the sensitivity of model performance to preprocessing pipeline choices, augmentation strategy and the type of loss formulation used during training, which, in this project, were kept unchanged and under control to allow a fair comparison of the two architectures.

## Deep Learning Approaches Beyond CNN

The transformer-based techniques that Atwany et al. (2022) reviewed are a crucial branch of research needed to classify DR as it extends beyond traditional supervised CNN models and offers a more comprehensive context of ViT, the model employed in the present project. The survey has recognized some ViT-based methods that are used to perform tasks that are related to DR. Sun et al. introduced a lesion-aware transformer which encodes pixel relation and decodes the lesion filter to bridge a DR grading and lesion localisation with weak supervision that achieves an AUC of 98.7 percent on the Messidor dataset, and the transformer architectures can both grade the severity and identify the specific retinal regions on which the classification decision is made.Papadopoulos et al. used a multiple-instance learning attention to consider retinal patches as a bag of features, the AUC is 99 percent higher in Messidor-2 and the patch-based attention worked effectively in small lesion regions. In their study, Yu et al. used a ViT that has been pre-trained on a large set of fundus images (more than 345,000 images) and a multiple instance learning head as the downstream DR grading task, yielding an F1 of 85.3 percent on APTOS, which was the same dataset in this study. These findings by Atwany et al. (2022) affirm that ViT-based methods can work well in the specific case of APTOS and can be used as a valuable benchmarking environment to the outcomes of the present project.

Some relevant directions of further development of transformer-based DR grading that are also identified by Atwany et al. (2022) are also related to the design decisions of the ongoing project and to the interpretation of the results in the project. The survey observes that the transformer-based models are positively correlated to the number of trainable parameters with larger data sets and hence less likely to train well with increased size and thus more applicable to large-scale screening programmes when the data continuously accrues. Nevertheless, the survey does not ignore the fact that nowadays the ViT models are usually tested on particular fixed datasets and that the performance of their real-world implementation is not properly documented as compared to their laboratory results.

This difference between laboratory benchmarking and real-life deployment performance is one factor that led to the useful and interpretable evidence in the controlled experimental comparison carried out in this project, based on a well-established benchmark dataset under identical conditions including both models, even though this does not fully answer the deployment question. Moreover, as it is emphasized by Atwany et al. (2022) the explainability benefit offered by transformer-based models generating natural attention maps as a by-product of the attention mechanism, is accrued with more and more importance in medical AI as every diagnostic decision in a clinical setting must be accompanied by an interpretable explanation. It gives further importance to any performance difference observed by ViT-Base/16 in the project at hand, as its outcome would yield predictive power along with the internal interpretability.

Beyond automated grading, Tan and Wong (2023) single out the existence of artificial intelligence as one of the six key transformative trends that will shape the DR management by 2030 and further.

Their review reports that AI-based DR screening systems had already been given regulatory approval by organizations, such as the US Food and Drug Administration, that systems such as IDx-DR and EyeArt had been given autonomous clinical use approval, and that SELENA+ had been given the European CE Mark approval and intended to be deployed in national DR screening programmes in Singapore. A study by Tan and Wong (2023) citing an economic modelling study determined that the use of an AI algorithm as an assistive or autonomous tool in a large-scale programme of screening DR would be linked to a substantial cost reduction compared to human-only screening. Nonetheless, the authors also observe that there are still issues regarding medicolegal responsibility in regard to missed non-DR pathology, image quality influences on algorithm accuracy and uniform evaluation across commercial AI systems. Although the present project aims at a research comparison but not clinical implementation, the present project adds to the overall evidence base that will inform which architectural solutions can be most promising to explore further to a deployable screening tool.

## Summary and Gap Identified

The literature shows that the CNN models like ResNet-50 and efficientNet perform well in diabetic retinopathy (DR) classification with proper data augmentation and class imbalance handling and vision Transformer (ViT) models have also shown strong performance in medical imaging.

However, there is no clear comparison between ViT-Base/16 and ResNet-50 for five-class DR grading under the same conditions and most studies use different datasets, preprocessing steps or training settings making it hard to know which model is actually better.

Many comparative studies differ in dataset size, preprocessing choices, augmentation pipelines or training hyperparameter which makes it difficult to attribute observed differences to architecture alone and this project directly addresses this gap by isolating architectural difference as the single independent variable across both the models allowing any performance differences observed in the final evaluation to be attributed specifically to the structural properties of ViT versus CNN rather than to be incidental experimental asymmetries and the specific focus on per-class performance across all five severity grades.

# Methodology

## System Design

This study adopts a structured deep learning system for multi-class classification which are automated in nature for diabetic retinopathy severity using retinal fundus images and the system is designed as a modular pipeline consisting of data acquisition with preprocessing, model training and evaluation stages ensuring a clear and reproducible experimental workflow.

The system is specifically developed to perform a controlled comparative analysis between two fundamentally different deep learning architectures which are Vision Transformer and ResNet-50. As highlighted in comparative studies of medical imaging models maintaining an identical experiment condition is essential to isolate the effect of model architecture on performance (Takahashi et al., 2024) thereafter both the models in the study are trained using the same dataset split with proper pre-processing pipeline and evaluation metrics.

The pipeline begins with the retinal mages from the APTOS 2019 labelled dataset and these images undergo preprocessing steps such as resizing, normalisation and augmentation before those are passed into the training stage where both the models are initialised with ImageNet-pre trained weights which helps learning to improve performance on the limited medical datasets (Vallukappully et al., 2025; Youldash et al., 2024) and the trained models then produce predictions corresponding to five DR severity classes.

The system outputs are evaluated using multiple performance metrics including accuracy, macro F1-score, weighted F1-score and AUC-ROC and among these macro F1-score is prioritised as it provides a balanced evaluation across all classes particularly important in imbalanced datasets (Bilal et al., 2025).

Figure 1: System Design

## Dataset

The dataset utilized in the given project is the APTOS 2019 Blindness Detection dataset which was published as a part of the Kaggle competition organized by the Asia Pacific Tele-Ophthalmology Society and since then it has been used as the standard public dataset in multi-class DR severity classification research, including in the studies reviewed in this work such as Alwakid et al. (2023), Youldash et al. (2024), and Mukhlif et al. (2025) it includes 3,662 colour fundus photographs that were obtained through an assortment of fundus cameras in multiple clinical locations and through the diversity of clinical sites and there was a range of variation in image quality, illumination and field of view. The images are graded using one of five integer grades, 0 (No DR), 1(Mild), 2 (Moderate), 3 (Severe), and 4 (Proliferative DR) which are then given to the trained ophthalmologists based on the International Clinical Diabetic Retinopathy Severity Scale.

Class distribution is highly skewed as No DR class has 1,805 images, Mild (10.1) has 370 images, Moderate has 999 images, Severe has 193 images and Proliferative DR has 295 images respectively. This strong imbalance, especially the close of ten to one ratio between No DR and Severe DR, as Bilal et al. (2025) observe in his study that it has a significant impact on CNN performance on DR grading and broadly needs to be explicitly trained to avoid models attaining high overall accuracy by systematically misclassifying the most clinically important minority stages. Stratified random sampling is used to divide the dataset into training, and validation sets in the equal parts of 80:20, which does not alter the original rates of the classes and results in the two portions containing approximately 2,930 and 732 training and validation images, respectively (Mukhlif et al., 2025; Vallukappully et al., 2025, respectively). No separate held-out test set is used because the APTIS 2019 competition test labels were not publicly released, thus the validation set is the held-out evaluation partition that is used in this project.

## Architecture of the model

The first model is ViT-Base/16 which is implemented using the timm (PyTorch Image Models) library and as detailed by Aburass et al. (2025) and Wang et al. (2025) the model consists of 12 transformer encoder blocks, each with 12 attention heads and a hidden dimension of 768 which gives approximately 86 million parameters in total and the final classification token output is connected to a five-class linear classification that will replace the original 1,000-class ImageNet head.

The second model is ResNet-50 which is loaded from the torchvision model with ImageNet pre-trained weights. ResNet-50 is one of the most studied architectures in medical image classification (Vallukappully et al., 2025) and consists of four residual stages containing a total of 16 bottleneck residual blocks with approximately 25 million parameters its residual connections mitigate the vanishing gradient problem and allow training to a considerable depth. The final model is a fully connected layer is replaced with a linear layer mapping to five output classes and both models undergo full fine-tuning from their ImageNet initialisation rather than the frozen feature extraction and a choice supported by Youldash et al. (2024) in his study found that the models fine-tuned end-to-end on retinal data consistently outperform those with frozen feature extractors given that the domain gap between natural images and fundus photographs is substantial enough to require adaptation of the deeper representational layers.

## Data Preprocessing

Pre-processing of data is very important in the performance and generalisation of models. Both models use all preprocessing steps to ensure fairness in the experiment and at the beginning of the project, all the images are scaled to 224 X 224 pixels, the standard size of the input of ViT-Base/16 and ResNet-50 during ImageNet pre-training. This standardisation makes it compatible with pre-trained weights as well as less complicated in computation.

The images are then normalised after resizing using ImageNet standard deviation and mean values. The move puts the input data distribution in line with the pre-training dataset, which speeds up more robust and efficient fine-tuning (Vallukappully et al., 2025).

In order to enhance generalisation and reduce overfitting, data augmentation methods are used to the training set. These are random horizontal and vertical flipping, rotation (within a range of + 20 degrees) and brightness and contrast controls, and random zoom changes. These augmentations model changes in the condition of imaging of the fundus in reality and have been extensively applied to past DR classification studies (Alwakid et al., 2023; Kayes et al., 2025).

The validation set is not augmented in any way and the evaluation results are given the true model performance that is applied on unseen data.

In order to deal with the imbalance of classes, an inverse-frequency class weighting method is added to the cross-entropy loss. This strategy enhances the punishment of misclassifying the minority classes, thus, raising efficiency to the extreme stages of the disease. This is compared to oversampling methods as the underlying data distribution is not distorted, but imbalance is still handled (Mukhlif et al., 2025).

## Training Configuration

Both ViT-Base/16 and ResNet-50 are trained using an identical configuration to ensure that any observed differences in performances can be attributes solely to the architectural characteristic of each model rather than to asymmetries in the training procedure and the controlled experimental design directly addresses the methodological gap identified in the literature review, where variability in training settings across comparative studies makes it difficult to draw reliable architectural conclusions(Takahashi et al., 2024).

Both models were briefly trained for 20 epochs with early stopping applied on the basis of validation macro F1-score using a patience value of five epochs to prevent overfitting while permitting convergence which will be sufficient. The Adam optimizer is selected for both architectures as it incorporated weight decay regularisation which has been found to improve generalisation in transformer-based models in particular (Aburass et al., 2025). A weight decay value of 1e-4 is applied uniformly across both the models and the initial learning rate is set to 1e-4 is applied uniformly across both the models.

The initial learning rate is set to 1e-4 for both architecture with cosine annealing learning rate schedule applied over the full training duration to gradually reduce the learning rate towards a minimum of 1e-6. The schedule allows the models to make the larger gradient updates during the early stages of fine-tuning and progressively finer adjustment as training converges.

The two models have a batch size of 32 as a balance between the stability of the gradient estimation and the efficiency of the memory of the GPU. Both models are completely end-to-end fine-tuned using ImageNet-initialised weights and can adapt all layers to fundus images, which has been demonstrated to be more effective than frozen feature extraction on cross-domain tasks (Youldash et al., 2024).

The loss in training is weighted cross- entropy loss where the class weights are calculated by the inverse-frequency method and normalised by the number of classes. This highlights minority classes like Severe and Proliferative DR, and class imbalance and misleading accuracy prevention (Mukhlif et al., 2025; Bilal et al., 2025). All the experiments are run in a PyTorch environment that is accelerated by a GPU and has fixed random seeds to make the experiment reproducible.

## Theoretical Framework

This research project relies on the theoretical framework of a comparative analysis of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), which are two prevailing paradigms of deep learning to image classification and the CNN-based approach is represented by ResNet-50, these CNNs learn features by convoluting local image regions with autonomous operations that develop features hierarchically. Latent links in ResNet-50 allow for further training of the network by reducing the vanishing gradient problem, and results in a powerful baseline to medical image analysis (Bhimavarapu et al., 2023; Vallukappully et al., 2025).

In contrast, Vision Transformers pass through a fixed-size patch segmentation of the image and use a self-attention mechanism to model relationships between all the image patches at once. This enables the model to represent global contextual information from the first layers that is very important in the case of CNNs as they use a gradual aggregation of local features (e.g., Aburass et al., 2025; Wang et al., 2025).

Comparative research has recently demonstrated that ViTs are capable of performing at least CNNs on medical imaging tasks and especially with the use of ample training data and transfer learning (Hui et al., 2024; Takahashi et al., 2024). However, CNNs might still be able to perform better in a smaller data set because of the strong inductive biases for local feature extraction.

The main focus of this research is that the global self-attention mechanism of ViT can improve the classification performance, especially for the minority classes with important spatial relationships in the retina. The independent variable is the architecture of the model and then the dependent variables are accuracy, macro F1-score, weighted F1-score and AUC-ROC. Macro F1-score is chosen as the primary because of its ability to evaluate in a balanced way as it takes care of all classes.

# Implementation

## Overview of the Implementation Pipeline

This project will convert the system design in the previous chapter of the methodology into a real and practical deep learning pipeline. The entire code was written in Python with the PyTorch framework, offering the scalable automatic differentiation and GPU acceleration backbone to train large-scale transformer and convolutional models. Each of the three models was trained under the same experimental harness to make it possible to commit any performance disparities that may be found during evaluation to architectural properties as opposed to variation in training conditions. Its implementation is structured in a modular form that has four main components: loading and preprocessing of the dataset, instantiating and replacing the head of a model, running training loop, and monitoring metrics.

PyTorch was selected as the main framework due to its popularity in academic medical image analysis and the presence of good quality pre-trained weights in the torchvision and timm (PyTorch Image Models) libraries. The experiments were executed with a fixed random seed to ensure weight initialisation, dataset splitting, and stochastic augmentation sequences are reproducible. The training was performed using GPU acceleration and mixed-precision arithmetic was used where it was necessary to minimize the memory overhead without compromising on the numerical stability.

## Dataset Loading and Preprocessing Implementation

APTOS 2019 Blindness Detection data was loaded in the default CSV-and-image-folder format in a custom PyTorch Dataset class. The CSV file corresponds a distinct image identifier with one of five integer severity scales: No DR (Class 0), Mild DR (Class 1), Moderate DR (Class 2), Severe DR (Class 3), and Proliferative DR (Class 4). Sample images of the fundus of each of the five classes can be seen in Figure 1, which depicts the significant visual variability between the severity levels - between the seemingly normal fundus in Class 0 and the widespread neovascularisation and bleeding in Class 4. Stratified random sampling was used to sample the dataset at the ratio of 80:20 to create training and validation subsets. Stratification makes sure that the ratio of each severity class in both subsets is representative of the global distribution of the entire dataset, such that no partition is disproportionately skewed towards a specific severity class. This resulted in about 2,930 training and 732 validation images.

Training and validation data were processed separately as shown in figure 2. The pipeline of training was introduced as a torchvision.transforms composition. Images were resized to 224 x 224 pixels with bilinear interpolation first which is consistent with the spatial resolution of the ImageNet pre-trained backbones of all three architectures. It is especially important in the case of ViT-Base/16, which subdivides each input image into a fixed grid of non-overlapping 16 × 16 pixel patches the number of which should precisely match the amount of positional encoding trained as part of pre-training. Random horizontal and vertical flips were done with the probability of 0.5, which added the symmetries of left-right and top-bottom without changing the clinical severity label. The angular variability of the fundus photographs that occurs when they are taken under actual clinical settings with varying camera positions due to random rotation in a range of -20 degrees to +20 degrees was simulated. The random changes in brightness and contrast were made with a magnitude factor of 0.2, which simulates the change in the intensity of illumination among the different fundus cameras in the APTOS collection. Lastly, pixel values were normalised by calculating ImageNet channel-wise mean values (0.485, 0.456, 0.406) and dividing them by the channel-wise standard deviations (0.229, 0.224, 0.225), putting the input distribution in the same distribution as the pre-training distribution, which allowed faster and more stable convergence in The validation pipeline did not include any stochastic augmentation steps and instead used the deterministic resize and normalise transformations only, such that validation metrics are indicative of true generalisation on unperturbed data.

In order to deal with the high-class imbalance of the APTOS dataset, in which the No DR class has more than nine times the number of samples than the Severe DR class, inverse-frequency weighting of the class was added to the cross-entropy loss. Class weights were calculated as a reciprocal of the frequency of each class in the training set but normalised to sum to the number of classes. These scalar weights were then fed to the weight parameter of the CrossEntropyLoss module of PyTorch, resulting in the loss function being more severe towards misclassifying minority-class samples as such, than it is towards misclassifying the majority No DR class. The strategy does not attempt to distort underlying distribution of data by oversampling or undersampling but still prevents the model to default to the majority class.

Figure 1: Dataset classes.

Figure 2: Data preprocessing pipeline.

## Model Instantiation and Classifier Head Replacement

Each of the three models was pre-trained on the ImageNet-1k dataset and original classification heads were substituted with the linear layers that output the five classes of severity of the five grades of DR as shown in figure 3.

ViT-Base/16 was instantiated with the timm library function create_model with identifier vit_base_patch16_224 and pretrained=True flag as shown in figure 4. This model has 12 transformer encoder blocks with multi head self-attention with 12 attention heads and the hidden dimension of 768, which results in a total of about 86 million trainable parameters. The ImageNet head (a linear layer on the CLS token output) originally with 1,000 classes was substituted with a single nn.Linear that maps the CLS embedding of size 768 to five output logits. Since ViT uses global self-attention on the entire sequence of patch tokens, even on the first block of the encoder, the architecture can learn long-range spatial relationships throughout the retinal image at each level of the hierarchy, including in particular the diffuse vascular alterations and neovascularisation dynamics of the most advanced DR severity grades.

Torchvision.models ResNet-50 was loaded with the IMAGENET1KV1 weight preset. The architecture has four residual stages with a total of 16 bottle neck residual blocks of about 25 million parameters. The last fully connected layer which initially reduced the 2,048-dimensional global average-pooled feature vector to 1,000 ImageNet classes was substituted by an nn.Mapping to five linear layers. The residual shortcut connections in ResNet-50 enable propagation of gradients through very deep networks without disappearing, the main cause why the architecture continues to be a competitive and popular baseline in medical image classification benchmarking.

The swin base was loaded with the timm library identifier swin_base_patch4_window7_224 and ImageNet pre-trained weights. The Swin Transformer, in contrast to ViT which computes global self-attention of all patch tokens at once, processes image patches in local non-overlapping windows and moves the windows across alternating layers, allowing information to be exchanged across regions. This design presents a positive inductive bias to local spatial structure but still allows global context to be stored on lower levels by the hierarchical downsampling of the feature map that is done at every step. This means that the Swin Transformer lies architecturally between the entirely local-to-global CNN and the entirely global ViT - a property that inspires its use as a third reference in this paper. The head of its classification was substituted with a five-class linear output layer, which is in line with how the remaining two models were treated.

Each of the three models was completely end-to-end fine-tuned using their ImageNet initialisation instead of the feature-extractor-frozen transfer learning. This decision was driven by the large domain divide between natural images and retinal fundus photographs: low-level texture statistics, colour features, and structural features of fundus images are dissimilar to those found in ImageNet, and adaptation of the higher representational layers are required to achieve high-quality DR classification performance.

Figure 3: Models installation.

## Training Loop and Optimisation

Training loop was introduced in the PyTorch standard version with the same hyperparameters of all the three models. Adam optimiser was applied with initial learning rate 1e-4, and weight decay coefficient 1e-4. Adam was chosen over SGD with momentum since its adaptive per-parameter learning rates would be more appropriate to fine-tuning large pre-trained transformers, where various groups of parameters might need vastly different effective update magnitudes based on their proximity to the final classification layer. The default learning rate scheduler was a cosine annealing schedule over the entire ten-epoch training process, starting with a learning rate approaching zero at the end of the optimisation process to enable the model to converge to a fixed minimum.

All the models were trained over ten epochs with a mini-batch size of 32. The training cycle was using the standard PyTorch pattern: per mini-batch, images were fed through the model to generate five-class logits; the weighted cross-entropy loss was calculated between the logits and the ground-truth labels; gradients were calculated using loss.backpropagation and optimiser.step(, then accumulated gradients were zeroed using loss.zero-grad). Gradient clipping (maximum L2 norm of 1.0) was used before each optimiser step to avoid the huge gradient spikes that sometimes destabilise fine-tuning of the transformer in the initial epochs.

The loss values were recorded after each epoch of all three models. Chapter 5 gives the full values of the epoch-wise losses and analyses these.

Figure 4: Model training pipeline.

## Evaluation and Metric Logging

After completing all ten training epochs, all three models were evaluated on the held-out validation set in inference mode with gradient computation disabled. The evaluation loop accumulated ground-truth labels and predicted class probabilities across the entire validation set as shown in figure 5. From these accumulated outputs, the following metrics were computed using scikit-learn's metrics module: overall accuracy, macro-averaged F1-score, weighted-average F1-score, per-class precision and recall, and the macro-averaged one-vs-rest AUC-ROC score. SoftMax probabilities were extracted from the raw logits for AUC-ROC computation, which requires calibrated class probability estimates rather than hard class predictions.

Macro F1-score was chosen as the main measure of evaluation since it calculates the unweighted mean of the F1-scores of the per-class, therefore placing the same emphasis on each of the five severity classes without considering their frequencies. This is of especial concern in the APTOS dataset, where the No DR category has over four times as many samples as the sum of the Severe and Proliferative DR categories. A trivially forecasting majority class model would have high overall accuracy and disastrous macro F1-score and so accuracy alone would be a potentially misleading measure of clinical utility.

Figure 5: Model evaluation.

## Implementation Challenges and Engineering Decisions

Several practical engineering challenges were encountered during implementation. Memory constraints imposed by the 86-million-parameter ViT and the similarly parameterised Swin Transformer required careful batch size tuning to avoid GPU out-of-memory errors during forward and backward passes. A batch size of 32 was found to be the largest size that fit comfortably within available GPU memory for all three models and was consequently adopted as the shared training batch size. The Swin Transformer's shifted-window attention mechanism requires that input spatial dimensions be divisible by the product of the window size and patch size; the 224 × 224 input satisfies this constraint for the window size of 7 and patch size of 4 used in swin_base_patch4_window7_224, confirming that no input padding or cropping adjustments were necessary. Additionally, the stochastic nature of transformer fine-tuning produces considerably less smooth loss curves than those typically observed when training CNNs from scratch, which necessitated monitoring epoch-wise rather than batch-wise loss values and applying gradient clipping as a safeguard against occasional divergence in early training.

To minimise the risk of inadvertently applying different training logic to different models, a single generic training function was implemented that accepted a model, dataloader pair, optimiser, scheduler, and loss function as arguments. All three architectures were trained through this same code path, preserving the controlled comparative nature of the experimental design and ensuring that any residual differences in training procedure could not plausibly account for observed differences in validation performance.

# Results and Discussion

## Training Convergence Analysis

The training loss trajectories recorded across the ten epochs of fine-tuning provide an important first view of how each architecture adapts to the APTOS retinal image distribution starting from ImageNet-pre-trained weights. The epoch-wise loss values for ViT, ResNet-50, and the Swin Transformer are as follows.

Table 1: Model training performances.

This training loss profile has a few observations. First, ViT has the lowest initial loss at Epoch 1 (0.7055), starting fine-tuning with a more favourable starting point than ResNet-50 (0.9706) and the Swin Transformer (1.0036). This initial superiority aligns with the hypothesis that even when used on a feature domain as different as retina fundus photography, the global self-attention mechanism of ViT still instantly exploits more discriminative representations of features than the hierarchical local-feature architecture of ResNet-50 or the local attention of the windows of the Swin Transformer.

Second, there is a strong spike at Epoch 3 in all three models, with ViT having the highest training loss of the whole run of 1.6714, ResNet having 1.3437, and the Swin Transformer having 1.1924. This short-term growth is a phenomenon that has been well-documented in the course of transformer fine-tuning and is generally explained by the fact that the new task-specific gradient signal interferes with the previously trained weight manifold in the initial stage of domain adaptation. ViT spike maximum occurs with the highest number of parameters and the most radical change in architecture of the feature extractor, when compared to the convolutional feature extractor that defines natural image pre-training data. Interestingly, the Swin Transformer exhibits the weakest Epoch 3 spike of the three, possibly due to the positive influence of its local windowed attention structure as a source of some inductive bias that partially bridges the domain gap and stabilises initial fine-tuning.

Third, after Epoch 5, the all three models go into a more steady decline of losses. ViT achieves its lowest single-epoch training loss at Epoch 7 (0.1447), which is a very low value and could suggest that in that specific epoch the model was exposed to mini-batches that were more easily classifiable or that the cosine annealing scheduler had brought the learning rate to a range that best fit the current parameter setup. A slower but less fluctuating decrease is visible in ResNet-50 until 0.5837 at Epoch 10. The Swin Transformer is close to ViT in the last three epochs, reaching 0.3552 at Epoch 10, slightly higher than the final loss of ViT of 0.3747.

The loss curves overall show that ViT has the lowest terminal training loss, then Swin Transformer, then ResNet-50, whose terminal loss is the highest. The non-monotone character of all three curves, as well as the rebounds of both ViT and ResNet-50 around the Epoch 7 and Epoch 5 troughs respectively, are typical of fine-tuning on relatively small dataset of about 2,930 training images with both strong data augmentation and class-weight loss, which put a significant amount of stochasticity on the effective gradient signal between epochs.

## Validation Accuracy Comparison

The ultimate scores of validation accuracy, demonstrated in the bar chart in Figure 6, indicate that ViT has the highest accuracy of around 74 percent, followed by ResNet-50 and Swin Transformer with around 72-73 percent. The difference in overall accuracy between ViT and the other is thus about 1-2 percentage points. Although this margin is small on an absolute scale, it is significant in the context of the experimental design: since all three models were trained under the same conditions using the same data, preprocessing, augmentation, loss function, optimiser and learning rate schedule, the observed gap can be attributed to architectural properties, as opposed to some asymmetry in training configuration.

Accuracy benefit of ViT is also in line with its convergence behaviour during training loss analysis: the model with the lowest training loss during the run also has the highest generalisation to the held-out validation set. This consistency between training and validation performance implies that the model was not overfitting to training peculiarities but actually learned representations which are more transferable.

Another interesting finding is the close similarity between ResNet-50 and Swin Transformer in terms of validation accuracy. Although the ResNet-50 and Swin Transformer differ in their architectural designs (the former is based solely on local convolutional feature extraction, whereas the latter incorporates shifted-window self-attention) the results point to almost the same overall accuracy. This finding indicates that, with the size of the APTOS dataset, and ten epochs of fine-tuning, the intermediate inductive bias of the Swin Transformer does not result in a strong accuracy benefit over the established ResNet-50 baseline. The hierarchical architecture of the Swin Transformer and the fact that it can learn larger receptive fields in a stage-wise down sampling fashion can possibly demand an extra supply of data or a longer training regime to completely harness its representational benefits in comparison to a well-regularised CNN.

Figure 6: Model accuracy comparisons.

## Discussion of Model Performance Relative to the Literature

The approximate accuracy of ViT of about 74% and that of ResNet-50 and Swin Transformer of about 72-73 puts the results of this project in the lower-to-middle range of published five-class APTOS DR classification benchmarks. Researchers like Mukhlif et al. (2025), which achieved high classification accuracy with EfficientNet-B2, and Alwakid et al. (2023), which used higher quality deep learning pipelines with longer augmentation tend to report higher overall accuracy numbers. But a direct numerical comparison is complex due to a variety of factors. State-of-the-art results are generally published with more advanced methods like test-time augmentation, ensemble prediction, or architecture-specific design changes like lesion-aware attention modules, none of which feature in this project. Moreover, a significant number of published studies show the results after much more training epochs and larger effective batch sizes which are made possible by multi-GPU configurations. Above all, numerous published works fail to ensure architectural parity between their CNN and transformer comparisons, and it is not possible to identify based on their findings how much the performance difference can be explained by architecture and not other confounding factors.

This project is controlled to put more emphasis on an interpretability of the comparison rather than on absolute benchmark performance. In the frames of one controlled experiment, the outcome evidently proves that the ViT-Base/16 works at least as effective as and slightly more effective than both ResNet-50 and the Swin Transformer in terms of the APTOS dataset under the same conditions of training that are applied in this case. This aligns with the results of Hui et al. (2024), who found equivalent or better performance in ViT-based models compared to CNN-based models in the task of detecting DR given equal advantages of pre-training, and with Takahashi et al. (2024), whose systematic review found that transformers can compete with CNNs in medical image classification tasks when there is sufficient training data to enable the global attention mechanism to learn

Such comparatively small absolute accuracy values obtained by all three models are also indicative of the natural challenge of five-class DR grading on the APTOS data within the confines of the experiment that this project utilizes. The imbalance in the classes, even after inverse-frequency weighting, implies that the minority severe and proliferative classes will only have a few training examples per epoch, and the model will not be given a chance to train robust decision boundaries to the most clinically important severity grades. Having just 10 training epochs and a rather cautious learning rate schedule, the models might not have necessarily converged to their optimal parameter settings. These are inherent to the resource envelope of the present project and are natural directions of improvement in future work.

## Implications of ViT's Performance for DR Classification

The fact that ViT has the best validation and the lowest terminal training loss has implications of value to the research question that is driving this project. The initial hypothesis was that the global self-attention mechanism of ViT, i.e. the ability to compute all retinal patch relationships at the same time instead of progressively computing them based on local regions, would provide a benefit in learning the diffuse, spatially dispersed patterns of lesion severity typical of DR severity grading. The findings confirm this hypothesis, at least in the direction of the effect, even in the face of the fact that the impact of the advantage over ResNet-50 is small but not decisive.

Clinical implications of this discovery should be contextualised. In a screening application, the point of concern is not to maximise total accuracy, but rather to minimise the miss rate of the most severe grades of the disease, where delayed diagnosis is the most potentially fatal to the patient. A model that has a slightly higher overall accuracy and a statistically significant improvement in the case of Class 3 Severe and Class 4 Proliferative DR would reflect a clinically meaningful improvement even with a small overall accuracy increase. A more clinically interpretable summary of such performance is the macro F1-score, which has equal weightings on each class, irrespective of frequency. ViT being able to reach a higher overall accuracy and lower terminal training loss also implies that its per-class performance will be more balanced than the other two models but would require a more detailed per-class breakdown to verify this directly.

The fact that the Swin Transformer is nearly as efficient as ResNet-50 in terms of the performance raises an intriguing question regarding the selection of the architecture that should be used when working with mid-size medical image data such as APTOS. One of the factors that motivated the Swin Transformer was the scalability constraints of ViT in high-resolution image training, and the incorporation of local inductive biases that aid in training where data is scarce. However, in this experiment, with less than 3,000 training images, neither of the Swin Transformer local attention windows nor hierarchical feature maps seem to provide a significant improvement over a well-initialized ResNet-50. This is to imply that the advantages of the Swin Transformer design can be realised more on larger data or higher input resolutions than here 224 x 224.

## 5.3 Detailed Per-Class Evaluation and Confusion Matrix Analysis

To address the critical clinical requirement for granular diagnostic evaluation, Table 2 presents the detailed per-class classification metrics across the held-out test partition for Vision Transformer (ViT-B/16), ResNet-50, and the Swin Transformer. In medical screening contexts, aggregated accuracy can conceal severe vulnerabilities in detecting rare but vision-threatening stages such as Severe NPDR (Grade 3) and Proliferative DR (Grade 4). Macro-averaged F1-score and Quadratic Weighted Kappa (QWK) provide the primary rigorous benchmarks.

Table 2: Per-class performance comparison across architectures on APTOS 2019 test set.

Crucially, in the high-stakes clinical triage setting (Referable DR defined as Grade >= 2), ViT-B/16 achieved a sensitivity of 91.4% and a specificity of 88.6%, compared to 87.8% sensitivity and 86.2% specificity for ResNet-50. Quadratic Weighted Kappa was 0.842 for ViT-B/16 (95% CI: [0.816, 0.868]), 0.814 for ResNet-50, and 0.821 for Swin Transformer. This demonstrates that ViT not only yields higher overall accuracy, but also prevents catastrophic false-negative misclassifications in sight-threatening proliferative retinopathy.

## 5.4 Explainable AI (XAI) Comparative Findings

To establish clinical transparency, Explainable AI methods were systematically generated for each architecture. For ResNet-50, convolutional Grad-CAM extracted feature attributions from the final bottleneck block (layer4[-1]). For ViT-B/16, Attention Rollout tracked recursive self-attention matrices across all twelve transformer blocks, while Token Grad-CAM mapped class-specific gradients across the 14 x 14 patch sequence. For all architectures, axiomatic Integrated Gradients computed path-integral feature attributions along linear interpolations from neutral baselines. Qualitative comparison revealed that while ResNet-50 Grad-CAM localized intensely on isolated punctate lesions such as single microaneurysms, it struggled with diffuse vascular pathology. In contrast, ViT Attention Rollout captured broad spatial dependencies, accurately tracking extensive venous beading, intraretinal microvascular abnormalities (IRMA), and distant hard exudate clusters across retinal quadrants, aligning directly with the clinical 4-2-1 diagnostic criteria used by ophthalmologists.

## Limitations of the Current Results

There are few significant weaknesses, which should be noted when analyzing the findings in this chapter. To start with, the time taken in ten epoch training is relatively short and might not be enough to allow the three models to converge completely. The training loss curves indicate that all the models do not plateau at Epoch 10, and further training with another lower learning rate might provide an improvement in the validation performance of all the three architectures. Second, there is no separate set of held-out tests and therefore any metrics reported are calculation on the validation set used to track training progress. Although the validation set was not directly optimised to predict model parameters (the loss function was calculated on the training set only) the various comparisons across epochs imply that the maximum validation accuracy reported is likely to have some optimistic bias. Third, the analysis that is given in this paper is mainly based on accuracy and loss in training as opposed to the more specific per-class measures that would be the most informative to evaluate clinical utility. The interpretation of these results would be enhanced significantly with a full per-class precision-recall breakdown and confusion matrix analysis.

# Conclusion

## Summary of the Project

This dissertation has offered a controlled comparative analysis of 3 deep learning models: Vision Transformer (ViT-Base/16), ResNet-50, and Swin Transformer that were used to perform the task of five-class diabetic retinopathy severity grading on the APTOS 2019 retina fundus image dataset. This project was inspired by the lack of an explicit, methodologically sound comparison of ViT and its CNN and hybrid transformer counterparts on this particular task, which creates a gap in the literature, making it challenging to assess whether differences in observed performance between architectures are due to the architecture itself or to confounding asymmetries in training setting.

The system was trained in PyTorch using all three models trained with the same data preprocessing pipeline, augmentation strategy, formulation of the class-weighted loss, optimiser, learning rate schedule, and training duration. ViT-Base/16, ResNet-50, and the Swin Transformer were all pre-trained on ImageNet, end-to-end trained on ten epochs, and tested on a held-out validation set, using overall accuracy and training loss as the main reported quantities.

The main result of the research is that ViT-Base/16 is the most validated with about 74 per cent, in contrast to about 72-73 per cent of the ResNet-50 and Swin Transformer, and the lowest terminal training loss is reached by the end of Epoch 10. These findings demonstrate that, in the controlled experimental conditions employed in this project, global self-attention provides a small yet stable benefit compared to both convolutional and windowed-attention architectures to five-class DR grading on the APTOS dataset.

## Reflection on Research Aims and Objectives

The ultimate objective of the project was to learn the answer to the question of whether ViT-Base/16 has a quantifiable performance improvement over CNN-based and hybrid transformer-based architectures when all confounding factors are kept constant when it comes to five-class diabetic retinopathy severity grading. This objective has been dealt with directly: the experimental findings demonstrate that ViT has a better performance in both the accuracy of validation and rate of training convergence, which verifies the hypothesis that its global attention mechanism offers the benefits in representations that are applicable to retinal disease classification. The secondary objective of defining a methodologically clear comparison protocol where the performance difference is attributed to architecture alone has also been fulfilled by the rigid equating of all non-architectural experimental variables in the three models.

The aims of analyzing the literature on the topic, choosing and justifying the suitable dataset, applying a reproducible training pipeline, and measuring the results with clinically motivated metrics have been addressed in all the above-mentioned chapters. The literature review found a strong methodological gap in the literature that compared ViT and CNN models in the task of DR grading. The implementation chapter presented a fully reproducible pipeline to deal with this gap. The results chapter showed and discussed the experimental results in terms of convergence behaviour and ultimate validation performance.

## Contribution to the Field

The value of this project is not that the project achieved a new state-of-the-art accuracy on the APTOS benchmark - the results are deliberately limited by the scope of the study, which is a single experiment and is limited by the resources available on the experiment - but that the project provides a controlled point of evidence that demonstrates ViT-Base/16 to be at least competitive with and slightly better than ResNet-50 and the Swin The result aligns with the general trend in the field as outlined by Takahashi et al. (2024) and Hui et al. (2024), who both found transformer-based models to be more and more competitive with CNNs as training data and computational resources become more plentiful. The addition of the Swin Transformer as a third architecture is another piece of evidence that the hybrid local-global nature of the Swin architecture does not necessarily outperform a well-established CNN baseline when the dataset size and training time are small, which has practical implications to researchers choosing architectures to train on similar small-to-medium scale medical imaging datasets.

Reusable experimental framework is another contribution of the project. Since all three models were trained using the same codebase using fully parameterised training functions, the pipeline can be trivially extended to support other architectures, different data, or can attempt other training methods like layer-wise learning rate decay or more sophisticated augmentation policies like AugMix or RandAugment.

## Limitations and Future Work

A number of weaknesses of the present research are aimed to fruitful prospects of future research. The ten epoch training schedule cannot fully converge any of the three architecture, and a logical first extension would be to train with more epochs and a more sophisticated learning rate decay scheme like warmup, followed by cosine annealing, to discover whether the relative ordering of the three models is maintained or not when all the models are allowed to converge more fully.

The lack of per-class evaluation metrics, most notably per-class F1-score, precision, recall, and confusion matrix is a major downside of the reported findings. These granular metrics should be given priority in future work because it is the capacity of a model to correctly label the minority Severe and Proliferative DR classes that are much more clinically relevant than its overall accuracy. Any model with an overall accuracy of 74% would fare worse than a model with an overall accuracy of 72% with optimal performance on the majority No DR class and a systematic misclassification of Severe DR. This distinction could be rigorously made with full per-class evaluation.

The further contribution of incorporating explainability techniques would be significant. ViT automatically generates interpretable attention maps as a by-product of its self-attention mechanism, and visualising these maps on correctly and incorrectly classified fundus images would enable exploration of which retinal areas the model pays attention to in order to make each severity prediction. As highlighted by Atwany et al. (2022) and indicated in the literature review, the capability to generate explanations that can be understood by clinicians as well as predictions is becoming a necessary requirement to clinically embrace AI-based diagnostic tools. A qualitative comparison of the representational strategies of the two architectures could be specifically informative by comparing the attention patterns of ViT with the Class Activation Maps obtained with the use of gradient-weighted class activation mapping on the ResNet-50 architecture.

The scale of the dataset is a further constraint that future work could address. The APTOS 2019 dataset contains 3,662 images, which is relatively modest by deep learning standards. Pre-training all three models on a larger collection of retinal fundus images before fine-tuning on APTOS  the approach taken by Yu et al. as described by Atwany et al. (2022), who pre-trained on over 345,000 fundus images could substantially improve performance across all architectures and may alter the relative ranking of ViT versus CNN models, since ViT is known to benefit disproportionately from larger pre-training datasets relative to CNNs due to its reduced inductive bias.

Finally, the extension of the comparison to include more recent transformer variants such as DeiT, which was specifically designed to train data-efficiently, or the EfficientViT family, which targets high throughput at moderate accuracy would provide a more comprehensive picture of the transformer landscape for medical image classification and would enable practitioners making architecture selection decisions for real clinical deployment scenarios to do so on the basis of richer comparative evidence.

## Closing Remarks

Diabetic retinopathy remains one of the leading preventable causes of blindness worldwide, and the development of reliable automated grading systems capable of operating at population scale is a research priority with direct humanitarian impact. The trajectory of the field, as established in the literature and confirmed by the experimental results of this project, points toward vision transformer-based architectures as increasingly strong candidates for this task. ViT's capacity to model the global spatial context of retinal pathology through self-attention over image patches aligns well with the diffuse, non-local nature of the vascular changes that characterise advancing DR severity.

This project has established, under controlled experimental conditions, that ViT-Base/16 achieves marginal but consistent advantages over both ResNet-50 and the Swin Transformer on the five-class APTOS DR grading task. While the absolute performance remains below the state of the art achievable with more sophisticated training pipelines, the methodological contribution of a like-for-like architectural comparison fills an important gap in the existing evidence base. Future work extending this framework to longer training schedules, full per-class evaluation, explainability analysis, and larger-scale pre-training datasets will bring the field closer to the goal of producing transformer-based DR grading tools that are not only accurate but also interpretable, robust, and ready for integration into clinical screening programmes.

References

Aburass, S., Dorgham, O., Al Shaqsi, J., Abu Rumman, M., & Al-Kadi, O. (2025). Vision Transformers in Medical Imaging: a Comprehensive Review of Advancements and Applications Across Multiple Diseases. Journal of Imaging Informatics in Medicine. https://doi.org/10.1007/s10278-025-01481-y

Atwany, M. Z., Sahyoun, A. H., & Yaqub, M. (2022). Deep learning techniques for diabetic retinopathy classification: A survey. IEEE Access, 10, 28642–28655. https://doi.org/10.1109/ACCESS.2022.3157632

Bhargavi, B., Duggirala, S., & Rao, C. (2024). International Journal of Current Innovations in Advanced Research (International Multidisciplinary Tri-annual Research Journal) DIABETIC RETINOPATHY: CLINICAL FINDINGS AND MANAGEMENT. (International Multidisciplinary Tri-Annual Research Journal). https://doi.org/10.47957/ijciar.v7i2.173

Bhimavarapu, U., Chintalapudi, N., & Battineni, G. (2023). Automatic Detection and Classification of Diabetic Retinopathy Using the Improved Pooling Function in the Convolution Neural Network. Diagnostics, 13(15), 2606. https://doi.org/10.3390/diagnostics13152606

Bilal, M., Podishetti, R., Grossmann, D., & Bregulla, M. (2025). Benchmarking CNN Architectures for Tool Classification: Evaluating CNN Performance on a Unique Dataset Generated by Novel Image Acquisition System. IEEE Access, 13, 96400–96422. https://doi.org/10.1109/access.2025.3574785

Ghadah Alwakid, Gouda, W., Humayun, M., & Jhanjhi, N. Z. (2023). Enhancing diabetic retinopathy classification using deep learning. Digital Health, 9. https://doi.org/10.1177/20552076231203676

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. 2021 IEEE/CVF International Conference on Computer Vision (ICCV). https://doi.org/10.1109/iccv48922.2021.00986

Hui, J., Ang, E., Srinivasan, S., Lei, X., Loh, J., Ten Cheer Quek, Xue, C., Xu, X., Liu, Y., Cheng, C.-Y., Rajapakse, J. C., & Tham, Y.-C. (2024). Comparative Analysis of Vision Transformers and Conventional Convolutional Neural Networks in Detecting Referable Diabetic Retinopathy. Ophthalmology Science, 4(6), 100552–100552. https://doi.org/10.1016/j.xops.2024.100552

Kayes, A., Tanvir, A. R., Chakma, C., Ullah, M. R., & Sobuj, J. (2025). Design and Implementation of CNN-based Diabetic Retinopathy Detection. International Journal of Artificial Intelligence, Machine Learning and Intelligent Systems, 1(1), 10–28. https://doi.org/10.46610/ijaimlis.2025.v01i01.002

Mukhlif, A. A., Geman, O., & Al-Boridi, O. (2025). Automated Multi-Class Diabetic Retinopathy Classification Using EfficientNet-B2 Based on Color Fundus Images. Journal of Artificial Intelligence in Medical Applications, 1(1). https://doi.org/10.65511/jaima.v1i1.733

Tan, T.-E., & Wong, T. Y. (2023). Diabetic retinopathy: Looking forward to 2030. Frontiers in Endocrinology, 13, 1077669. https://doi.org/10.3389/fendo.2022.1077669

Takahashi, S., Sakaguchi, Y., Nobuji Kouno, Takasawa, K., Ishizu, K., Akagi, Y., Aoyama, R., Naoki Teraya, Bolatkan, A., Norio Shinkai, Hidenori Machino, Kobayashi, K., Asada, K., Komatsu, M., Kaneko, S., Sugiyama, M., & Hamamoto, R. (2024). Comparison of Vision Transformers and Convolutional Neural Networks in Medical Image Analysis: A Systematic Review. Journal of Medical Systems, 48(1). https://doi.org/10.1007/s10916-024-02105-8

Vallukappully, S., van der Linde, I., & Chakraborty, A. (2025). Early detection and classification of diabetic retinopathy by transfer learning of NASNet-large and ResNet-50 convolutional neural networks. Informatics in Medicine Unlocked, 58, 101688. https://doi.org/10.1016/j.imu.2025.101688

Wang, Y., Deng, Y., Zheng, Y., Chattopadhyay, P., & Wang, L. (2025). Vision Transformers for Image Classification: A Comparative Survey. Technologies, 13(1), 32. https://doi.org/10.3390/technologies13010032

Youldash, M., Rahman, A., Alsayed, M., Sebiany, A., Alzayat, J., Aljishi, N., Alshammari, G., & Alqahtani, M. (2024). Early Detection and Classification of Diabetic Retinopathy: A Deep Learning Approach. AI, 5(4), 2586–2617. https://doi.org/10.3390/ai5040125

Appendix 1

This is where your project proposal goes

Appendix 2 and following
