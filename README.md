# VOC Classification Pipeline

## 1. Introduction
This project focuses on the classification of images using the PASCAL VOC dataset. The classification pipeline handles imbalanced data, incorporates custom architectures, and evaluates performance on various metrics. This document details the methods employed and the challenges encountered, such as class imbalance and adversarial attacks.

## 2. Data Augmentation

### 2.1 Upsampling Approach - Drawbacks
In the context of an imbalanced dataset, various strategies can address the issue:
- **Upsampling** generates synthetic data to increase the representation of minority classes. However, it risks overfitting by duplicating the same instances.
- **Undersampling** removes a portion of the majority class, but it risks losing valuable data from minority classes.

To address this, we augmented the dataset by increasing the number of instances for every class, except the "person" class. Despite these efforts, performance did not significantly improve, so we continued with the strategy of assigning weights to each class.

## 3. Model Architectures

### 3.1 ComplexNetwork Architecture
The best-performing custom architecture, **ComplexNetwork**, includes:
- Four sets of convolutional layers with batch normalization, ReLU activation, and max pooling.
- Convolutional layers with 3x3 kernels, stride 1, and padding 1.
- Max pooling with a 2x2 kernel and stride 2.
- Two fully connected layers (512 and 20 units).
- Final output from the second fully connected layer.

### 3.2 ResNet-34 vs ComplexNetwork
While **ResNet-34** is known for its deep architecture and skip connections, **ComplexNetwork** offers a computationally lighter alternative, which performed comparably well given limited resources.

## 4. Adversarial Attack

### 4.1 White-Box Attack
We implemented a white-box adversarial attack targeting a pretrained **ResNet34** model. The objective was to generate perturbations that confuse the model:
- Perturbations were regulated by applying a sigmoid or hyperbolic tangent, scaled by a constant, or through l2 regularization.
- The attack focused on manipulating fake labels (e.g., assigning 0, 0.5, or 1 to every class).

While the autoencoder architectures didn't fully fool the ResNet34 due to its robustness, the attack did succeed in reducing the model’s confidence in predictions. Future work could focus on stronger perturbations or targeting specific classes.

## 5. Discussion

### 5.1 Classification
The pretrained **ResNet50** significantly outperformed custom networks like **SimpleNetwork**, **ComplexNetwork**, and **TinyVGG**. The pretrained model's success is attributed to the availability of large-scale pretraining on ImageNet, whereas our models were constrained by limited data and resources.

### 5.2 Segmentation
For the segmentation task, we implemented a **U-Net** architecture from scratch and compared it with a pretrained ResNet-based model. The pretrained model again outperformed the U-Net due to the challenges of a small and imbalanced dataset.

### 5.3 Adversarial Attack
The adversarial attack, though limited in success, showed the potential for reducing the confidence of robust models like ResNet34. With improvements in the model architecture and training data, future attacks could achieve greater effectiveness.

### 5.4 Theoretical vs. Real-World Scenarios
The main takeaway from this project is the importance of sufficient data in deep learning models. While deep learning solutions are highly effective with enough data, limited datasets may lead to suboptimal performance. In such cases, more classic machine learning methods (e.g., feature extraction and SVMs) could offer alternative solutions.

## 6. References
1. https://doi.org/10.1016/j.asoc.2018.05.018  
2. https://doi.org/10.1109/TPAMI.2021.3059968  
3. https://arxiv.org/abs/1512.03385  
4. https://arxiv.org/abs/1505.04597  
5. https://doi.org/10.48550/arXiv.1409.1556

  
## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/lienertdemaeyer/Pascal-VOC-Classification-segmentation.git
cd Pascal-VOC-Classification-segmentation
pip install -r requirements.txt


