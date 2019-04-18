# CNN Visualiztion

## 1 CNN原理和性质（Understanding CNN）

论文：通过测量同变性和等价性来理解图像表示(Understanding image representations by measuring their equivariance and equivalence)

作者：Karel Lenc, Andrea Vedaldi, CVPR, 2015.

链接：http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Lenc_Understanding_Image_Representations_2015_CVPR_paper.pdf


论文：深度神经网络容易被愚弄：无法识别的图像的高置信度预测(Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images)

作者：Anh Nguyen, Jason Yosinski, Jeff Clune, CVPR, 2015.

链接：http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Nguyen_Deep_Neural_Networks_2015_CVPR_paper.pdf


论文：**通过反演理解深度图像表示(Understanding Deep Image Representations by Inverting Them)**

作者：Aravindh Mahendran, Andrea Vedaldi, CVPR, 2015.

链接：http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mahendran_Understanding_Deep_Image_2015_CVPR_paper.pdf


论文：深度场景CNN中的对象检测器(Object Detectors Emerge in Deep Scene CNNs)

作者：Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba, ICLR, 2015.

链接：http://arxiv.org/abs/1412.6856


论文：**用卷积网络反演视觉表示(Inverting Visual Representations with Convolutional Networks)**

作者：Alexey Dosovitskiy, Thomas Brox, arXiv, 2015.

链接：http://arxiv.org/abs/1506.02753


论文：**可视化和理解卷积网络(Visualizing and Understanding Convolutional Networks)(反卷积网络DeconvNet: Deconvolutional Networks)**

作者：Matthrew Zeiler, Rob Fergus, ECCV, 2014.

链接：http://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf

## 2

[pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations#smooth-grad)

* Gradient visualization with vanilla backpropagation
* Gradient visualization with guided backpropagation
* Gradient visualization with saliency maps
* Gradient-weighted class activation mapping
* Guided, gradient-weighted class activation mapping
* Smooth grad
* CNN filter visualization
* Inverted image representations
* Deep dream
* Class specific image generation

## 3

[Some CNN visualization tools and techniques](http://www.erogol.com/cnn-visualization-tools-techniques/)

## 4

[瞎谈CNN：通过优化求解输入图像](https://zhuanlan.zhihu.com/p/25559267)

* Visualizing Higher-Layer Features of a Deep Network
* 对抗样本（Adversarial Examples）
* 语义信息和高层神经元
* Deep Dream
* Neural Art/Style

## 5

[CNN可视化研究综述（一）](https://mp.weixin.qq.com/s/cXwjOssIi8_7GTmqaGIgAw)

### 通过Activation Maximization进行可视化。

**Visualizing Higher-Layer Features of a Deep Network**: 可视化DNN学习到的高层特征。激活最大化、采样、线性组合法。https://blog.csdn.net/zouxy09/article/details/10012747 https://www.jianshu.com/p/598998bf25e3 https://blog.csdn.net/sheng_ai/article/details/40628757

## 6 cs231n-卷积网络可视化

https://www.cnblogs.com/coldyan/p/8403506.html

## 7 t-SNE visualization of CNN codes

https://cs.stanford.edu/people/karpathy/cnnembed/

## 历程

可视化CNN结构：

参考自[杂谈CNN：如何通过优化求解输入图像](https://www.leiphone.com/news/201706/HZlXRPP2Txd79xmd.html)
[Feature Visualization: How neural networks build up their understanding of images（好文）](https://distill.pub/2017/feature-visualization/)
[Deep Visualization:可视化并理解CNN](https://blog.csdn.net/yj3254/article/details/79167338)
[Deep Visualization:可视化并理解CNN(转)](https://www.cnblogs.com/byteHuang/p/6932772.html)

* 最初的可视化工作见于AlexNet论文中。在这篇开创Deep Learning新纪元的论文中，Krizhevshy直接可视化了第一个卷积层的卷积核。
* 最开始使用图片块来可视化卷积核是在RCNN论文中，Girshick的工作显示了数据库中对AlexNet模型较高层(pool5)某个channel具有较强响应的图片块。
* ZFNet论文中，系统化地对AlexNet进行了可视化，并根据可视化结果改进了AlexNet得到了ZFNet,拿到了ILSVRC2014的冠军。即**Visualizing and Understanding Convolutional Networks**
* **Visualizing Higher-Layer Features of a Deep Network是最早的方法**
* Cornell的Jason Yosinski把公式改了改，提出Understanding Neural Networks Through Deep Visualizatio，官方网站地址：http://yosinski.com/deepvis#toolbox。**开发了可视化CNN的工具**，toolbox:yosinski/deep-visualization-toolbox。借助这种可视化，我们能够分析出网络是不是真的学习到了我们希望其所学的特征。
* Intriguing properties of neural networks的发现是CNN中表示高层学习到的语义信息的，并不是某一个神经元，而是高层神经元构成的空间。
* Deep Dream **understand a layer as a whole**
* **Understanding Deep Image Representations by Inverting Them**
* A Neural Algorithm of Artistic Style

恢复输入:

* **Visualizing and Understanding Convolutional Networks** [论文学习7“Visualizing and Understanding Convolutional Networks”文章学习](https://www.jianshu.com/p/d4012045cf43)
* **Inverting Visual Representations with Convolutional Networks** [Inverting Visual Representations with Convolutional Networks论文理解](https://blog.csdn.net/wyl1987527/article/details/73350903) [论文笔记 Inverting Visual Representations with Convolutional Networks](https://www.cnblogs.com/everyday-haoguo/p/Note-IVR.html) [Inverting Visual Representations with Convolutional Networks](http://www.echojb.com/image/2017/06/28/446633.html)

[深度学习中的可解释性](https://zhuanlan.zhihu.com/p/48279309)

其中的分类很有用

* 隐层分析法
* 模拟模型方法: 例如Interpreting Blackbox Models via Model Extraction
* 注意力机制
* 分段线性函数下的神经网络

