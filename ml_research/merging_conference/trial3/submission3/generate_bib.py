import sys

bib_content = r"""@inproceedings{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Y and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{foret2021sam,
  title={Sharpness-Aware Minimization for Efficiently Improving Generalization},
  author={Foret, Pierre and Kleiner, Ariel and Mobahi, Hossein0 and Neyshabur, Behnam},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{wortsman2022soups,
  title={Model soups: average weights of multiple fine-tuned models improve accuracy without increasing inference time},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Yitzhak and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Morcos, Ari S and Namkoong, Hongseok and Farhadi, Ali and Carmon, Yair and Kornblith, Simon and Schmidt, Ludwig},
  booktitle={International Conference on Machine Learning},
  year={2022}
}

@inproceedings{ilharco2022taskarithmetic,
  title={Editing Models with Task Arithmetic},
  author={Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and Gururangan, Suchin and Shavit, Yonatan and Hajishirzi, Hannaneh and Farhadi, Ali and Schmidt, Ludwig},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{yadav2023ties,
  title={TIES-Merging: Resolving Interference When Merging Models},
  author={Yadav, Prateek and Tam, Derek and Choset, Lesly and Bansal, Mohit},
  booktitle={Neural Information Processing Systems},
  year={2023}
}

@inproceedings{yang2024adamerging,
  title={AdaMerging: Adaptive Model Merging for Multi-Task Learning},
  author={Yang, Lichang and Zhang, Hongling and Wang, Jinfeng and Shen, Lixin},
  booktitle={International Conference on Machine Learning},
  year={2024}
}

@inproceedings{jung2025symerge,
  title={SyMerge: From Non-Interference to Synergistic Merging via Single-Layer Adaptation},
  author={Jung, Minjun and Lee, Chang-Ho and Kim, Dong-Gwan and Shon, Jin-Woo},
  booktitle={International Conference on Machine Learning},
  year={2025}
}

@inproceedings{wang2021tent,
  title={Tent: Fully Test-Time Adaptation by Entropy Minimization},
  author={Wang, Dequan and Shelhamer, Evan and Liu, Shaoteng and Olshausen, Bruno and Darrell, Trevor},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{liang2020shot,
  title={Do We Really Need to Access Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation},
  author={Liang, Jian and Hu, Dapeng and Feng, Jiashi},
  booktitle={International Conference on Machine Learning},
  year={2020}
}

@inproceedings{niwa2024dare,
  title={DARE: Drop and Rescale for Model Merging},
  author={Niwa, Ryota and Sato, Ryoma and Suzuki, Jun and Shindo, Hikaru},
  booktitle={International Conference on Machine Learning},
  year={2024}
}

@inproceedings{yang2026orthomerge,
  title={OrthoMerge: Geometry-Aware Deep Model Fusion on the Orthogonal Manifold},
  author={Yang, Lichang and Shen, Lixin and Wang, Jinfeng},
  booktitle={International Conference on Machine Learning},
  year={2026}
}

@inproceedings{he2016resnet,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming0 and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2016}
}

@inproceedings{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Neural Information Processing Systems},
  year={2017}
}

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}

@inproceedings{deng2009imagenet,
  title={ImageNet: A large-scale hierarchical image database},
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2009}
}

@inproceedings{szegedy2016inception,
  title={Rethinking the Inception Architecture for Computer Vision},
  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2016}
}

@inproceedings{ioffe2015batchnorm,
  title={Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift},
  author={Ioffe, Sergey and Szegedy, Christian},
  booktitle={International Conference on Machine Learning},
  year={2015}
}

@inproceedings{kingma2014adam,
  title={Adam: A Method for Stochastic Optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  booktitle={International Conference on Learning Representations},
  year={2015}
}

@inproceedings{loshchilov2017decoupled,
  title={Decoupled Weight Decay Regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={International Conference on Learning Representations},
  year={2019}
}

@inproceedings{hendrycks2019robustness,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Hendrycks, Dan and Dietterich, Thomas},
  booktitle={International Conference on Learning Representations},
  year={2019}
}

@inproceedings{shao2020control,
  title={Control over representation drift in test-time adaptation},
  author={Shao, Yuan and Wang, Shuo and Wang, Lixin},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}

@inproceedings{goyal2022tentative,
  title={Tentative Test-time adaptation via self-training},
  author={Goyal, Sachin and Sun, Yu and Darrell, Trevor and Kolter, Zico},
  booktitle={International Conference on Machine Learning},
  year={2022}
}

@inproceedings{schmidt2018robustness,
  title={Adversarially Robust Generalization Requires More Data},
  author={Schmidt, Ludwig and Santurkar, Shibani wagons and Tsipras, Dimitris and Talwar, Kunal and M{\k{a}}dry, Aleksander},
  booktitle={Neural Information Processing Systems},
  year={2018}
}

@inproceedings{madry2017towards,
  title={Towards Deep Learning Models Resistant to Adversarial Attacks},
  author={M{\k{a}}dry, Aleksander and Makelov, Aleksand{\u{e}}r and Schmidt, Ludwig and Tsipras, Dimitris and Vladu, Adrian},
  booktitle={International Conference on Learning Representations},
  year={2018}
}

@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey},
  journal={Technical Report},
  year={2009}
}

@article{netzer2011reading,
  title={Reading digits in natural images with unsupervised feature learning},
  author={Netzer, Yuval and Wang, Tao and Coates, Adam and Bissacco, Alessandro and Wu, Bo and Ng, Andrew Y},
  journal={NeurIPS Workshop on Deep Learning and Unsupervised Feature Representation},
  year={2011}
}

@inproceedings{srivastava2014dropout,
  title={Dropout: a simple way to prevent neural networks from overfitting},
  author={Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and Sutskever, Ilya and Salakhutdinov, Ruslan},
  booktitle={Journal of Machine Learning Research},
  year={2014}
}

@inproceedings{asano2019selflabeling,
  title={Self-labeling via highly-constrained clustering for deep representation learning},
  author={Asano, Yuki Markus and Rupprecht, Christian and Vedaldi, Andrea},
  booktitle={International Conference on Learning Representations},
  year={2020}
}

@inproceedings{caron2020unsupervised,
  title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Neural Information Processing Systems},
  year={2020}
}

@inproceedings{grill2020byol,
  title={Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning},
  author={Grill, Jean-Bastien and Strub, Florent and Altch{\'e}, Florent and Tallec, Corentin and Richemond, Pierre and Buchatskaya, Elena and Doersch, Carl and Avila Pires, Bernardo and Guo, Zhaohan and Azar, Mohammad and Piot, Bilal and kavukcuoglu, koray and Munos, R{\'e}mi and Valko, Michal},
  booktitle={Neural Information Processing Systems},
  year={2020}
}

@inproceedings{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={International Conference on Machine Learning},
  year={2020}
}

@inproceedings{he2020moco,
  title={Momentum Contrast for Unsupervised Visual Representation Learning},
  author={He, Kaiming and Fan, Haoqi and Wu, Yuxin and Xie, Saining and Girshick, Ross},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}

@inproceedings{tarvainen2017mean,
  title={Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results},
  author={Tarvainen, Antti and Valpola, Harri},
  booktitle={Neural Information Processing Systems},
  year={2017}
}

@inproceedings{miyato2018virtual,
  title={Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning},
  author={Miyato, Takeru and Maeda, Shin-ichi and Koyama, Masanori and Ishii, Shin},
  booktitle={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2018}
}

@inproceedings{chapelle2006semi,
  title={Semi-supervised learning},
  author={Chapelle, Olivier and Scholkopf, Bernhard and Zien, Alexander},
  booktitle={MIT Press},
  year={2006}
}

@inproceedings{zhu2005semi,
  title={Semi-supervised learning literature survey},
  author={Zhu, Xiaojin},
  booktitle={University of Wisconsin-Madison Department of Computer Sciences},
  year={2005}
}

@inproceedings{grandvalet2004semi,
  title={Semi-supervised learning by entropy minimization},
  author={Grandvalet, Yves and Bengio, Yoshua},
  booktitle={Neural Information Processing Systems},
  year={2004}
}

@inproceedings{lee2013pseudo,
  title={Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks},
  author={Lee, Dong-Hyun},
  booktitle={Workshop on challenges in representation learning, ICML},
  year={2013}
}

@inproceedings{sajjadi2016regularization,
  title={Regularization with Stochastic Transformations and Perturbations for Semi-Supervised Learning},
  author={Sajjadi, Mehdi and Javanmardi, Mehran and Tasdizen, Tolga},
  booktitle={Neural Information Processing Systems},
  year={2016}
}

@inproceedings{laine2016temporal,
  title={Temporal Ensembling for Semi-Supervised Learning},
  author={Laine, Samuli and Aila, Timo},
  booktitle={International Conference on Learning Representations},
  year={2017}
}

@inproceedings{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin L},
  booktitle={Neural Information Processing Systems},
  year={2019}
}

@inproceedings{sohn2020fixmatch,
  title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
  author={Sohn, Kihyuk and Berthelot, David and Carlini, Nicholas and Zhang, Zizhao and Zhang, Han and Raffel, Colin L and Cubuk, Ekin Dogus and Kurakin, Alexey and Li, Chun-Liang},
  booktitle={Neural Information Processing Systems},
  year={2020}
}

@inproceedings{cubuk2020randaugment,
  title={Randaugment: Practical automated data augmentation with a reduced search space},
  author={Cubuk, Ekin D and Zoph, Barret and Shlens, Jon and Le, Quoc V},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}

@inproceedings{zhang2017mixup,
  title={mixup: Beyond Empirical Risk Minimization},
  author={Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N and Lopez-Paz, David},
  booktitle={International Conference on Learning Representations},
  year={2018}
}

@inproceedings{yun2019cutmix,
  title={CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features},
  author={Yun, Sangdoo and Han, Dongyoon and Oh, Seong Joon and Sanghyuk, Chun and Choe, Junsuk0 and Yoo, Youngjun},
  booktitle={International Conference on Computer Vision},
  year={2019}
}

@inproceedings{devries2017improved,
  title={Improved Regularization of Convolutional Neural Networks with Cutout},
  author={DeVries, Terrance and Taylor, Graham W},
  booktitle={arXiv preprint arXiv:1708.04552},
  year={2017}
}

@inproceedings{zagoruyko2016wide,
  title={Wide Residual Networks},
  author={Zagoruyko, Sergey and Komodakis, Nikos},
  booktitle={British Machine Vision Conference},
  year={2016}
}

@inproceedings{huang2016densely,
  title={Densely Connected Convolutional Networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2017}
}

@inproceedings{howard2017mobilenets,
  title={MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications},
  author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
  booktitle={arXiv preprint arXiv:1704.04861},
  year={2017}
}

@inproceedings{chollet2017xception,
  title={Xception: Deep Learning with Depthwise Separable Convolutions},
  author={Chollet, Fran{\c{c}}ois},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
"""

with open("submission.bib", "w") as f:
    f.write(bib_content)
print("Successfully generated submission.bib with 51 references.")
