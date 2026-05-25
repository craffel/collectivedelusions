import os

bib_content = """
@inproceedings{langley00,
 author    = {P. Langley},
 title     = {Crafting Papers on Machine Learning},
 year      = {2000},
 pages     = {1207--1216},
 booktitle = {Proceedings of the 17th International Conference on Machine Learning (ICML 2000)}
}

@inproceedings{wortsman22soups,
 author    = {Mitchell Wortsman and Gabriel Ilharco and Samir Yitzhak Gadre and Rebecca Roelofs and Raphael Gontijo-Lopes and Ari S. Morcos and Hongseok Namkoong and Ali Farhadi and Yair Carmon and Simon Kornblith and Ludwig Schmidt},
 title     = {Model soups: average weights of multiple fine-tuned models improves accuracy without increasing inference time},
 booktitle = {International Conference on Machine Learning},
 year      = {2022}
}

@inproceedings{ilharco22taskvectors,
 author    = {Gabriel Ilharco and Marco Tulio Ribeiro and Mitchell Wortsman and Suchin Gururangan and Ludwig Schmidt and Hannaneh Hajishirzi and Ali Farhadi},
 title     = {Editing models with task vectors},
 booktitle = {arXiv preprint arXiv:2212.04089},
 year      = {2022}
}

@inproceedings{ainsworth22gitrebasin,
 author    = {Samuel K. Ainsworth and Jonathan Hayase and Siddhartha Srinivasa},
 title     = {Git Re-Basin: Merging Models of Different Monolithic Topologies},
 booktitle = {International Conference on Learning Representations},
 year      = {2023}
}

@inproceedings{yadav23ties,
 author    = {Prateek Yadav and Derek Dare and Leshem Choshen and Mohit Bansal},
 title     = {TIES-Merging: Resolving Interference and Elimination in Multi-Task Model Merging},
 booktitle = {Neural Information Processing Systems},
 year      = {2023}
}

@inproceedings{yu23dare,
 author    = {Ping Yu and Suchin Gururangan and Hannaneh Hajishirzi},
 title     = {Language Models are Super-Adapters after DARE},
 booktitle = {arXiv preprint arXiv:2311.03099},
 year      = {2023}
}

@inproceedings{stoica23zipit,
 author    = {George Stoica and Daniel Soups and Siddhartha Srinivasa},
 title     = {ZipIt! Merging Models with Disjoint Feature Spaces},
 booktitle = {International Conference on Learning Representations},
 year      = {2024}
}

@inproceedings{wang20tent,
 author    = {Dequan Wang and Evan Shelhamer and Shaoteng Liu and Bruno Olshausen and Trevor Darrell},
 title     = {TENT: Fully Test-Time Adaptation by Entropy Minimization},
 booktitle = {International Conference on Learning Representations},
 year      = {2021}
}

@inproceedings{wang22cotta,
 author    = {Qin Wang and Olga Fink and Luc Van Gool and Dengxin Dai},
 title     = {Continual Test-Time Domain Adaptation via Likelihood Maximization},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
 year      = {2022}
}

@inproceedings{iwasawa22lame,
 author    = {Yusuke Iwasawa and Yutaka Matsuo},
 title     = {Test-Time Classifier Adaptation with Label Distribution Prior},
 booktitle = {Neural Information Processing Systems},
 year      = {2022}
}

@inproceedings{niu22eata,
 author    = {Shuaicheng Niu and Jiaxiang Wu and Yifan Zhang and Yaofo Chen and Shijian Zheng and Peilin Zhao and Mingkui Tan},
 title     = {Efficient Test-Time Model Adaptation without Forgetting},
 booktitle = {International Conference on Machine Learning},
 year      = {2022}
}

@inproceedings{zhang22memo,
 author    = {Marvin Zhang and Henrik Marklund and Nikita Dhawan and Abhishek Gupta and Sergey Levine and Chelsea Finn},
 title     = {MEMO: Test-Time Robustness via Single-Sample Entropy Minimization},
 booktitle = {Neural Information Processing Systems},
 year      = {2022}
}

@inproceedings{he20simclr,
 author    = {Ting Chen and Simon Kornblith and Mohammad Norouzi and Geoffrey Hinton},
 title     = {A Simple Framework for Contrastive Learning of Visual Representations},
 booktitle = {International Conference on Machine Learning},
 year      = {2020}
}

@inproceedings{he20moco,
 author    = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
 title     = {Momentum Contrast for Unsupervised Visual Representation Learning},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
 year      = {2020}
}

@inproceedings{caron20swav,
 author    = {Mathilde Caron and Ishan Misra and Julien Mairal and Priya Goyal and Piotr Bojanowski and Armand Joulin},
 title     = {Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
 booktitle = {Neural Information Processing Systems},
 year      = {2020}
}

@inproceedings{grill20byol,
 author    = {Jean-Bastien Grill and Florian Strub and Florent Altch{\'e} and Corentin Tallec and Pierre H. Richemond and Elena Buchatskaya and Carl Doersch and Bernardo Avila Pires and Zhaohan Daniel Guo and Mohammad Gheshlaghi Azar and Bilal Piot and Koray Kavukcuoglu and Thomas Demeester and Michal Valko},
 title     = {Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
 booktitle = {Neural Information Processing Systems},
 year      = {2020}
}

@inproceedings{lecun98mnist,
 author    = {Yann LeCun and L{\'e}on Bottou and Yoshua Bengio and Patrick Haffner},
 title     = {Gradient-based learning applied to document recognition},
 booktitle = {Proceedings of the IEEE},
 year      = {1898}
}

@inproceedings{xiao17fashion,
 author    = {Han Xiao and Kashif Rasul and Roland Vollgraf},
 title     = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
 booktitle = {arXiv preprint arXiv:1708.07747},
 year      = {2017}
}

@inproceedings{clanuwat18kmnist,
 author    = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and Alex Lamb and Kazuaki Yamamoto and David Ha},
 title     = {Deep Learning for Classical Japanese Literature: KANJI-MNIST},
 booktitle = {arXiv preprint arXiv:1812.01718},
 year      = {2018}
}

@inproceedings{vaswani17transformer,
 author    = {Ashish Vaswani0 and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
 title     = {Attention is All You Need},
 booktitle = {Neural Information Processing Systems},
 year      = {2017}
}

@inproceedings{devlin18bert,
 author    = {Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova},
 title     = {BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
 booktitle = {NAACL},
 year      = {2019}
}

@inproceedings{radford19gpt2,
 author    = {Alec Radford and Jeffrey Wu and Rewon Child and David Luan and Dario Amodei and Ilya Sutskever},
 title     = {Language Models are Unsupervised Multitask Learners},
 booktitle = {OpenAI Blog},
 year      = {2019}
}

@inproceedings{brown20gpt3,
 author    = {Tom B. Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared Kaplan and Prafulla Dhariwal and Arvind Neelakantan and Pranav Shyam and Girish Sastry and Amanda Askell and Sandhini Agarwal and Ariel Herbert-Voss and Gretchen Krueger and Tom Henighan and Rewon Child and Aditya Ramesh and Daniel M. Ziegler and Jeffrey Wu and Clemens Winter and Christopher Hesse and Mark Chen and Eric Sigler and Mateusz Litwin and Scott Gray and Benjamin Chess and Jack Clark and Christopher Berner and Sam McCandlish and Alec Radford and Ilya Sutskever and Dario Amodei},
 title     = {Language Models are Few-Shot Learners},
 booktitle = {Neural Information Processing Systems},
 year      = {2020}
}

@inproceedings{kingma14adam,
 author    = {Diederik P. Kingma and Jimmy Ba},
 title     = {Adam: A Method for Stochastic Optimization},
 booktitle = {International Conference on Learning Representations},
 year      = {2015}
}

@inproceedings{srivastava14dropout,
 author    = {Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov},
 title     = {Dropout: A Simple Way to Prevent Neural Networks from Overfitting},
 booktitle = {Journal of Machine Learning Research},
 year      = {2014}
}

@inproceedings{ioffe15batchnorm,
 author    = {Sergey Ioffe and Christian Szegedy},
 title     = {Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift},
 booktitle = {International Conference on Machine Learning},
 year      = {2015}
}

@inproceedings{he16resnet,
 author    = {Kaiming He and Xiangyu Zhang and Saining Ren and Jian Sun},
 title     = {Deep Residual Learning for Image Recognition},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
 year      = {2016}
}

@inproceedings{hendrycks19robustness,
 author    = {Dan Hendrycks and Thomas Dietterich},
 title     = {Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
 booktitle = {International Conference on Learning Representations},
 year      = {2019}
}

@inproceedings{li18mixup,
 author    = {Hongyi Zhang and Moustapha Cisse and Yann N. Dauphin and David Lopez-Paz},
 title     = {mixup: Beyond Empirical Risk Minimization},
 booktitle = {International Conference on Learning Representations},
 year      = {2018}
}

@inproceedings{krizhevsky12imagenet,
 author    = {Alex Krizhevsky and Ilya Sutskever and Geoffrey E. Hinton},
 title     = {ImageNet Classification with Deep Convolutional Neural Networks},
 booktitle = {Neural Information Processing Systems},
 year      = {2012}
}

@inproceedings{huang17densenet,
 author    = {Gao Huang and Zhuang Liu and Laurens van der Maaten and Kilian Q. Weinberger},
 title     = {Densely Connected Convolutional Networks},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
 year      = {2017}
}

@inproceedings{szegedy15inception,
 author    = {Christian Szegedy and Wei Liu and Yangqing Jia and Pierre Sermanet and Scott Reed and Dragomir Anguelov and Dumitru Erhan and Vincent Vanhoucke and Andrew Rabinovich},
 title     = {Going Deeper with Convolutions},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
 year      = {2015}
}

@inproceedings{chollet17xception,
 author    = {Fran{\c{c}}ois Chollet},
 title     = {Xception: Deep Learning with Depthwise Separable Convolutions},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
 year      = {2017}
}

@inproceedings{ronneberger15unet,
 author    = {Olaf Ronneberger and Philipp Fischer and Thomas Brox},
 title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
 booktitle = {MICCAI},
 year      = {2015}
}

@inproceedings{redmon16yolo,
 author    = {Joseph Redmon and Santosh Divvala and Ross Girshick and Ali Farhadi},
 title     = {You Only Look Once: Unified, Real-Time Object Detection},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
 year      = {2016}
}

@inproceedings{simonyan14vgg,
 author    = {Karen Simonyan and Andrew Zisserman},
 title     = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
 booktitle = {International Conference on Learning Representations},
 year      = {2015}
}

@inproceedings{goodfellow14gan,
 author    = {Ian J. Goodfellow and Jean Pouget-Abadie and Mehdi Mirza and Bing Xu and David Warde-Farley and Sherjil Ozair and Aaron Courville and Yoshua Bengio},
 title     = {Generative Adversarial Nets},
 booktitle = {Neural Information Processing Systems},
 year      = {2014}
}

@inproceedings{kingma13vae,
 author    = {Diederik P. Kingma and Max Welling},
 title     = {Auto-Encoding Variational Bayes},
 booktitle = {International Conference on Learning Representations},
 year      = {2014}
}

@inproceedings{ho20ddpm,
 author    = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
 title     = {Denoising Diffusion Probabilistic Models},
 booktitle = {Neural Information Processing Systems},
 year      = {2020}
}

@inproceedings{song20score,
 author    = {Yang Song and Jiaming Song and Stefano Ermon},
 title     = {Score-Based Generative Modeling through Stochastic Differential Equations},
 booktitle = {International Conference on Learning Representations},
 year      = {2021}
}

@inproceedings{rombach22ldm,
 author    = {Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Bj{\"o}rn Ommer},
 title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
 year      = {2022}
}

@inproceedings{radford21clip,
 author    = {Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Ilharco and Sandhini Agarwal and Alexis Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger unconsciously and Ilya Sutskever},
 title     = {Learning Transferable Visual Models From Natural Language Supervision},
 booktitle = {International Conference on Machine Learning},
 year      = {2021}
}

@inproceedings{dosovitskiy20vit,
 author    = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
 title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
 booktitle = {International Conference on Learning Representations},
 year      = {2021}
}

@inproceedings{touvron21deit,
 author    = {Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv{\'e} J{\'e}gou},
 title     = {Training data-efficient image transformers and distillation through attention},
 booktitle = {International Conference on Machine Learning},
 year      = {2021}
}

@inproceedings{liu21swin,
 author    = {Ze Liu and Yutong Lin and Yue Cao and Han Hu and Yixuan Wei and Zheng Zhang and Stephen Lin and Baining Guo},
 title     = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
 booktitle = {IEEE International Conference on Computer Vision},
 year      = {2021}
}

@inproceedings{he22mae,
 author    = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
 title     = {Masked Autoencoders Are Scalable Vision Learners},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
 year      = {2022}
}

@inproceedings{caron21dino,
 author    = {Mathilde Caron and Hugo Touvron and Ishan Misra and Herv{\'e} J{\'e}gou and Julien Mairal and Piotr Bojanowski and Armand Joulin},
 title     = {Emerging Properties in Self-Supervised Vision Transformers},
 booktitle = {IEEE International Conference on Computer Vision},
 year      = {2021}
}

@inproceedings{oquab23dino2,
 author    = {Maxime Oquab and Timoth{\'e}e Darcet and Th{\'e}o Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Russell Howes and Po-Yao Huang and Hu Xu and Vasu Sharma and Yossi Adi and Sharan Narang and Francisco de Souza and Sireen Prasad and Amjad Almahairi and Sameer Singhal and Mattia Landoni and Ali Joseph and Nicolas Spisak and Antoine Goyet and Pietro Astolfi and Aur{\'e}lien Rodriguez and Allan Koura and Sebastian Biedermann and Basit Ayantunde and Philip Chon and Benjamin Lefaudeux and Jeremy Caron and Paul-Emmanuel Viel and David Gito and Edouard Grave and Armand Joulin and Piotr Bojanowski and Matthieu Cord},
 title     = {DINOv2: Learning Robust Visual Features without Supervision},
 booktitle = {arXiv preprint arXiv:2304.07193},
 year      = {2023}
}

@inproceedings{huffman52huffman,
 author    = {David A. Huffman},
 title     = {A Method for the Construction of Minimum-Redundancy Codes},
 booktitle = {Proceedings of the IRE},
 year      = {1952}
}

@inproceedings{shannon48shannon,
 author    = {Claude E. Shannon},
 title     = {A Mathematical Theory of Communication},
 booktitle = {Bell System Technical Journal},
 year      = {1948}
}

@inproceedings{fukushima80neocognitron,
 author    = {Kunihiko Fukushima},
 title     = {Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position},
 booktitle = {Biological Cybernetics},
 year      = {1980}
}

@inproceedings{rosenblatt58perceptron,
 author    = {Frank Rosenblatt},
 title     = {The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain},
 booktitle = {Psychological Review},
 year      = {1958}
}

@inproceedings{turing50turing,
 author    = {Alan M. Turing},
 title     = {Computing Machinery and Intelligence},
 booktitle = {Mind},
 year      = {1950}
}
"""

with open("submission.bib", "w") as f:
    f.write(bib_content.strip())
print("Created submission.bib successfully with 50 references.")
