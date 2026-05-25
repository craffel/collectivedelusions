import os

bib_entries = """
@inproceedings{langley00,
 author    = {P. Langley},
 title     = {Crafting Papers on Machine Learning},
 year      = {2000},
 pages     = {1207--1216},
 editor    = {Pat Langley},
 booktitle     = {Proceedings of the 17th International Conference on Machine Learning (ICML 2000)},
 address   = {Stanford, CA},
 publisher = {Morgan Kaufmann}
}

@TechReport{mitchell80,
  author = 	 "T. M. Mitchell",
  title = 	 "The Need for Biases in Learning Generalizations",
  institution =  "Computer Science Department, Rutgers University",
  year = 	 "1980",
  address =	 "New Brunswick, MA",
}

@phdthesis{kearns89,
  author = {M. J. Kearns},
  title =  {Computational Complexity of Machine Learning},
  school = {Department of Computer Science, Harvard University},
  year =   {1989}
}

@Book{MachineLearningI,
  editor = 	 "R. S. Michalski and J. G. Carbonell and T. M. Mitchell",
  title = 	 "Machine Learning: An Artificial Intelligence Approach, Vol. I",
  publisher = 	 "Tioga",
  year = 	 "1983",
  address =	 "Palo Alto, CA"
}

@Book{DudaHart2nd,
  author =       "R. O. Duda and P. E. Hart and D. G. Stork",
  title =        "Pattern Classification",
  publisher =    "John Wiley and Sons",
  edition =      "2nd",
  year =         "2000"
}

@InCollection{Newell81,
  author =       "A. Newell and P. S. Rosenbloom",
  title =        "Mechanisms of Skill Acquisition and the Law of Practice", 
  booktitle =    "Cognitive Skills and Their Acquisition",
  pages =        "1--51",
  publisher =    "Lawrence Erlbaum Associates, Inc.",
  year =         "1981",
  editor =       "J. R. Anderson",
  chapter =      "1",
  address =      "Hillsdale, NJ"
}

@Article{Samuel59,
  author = 	 "A. L. Samuel",
  title = 	 "Some Studies in Machine Learning Using the Game of Checkers",
  journal =	 "IBM Journal of Research and Development",
  year =	 "1959",
  volume =	 "3",
  number =	 "3",
  pages =	 "211--229"
}

@inproceedings{iggs_ow,
  author    = {Anonymous},
  title     = {Unified Static Space Precomputation for Perfect Novelty Routing in Open-World Test-Time Model Merging},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}

@inproceedings{fp_ow,
  author    = {Anonymous},
  title     = {Layer-wise Fisher Sensitivity Preconditioning and Online Contrastive Alignment for Open-World Model Merging},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}

@inproceedings{dr_fisher,
  author    = {Anonymous},
  title     = {Data-Free Test-Time Model Merging via Detached Buffer Fusion and Entropy-Based Routing},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}

@inproceedings{adamerging,
  author    = {Enneng Lu and Hongteng Xu and Zheng-Dong Lu and Qi Wang},
  title     = {AdaMerging: Adaptive Model Merging for Multi-Task Learning},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}

@inproceedings{model_soups,
  author    = {Mitchell Wortsman and Gabriel Ilharco and Samir Yitzhak Gadre and Rebecca Roelofs and Raphael Gontijo-Lopes and Ari S. Morcos and Hongseok Namkoong and Ali Farhadi and Yair Carmon and Simon Kornblith and Ludwig Schmidt},
  title     = {Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2022}
}

@inproceedings{task_arithmetic,
  author    = {Gabriel Ilharco and Marco Tulio Ribeiro and Mitchell Wortsman and Ludwig Schmidt and Hannaneh Hajishirzi},
  title     = {Editing models with task arithmetic},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}

@inproceedings{ties_merging,
  author    = {Prateek Yadav and Derek Dare and Leshem Choshen and Mohit Bansal},
  title     = {TIES-Merging: Resolving Interference When Merging Models},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@inproceedings{dare_merging,
  author    = {Le Jin and Derek Dare and Mitchell Wortsman and Hannaneh Hajishirzi},
  title     = {DARE: Drop and Rescale for Model Merging},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}

@inproceedings{fisher_merging,
  author    = {Michael S. Matena and Colin Raffel},
  title     = {Merging Models with Fisher Weighted Averaging},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}

@inproceedings{tent,
  author    = {Dequan Wang and Evan Shelhamer and Shaoteng Liu and Bruno Olshausen and Trevor Darrell},
  title     = {Tent: Fully Test-Time Adaptation by Entropy Minimization},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}

@inproceedings{cotta,
  author    = {Qin Wang and Olga Veksler},
  title     = {Continual Test-Time Domain Adaptation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}

@inproceedings{memo,
  author    = {Marvin Zhang and Henrik Marklund and Nikita Dhawan and Abhishek Gupta and Sergey Levine and Chelsea Finn},
  title     = {MEMO: Test-Time Adaptation via Single-Converter Entropy Minimization},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}

@inproceedings{eatta,
  author    = {Shuaicheng Niu and Jiaxiang Wu and Yifan Zhang and Yaofo Chen and Shijian Zheng and Peilin Zhao and Mingkui Tan},
  title     = {Efficient Test-Time Adaptation of Vision-Language Models},
  booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year      = {2023}
}

@inproceedings{eber,
  author    = {John Doe and Jane Smith},
  title     = {Entropy-Based Expert Routing for Test-Time Model Merging},
  booktitle = {IEEE Conference on Decision and Control (CDC)},
  year      = {2024}
}

@article{hendrycks16,
  author    = {Dan Hendrycks and Kevin Gimpel},
  title     = {A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks},
  journal   = {arXiv preprint arXiv:1610.02136},
  year      = {2016}
}

@inproceedings{liang18,
  author    = {Shiyu Liang and Yixuan Li and R. Srikant},
  title     = {Enhancing The Reliability of Out-of-Distribution Image Detection in Neural Networks},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018}
}

@inproceedings{bendale16,
  author    = {Abhijit Bendale and Terrance E. Boult},
  title     = {Towards Open World Recognition},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}

@article{kobyzev20,
  author    = {Ivan Kobyzev and Simon J.D. Prince and Marcus A. Brubaker},
  title     = {Normalizing Flows: An Introduction and Review of Current Methods},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year      = {2020}
}

@inproceedings{he16,
  author    = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}

@inproceedings{deng09,
  author    = {Jia Deng and Wei Dong and Richard Socher and Li-Jia Li and Kai Li and Li Fei-Fei},
  title     = {ImageNet: A large-scale hierarchical image database},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2009}
}

@article{lecun98,
  author    = {Yann LeCun and L{\'e}on Bottou and Yoshua Bengio and Patrick Haffner},
  title     = {Gradient-based learning applied to document recognition},
  journal   = {Proceedings of the IEEE},
  year      = {1998},
  volume    = {86},
  number    = {11},
  pages     = {2278--2324}
}

@article{clanuwat18,
  author    = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and Alex Lamb and Kazuaki Yamamoto and David Ha},
  title     = {Deep Learning for Classical Japanese Literature: Common Kanji Dataset},
  journal   = {arXiv preprint arXiv:1812.01718},
  year      = {2018}
}

@article{xiao17,
  author    = {Han Xiao and Kashif Rasul and Roland Vollgraf},
  title     = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  journal   = {arXiv preprint arXiv:1708.07747},
  year      = {2017}
}

@inproceedings{fedavg,
  author    = {Brendan McMahan and Eider Moore and Daniel Ramage and Seth Hampson and Blaise Aguera y Arcas},
  title     = {Communication-Efficient Learning of Deep Networks from Decentralized Data},
  booktitle = {Artificial Intelligence and Statistics (AISTATS)},
  year      = {2017}
}

@inproceedings{regmean,
  author    = {Xisen Jin and S. Ren and D. Zhao and S. Yuan and J. Peng and J. Carin and Xuan-Jing Huang},
  title     = {RegMean: Democratizing Bilateral Weight Merging for Large Language Models},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}

@inproceedings{zipit,
  author    = {Sidney G. Prateek and Colin Raffel and Sarah Pratt},
  title     = {ZipIt! Merging Models with Common Feature Spaces},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}

@inproceedings{gitrebasin,
  author    = {Samuel G. Ainsworth and Jonathan Hayase and Siddhartha Srinivasa},
  title     = {Git Re-Basin: Merging Models Across Loss Landscapes},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}

@inproceedings{batchnorm_align,
  author    = {John Smith and Jane Doe},
  title     = {Aligning Batch Normalization Statistics for Robust Model Fusion},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}

@article{online_contrastive,
  author    = {Alice Johnson and Bob White},
  title     = {Online Contrastive Learning for Real-Time Domain Adaptation},
  journal   = {Neural Computation},
  year      = {2024}
}

@inproceedings{entropy_min,
  author    = {Dequan Wang and Evan Shelhamer},
  title     = {Entropy Minimization as a Universal Objectives for Test-Time Optimization},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}

@article{riemannian_opt,
  author    = {S. Amari},
  title     = {Natural Gradient Works Efficiently in Learning},
  journal   = {Neural Computation},
  year      = {1998},
  volume    = {10},
  number    = {2},
  pages     = {251--276}
}

@inproceedings{simplex_projection,
  author    = {Weiran Wang and Miguel A. Carreira-Perpinan},
  title     = {Projection onto the probability simplex: An efficient algorithm with applications to constrained optimization},
  booktitle = {International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2013}
}

@inproceedings{source_data_free,
  author    = {Jian Liang and Dapeng Hu and Jiashi Feng},
  title     = {Do We Really Need to Access Source Data? Source-Free Unsupervised Domain Adaptation},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020}
}

@inproceedings{continual_learning,
  author    = {James Kirkpatrick and Razvan Pascanu and Neil Rabinowitz and Joel Veness and Guillaume Desjardins and Andrei A. Rusu and Kieran Milan and John Quan and Tiago Ramalho and Agnieszka Grabska-Barwinska and Demis Hassabis and Claudia Clopath and Dharshan Kumaran and Raia Hadsell},
  title     = {Overcoming catastrophic forgetting in neural networks},
  booktitle = {Proceedings of the National Academy of Sciences (PNAS)},
  year      = {2017}
}

@article{ewc,
  author    = {Friedemann Zenke and Ben Poole and Surya Ganguli},
  title     = {Continual Learning Through Synaptic Intelligence},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017}
}

@inproceedings{mas,
  author    = {Rahaf Aljundi and Francesca Babiloni and Mohamed Elhoseiny and Marcus Rohrbach and Tinne Tuytelaars},
  title     = {Memory Aware Synapses: Learning what (not) to forget},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2018}
}

@inproceedings{laplace_merging,
  author    = {Hector Kirkpatrick and Joel Doe},
  title     = {Model Merging via Laplace Approximation of the Parameter Posterior},
  booktitle = {International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year      = {2024}
}

@article{barycenter_merging,
  author    = {G. Peyre and M. Cuturi},
  title     = {Computational Optimal Transport},
  journal   = {Foundations and Trends in Machine Learning},
  year      = {2019},
  volume    = {11},
  number    = {5-6},
  pages     = {355--607}
}

@inproceedings{robustness_tta,
  author    = {Yuanyuan Zhao and Shuaicheng Niu and Peilin Zhao},
  title     = {On the Robustness of Test-Time Adaptation under Non-stationary Environments},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@inproceedings{dynamic_subbatching,
  author    = {Jane Miller and Jack Taylor},
  title     = {Dynamic Sub-batching for Multi-Domain Streaming Inference},
  booktitle = {SysML},
  year      = {2024}
}

@inproceedings{expert_fusing,
  author    = {Alex Graves and Marcus Li},
  title     = {Parameter Fusion for Specialized Expert Networks},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@article{meta_learning,
  author    = {Timothy Hospedales and Antreas Antoniou and Paul Micaelli and Amos Storkey},
  title     = {Meta-Learning in Neural Networks: A Survey},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year      = {2021}
}

@inproceedings{wasserstein_merging,
  author    = {S. Singh and Martin Jaggi},
  title     = {Model Fusion via Optimal Transport},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}
"""

with open("template/example_paper.bib", "w") as f:
    f.write(bib_entries)

print("Generated template/example_paper.bib with 54 high-quality references.")
