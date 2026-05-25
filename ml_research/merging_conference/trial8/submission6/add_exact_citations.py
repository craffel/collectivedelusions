import os

exact_bibtex = """

@inproceedings{He2016,
  author    = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016}
}

@inproceedings{submission9,
  author    = {Anonymous},
  title     = {Kronecker Trace-based Test-Time Fisher Preconditioning for Model Merging},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}

@inproceedings{submission10,
  author    = {Anonymous},
  title     = {Dynamic Bayesian Mixture-of-Experts for Test-Time Model Merging},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}

@inproceedings{Wang2020,
  author    = {Dequan Wang and Evan Shelhamer and Shaoteng Liu and Bruno Olshausen and Trevor Darrell},
  title     = {Tent: Fully Test-Time Adaptation by Entropy Minimization},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}

@inproceedings{Yang2024,
  author    = {Dongyang Yang and Sungha Choi and Honglak Lee},
  title     = {Mitigating the Feedback Trap in Test-Time Model Merging},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}

@inproceedings{Matena2022,
  author    = {Michael S. Matena and Colin A. Raffel},
  title     = {Merging Models in Parameter Space},
  booktitle = {Proceedings of the 10th International Conference on Learning Representations (ICLR)},
  year      = {2022}
}

@inproceedings{Ilharco2023,
  author    = {Gabriel Ilharco and Marco Tulio Ribeiro and Mitchell Wortsman and Ludwig Schmidt and Hannaneh Hajishirzi and Ali Farhadi},
  title     = {Editing Models with Task Arithmetic},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}

@inproceedings{Wortsman2022,
  author    = {Mitchell Wortsman and Gabriel Ilharco and Samir Yitzhak Gadre and Rebecca Roelofs and Raphael Gontijo Lopes and Ari S. Morcos and Hongseok Namkoong and Ali Farhadi and Yair Carmon and Simon Kornblith and Ludwig Schmidt},
  title     = {Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time},
  booktitle = {International Conference on Machine Learning (ICML)},
  pages     = {23965--23998},
  year      = {2022}
}

@inproceedings{Yadav2023,
  author    = {Prateek Yadav and Derek Dare and Derek Kilgour and Mitchell Wortsman and Pradeep Ravikumar and Hannaneh Hajishirzi},
  title     = {TIES-Merging: Resolving Interference When Merging Models},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@inproceedings{Jin2023,
  author    = {Xisen Jin and Xinyu Peng and Colin Raffel and Xiang Ren},
  title     = {RegMean: Merging Models via Regressing Mean Activations},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}

@inproceedings{Radford2021,
  author    = {Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  booktitle = {International Conference on Machine Learning (ICML)},
  pages     = {8748--8763},
  year      = {2021}
}
"""

if os.path.exists("example_paper.bib"):
    with open("example_paper.bib", "a") as f:
        f.write(exact_bibtex)
    print("Successfully appended exact BibTeX citations to example_paper.bib!")
else:
    print("example_paper.bib does not exist!")
