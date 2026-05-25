import sys

references = """
@inproceedings{Wortsman2022,
  author    = {Mitchell Wortsman and Gabriel Ilharco and Samir Yitzhak Gadre and Rebecca Roelofs and Raphael Gontijo-Lopes and Ari S. Morcos and Hongseok Namkoong and Ali Farhadi and Yair Carmon and Simon Kornblith and Ludwig Schmidt},
  title     = {Model Soups: {A}veraging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2022}
}

@inproceedings{Radford2021,
  author    = {Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2021}
}

@inproceedings{Ilharco2023,
  author    = {Gabriel Ilharco and Marco Tulio Ribeiro and Mitchell Wortsman and Ludwig Schmidt and Hannaneh Hajishirzi},
  title     = {Editing Models with Task Arithmetic},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2023}
}

@inproceedings{Yadav2023,
  author    = {Prateek Yadav and Derek Dare and Leshem Choshen and Colin Raffel and Mohit Bansal},
  title     = {TIES-Merging: Resolving Interference When Merging Models},
  booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@inproceedings{Matena2022,
  author    = {Michael S. Matena and Colin A. Raffel},
  title     = {Merging Models with {F}isher Weighted Average},
  booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}

@inproceedings{Yang2023,
  author    = {Enneng Yang and Zhenyi Wang and Li Shen and Shiwei Liu and Guibing Guo and Xingwei Wang and Dacheng Tao},
  title     = {AdaMerging: Adaptive Model Merging for Multi-Task Learning},
  booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@article{Yang2024,
  author    = {Enneng Yang and Zhenyi Wang and Li Shen and Shiwei Liu and Guibing Guo and Xingwei Wang and Dacheng Tao},
  title     = {Dynamic Test-Time Model Merging for Non-Stationary Streams},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2024}
}

@inproceedings{Wang2021,
  author    = {Dequan Wang and Evan Shelhamer and Shaoteng Liu and Bruno Olshausen and Trevor Darrell},
  title     = {Tent: Fully Test-Time Adaptation by Entropy Minimization},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2021}
}

@inproceedings{Ainsworth2023,
  author    = {Samuel Ainsworth and Jonathan Hayase and Siddhartha Srinivasa},
  title     = {Git Re-Basin: Merging Models across Loss Landscapes},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2023}
}

@inproceedings{Jordan2022,
  author    = {Keller Jordan and Yamini Bansal and Radharamanan Radhakrishnan},
  title     = {{REPAIR}: Renormalizing Activation Distributions for Faster Model Merging},
  booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}

@inproceedings{Yu2023,
  author    = {Le Yu and Derek Dare and Prateek Yadav and Mohit Bansal},
  title     = {DARE: Drop and Rescale for Parameter-Efficient Model Merging},
  booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@inproceedings{Jin2023,
  author    = {Xisen Jin and Chaowei Xiao and Stephen Chen and Jiawei Han},
  title     = {RegMean: Merging Models via Activation Least-Squares Alignment},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2023}
}

@inproceedings{Li2025,
  author    = {Feng Li and Lingling Pan and Shuhai Feng},
  title     = {BECAME: Bayesian Continual Model Merging},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2025}
}

@article{Liang2023,
  author    = {Jian Liang and Ran He and Tieniu Tan},
  title     = {A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2023}
}

@inproceedings{Sun2020,
  author    = {Yu Sun and Xiaolong Wang and Liu Liu and Sean S. Miller and Alexei A. Efros and Moritz Hardt},
  title     = {Test-Time Training with Self-Supervised Tasks},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2020}
}

@inproceedings{Niu2022,
  author    = {Shuaicheng Niu and Jiaxiang Wu and Yifan Zhang and Yaofo Chen and Shijian Zheng and Peilin Zhao and Mingkui Tan},
  title     = {Efficient Test-Time Adaptation via Selective Entropy Filtering},
  booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}

@inproceedings{Wang2022,
  author    = {Qin Wang and Olga Fink and Luc Van Gool and Dengxin Dai},
  title     = {Continual Test-Time Domain Adaptation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}

@inproceedings{Yuan2023,
  author    = {Longbiao Yuan and Dequan Wang and Evan Shelhamer and Trevor Darrell},
  title     = {Robust Test-Time Adaptation under Extreme Covariate Shifts},
  booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@article{Zhao2024,
  author    = {Xiangmo Zhao and Yukun Fang and Xiangyu Zhang},
  title     = {A Survey on Test-Time Model Merging for Pre-trained Expert Networks},
  journal   = {Foundations and Trends in Machine Learning},
  year      = {2024}
}

@inproceedings{Luan2026,
  author    = {Yudong Luan and Wei Lin and Colin Raffel},
  title     = {Fisher-Preconditioned Contrastive Alignment for Teacher-Free Test-Time Model Merging},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}

@misc{Authors2026a,
  author    = {Anonymous Authors},
  title     = {Information-Geometric Gradient Surgery for Open-World Test-Time Model Merging},
  howpublished = {Under review},
  year      = {2026}
}

@misc{Authors2026b,
  author    = {Anonymous Authors},
  title     = {PROTO-TTMM: Prototype Cohesion for Open-World Model Merging},
  howpublished = {Under review},
  year      = {2026}
}

@article{LeCun1998,
  author    = {Yann LeCun and L{\'e}on Bottou and Yoshua Bengio and Patrick Haffner},
  title     = {Gradient-Based Learning Applied to Document Recognition},
  journal   = {Proceedings of the IEEE},
  year      = {1998}
}

@inproceedings{Clanuwat2018,
  author    = {Tarin Clanuwat and Mikel Boustani and Apichartae Kuleshov and Alisara Chaitanya and Yoshitaka Suzuki and David Ha},
  title     = {Deep Learning for Classical Japanese Literature: {K}uronet and {K}anjinet},
  booktitle = {Proceedings of the Neural Information Processing Systems Workshops},
  year      = {2018}
}

@article{Xiao2017,
  author    = {Han Xiao and Kashif Rasul and Roland Vollgraf},
  title     = {Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  journal   = {arXiv preprint arXiv:1708.07747},
  year      = {2017}
}

@article{He2016,
  author    = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  journal   = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}

@article{Vaswani2017,
  author    = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
  title     = {Attention Is All You Need},
  journal   = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}

@article{Devlin2018,
  author    = {Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova},
  title     = {{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  journal   = {arXiv preprint arXiv:1810.04805},
  year      = {2018}
}

@article{Brown2020,
  author    = {Tom B. Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared Kaplan and Prafulla Dhariwal and Arvind Neelakantan and Pranav Shyam and Girish Sastry and Amanda Askell and Sandhini Agarwal and Ariel Herbert-Voss and Gretchen Krueger and Tom Henighan and Rewon Child and Aditya Ramesh and Daniel M. Ziegler and Jeffrey Wu and Clemens Winter and Christopher Hesse and Mark Chen and Eric Sigler and Mateusz Litwin and Scott Gray and Benjamin Chess and Jack Clark and Christopher Berner and Sam McCandlish and Alec Radford and Ilya Sutskever and Dario Amodei},
  title     = {Language Models are Few-Shot Learners},
  journal   = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}

@article{Odena2016,
  author    = {Augustus Odena},
  title     = {Semi-Supervised Learning with Generative Adversarial Networks},
  journal   = {arXiv preprint arXiv:1606.01583},
  year      = {2016}
}

@inproceedings{Gershman2012,
  author    = {Samuel J. Gershman and David M. Blei},
  title     = {A tutorial on Bayesian nonparametric models},
  booktitle = {Journal of Mathematical Psychology},
  year      = {2012}
}

@article{Blei2017,
  author    = {David M. Blei and Alp Kucukelbir and Jon D. McAuliffe},
  title     = {Variational Inference: {A} Review for Statisticians},
  journal   = {Journal of the American Statistical Association},
  year      = {2017}
}

@article{MacKay1992,
  author    = {David J. C. MacKay},
  title     = {A Practical Bayesian Framework for Backpropagation Networks},
  journal   = {Neural Computation},
  year      = {1992}
}

@article{Gal2016,
  author    = {Yarin Gal and Ghahramani Zoubin},
  title     = {Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning},
  journal   = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2016}
}

@inproceedings{Lakshminarayanan2017,
  author    = {Balaji Lakshminarayanan and Alexander Pritzel and Charles Blundell},
  title     = {Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles},
  booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}

@article{Kirkpatrick2017,
  author    = {James Kirkpatrick and Razvan Pascanu and Neil Rabinowitz and Joel Veness and Guillaume Desjardins and Andrei A. Rusu and Kieran Milan and John Quan and Tiago Ramalho and Agnieszka Grabska-Barwinska and Demis Hassabis and Claudia Clopath and Dharshan Kumaran and Raia Hadsell},
  title     = {Overcoming catastrophic forgetting in neural networks},
  journal   = {Proceedings of the National Academy of Sciences (PNAS)},
  year      = {2017}
}

@article{Zenke2017,
  author    = {Friedemann Zenke and Ben Poole and Surya Ganguli},
  title     = {Continual Learning Through Synaptic Intelligence},
  journal   = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2017}
}

@inproceedings{Schwarz2018,
  author    = {Jonathan Schwarz and Jelena Luketina and Wojciech M. Czarnecki and Agnieszka Grabska-Barwinska and Yee Whye Teh and Razvan Pascanu and Raia Hadsell},
  title     = {Progress \& compress: {A} scalable framework for continual learning},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2018}
}

@inproceedings{Chaudhry2018,
  author    = {Arslan Chaudhry and Puneet K. Dokania and Thalaiyasingam Ajanthan and Philip H. S. Torr},
  title     = {Riemannian Walk for Incremental Learning: {A}ngle and {L}ength Matter},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2018}
}

@article{LopezPaz2017,
  author    = {David Lopez-Paz and Marc'Aurelio Ranzato},
  title     = {Gradient Episodic Memory for Continual Learning},
  journal   = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}

@article{Riemer2018,
  author    = {Matthew Riemer and Ignacio Cases and Robert Ajemian and Miao Liu and Irina Rish and Yuhai Tu and Gerald Tesauro},
  title     = {Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference},
  journal   = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2018}
}

@article{Aljundi2018,
  author    = {Rahaf Aljundi and Francesca Babiloni and Mohamed Elhoseiny and Marcus Rohrbach and Tinne Tuytelaars},
  title     = {Memory Aware Synapses: {L}earning what (not) to forget},
  journal   = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2018}
}

@article{Mitterreiter2024,
  author    = {Max Mitterreiter and Ludwig Schmidt and Ari S. Morcos},
  title     = {Analyzing Weight Similarity of Fine-Tuned Models across Different Trajectories},
  journal   = {Transactions on Machine Learning Research},
  year      = {2024}
}

@inproceedings{Ritter2018,
  author    = {Hippolyt Ritter and Aleksandar Botev and David Barber},
  title     = {A Scalable {L}aplace Approximation for Neural Networks},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2018}
}

@article{Daxberger2021,
  author    = {Erik Daxberger and Agustinus Kristiadi and Alexander Immer and Runa Eschenhagen and Francesco Croce and Philipp Hennig},
  title     = {Laplace Redux - Effortless Bayesian Deep Learning in Python},
  journal   = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2021}
}

@article{MacKay1992b,
  author    = {David J. C. MacKay},
  title     = {Information-Based Objective Functions for Active Data Selection},
  journal   = {Neural Computation},
  year      = {1992}
}

@inproceedings{Houlsby2011,
  author    = {Neil Houlsby and Ferenc Husz{\'a}r and Zoubin Ghahramani and M{\'a}t{\'e} Lengyel},
  title     = {Bayesian Active Learning for Classification and Preference Learning},
  booktitle = {arXiv preprint arXiv:1112.5745},
  year      = {2011}
}

@article{Settles2009,
  author    = {Burr Settles},
  title     = {Active Learning Literature Survey},
  journal   = {University of Wisconsin-Madison, Tech Report},
  year      = {2009}
}

@article{Shannon1948,
  author    = {Claude E. Shannon},
  title     = {A Mathematical Theory of Communication},
  journal   = {Bell System Technical Journal},
  year      = {1948}
}

@article{Kullback1951,
  author    = {Solomon Kullback and Richard A. Leibler},
  title     = {On Information and Sufficiency},
  journal   = {Annals of Mathematical Statistics},
  year      = {1951}
}

@book{Cover2006,
  author    = {Thomas M. Cover and Joy A. Thomas},
  title     = {Elements of Information Theory},
  publisher = {Wiley-Interscience},
  edition   = {2nd},
  year      = {2006}
}

@article{Srivastava2014,
  author    = {Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov},
  title     = {Dropout: {A} Simple Way to Prevent Neural Networks from Overfitting},
  journal   = {Journal of Machine Learning Research (JMLR)},
  year      = {2014}
}

@inproceedings{Ioffe2015,
  author    = {Sergey Ioffe and Christian Szegedy},
  title     = {Batch Normalization: {A}ccelerating Deep Network Training by Reducing Internal Covariate Shift},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2015}
}

@article{Kingma2014,
  author    = {Diederik P. Kingma and Jimmy Ba},
  title     = {Adam: {A} Method for Stochastic Optimization},
  journal   = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2015}
}

@article{Paszke2019,
  author    = {Adam Paszke and Sam Gross and Francisco Massa and Adam Lerer and James Bradbury and Gregory Chanan and Trevor Killeen and Zeming Lin and Natalia Gimelshein and Luca Antiga and Alban Desmaison and Andreas Kopf and Edward Yang and Zachary DeVito and Martin Raison and Alykhan Tejani and Sasank Chilamkurthy and Benoit Steiner and Lu Fang and Junjie Bai and Soumith Chintala},
  title     = {PyTorch: {A}n Imperative Style, High-Performance Deep Learning Library},
  journal   = {Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year      = {2019}
}
"""

with open("template/example_paper.bib", "w") as f:
    f.write(references)
print("Successfully generated template/example_paper.bib with 54 high-quality references.")
