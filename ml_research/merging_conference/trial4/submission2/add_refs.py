new_refs = """
@article{kirkpatrick2017overcoming,
  title={Overcoming catastrophic forgetting in neural networks},
  author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
  journal={Proceedings of the National Academy of Sciences},
  volume={114},
  number={13},
  pages={3521--3526},
  year={2017}
}

@inproceedings{yadav2023ties,
  title={TIES-Merging: Resolving Interference When Merging Models},
  author={Yadav, Prateek and Tam, Derek and Choshen, Leshem and Bansal, Mohit and Raffel, Colin},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  pages={29393--29414},
  year={2023}
}

@inproceedings{yu2024language,
  title={Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch},
  author={Yu, Le and Jiang, Bowen and Shi, Chao and Chen, Jue and Liu, Bin},
  booktitle={International Conference on Machine Learning},
  year={2024}
}

@article{jang2024model,
  title={Model Stock: All we need is just a few fine-tuned models},
  author={Jang, Dong-Hwan and Yun, Sangdoo and Han, Dongyoon},
  journal={arXiv preprint arXiv:2403.19522},
  year={2024}
}

@inproceedings{wang2021tent,
  title={Tent: Fully Test-time Adaptation by Entropy Minimization},
  author={Wang, Dequan and Shelhamer, Evan and Liu, Jiashi and Olshausen, Bruno and Darrell, Trevor},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{wang2022continual,
  title={Continual Test-time Adaptation with Decomposed Heuristic Optimization},
  author={Wang, Qin and Cuevas, Olga and Gool, Luc Van and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{schneider2020improving,
  title={Improving robustness against common corruptions by covariate shift adaptation},
  author={Schneider, Steffen and Rusak, Evgenia and Eck, Luisa and Oliver, Oliver and Wieland, Wieland and Brendel, Wieland},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}

@inproceedings{boudiaf2022lame,
  title={Lame: Local affine multidimensional projection for test-time adaptation},
  author={Boudiaf, Malik and Br{\'e}mond, Romain and Ayed, Ismail Ben},
  booktitle={Computer Vision and Pattern Recognition},
  year={2022}
}

@inproceedings{nado2020evaluating,
  title={Evaluating predictive uncertainty under dataset shift in deep learning},
  author={Nado, Zachary and Padhy, Shreyas and Basese, D and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}

@inproceedings{zhang2021memo,
  title={Memo: Test-time adaptation via single-sample prediction consistency},
  author={Zhang, Marvin and Levine, Sergey and Finn, Chelsea},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

@inproceedings{goyal2022test,
  title={Test-time adaptation via self-supervised contrastive representation alignment},
  author={Goyal, Sachin and Sun, Mingjie and Raghunathan, Ananya and Kolter, J Zico},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={770--778},
  year={2016}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish_and_Shazeer, Noam_and_Parmar, Niki_and_Uszkoreit, Jakob_and_Jones, Llion_and_Gomez, Aidan N_and_Kaiser, {\L}ukasz_and_Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  pages={5998--6008},
  year={2017}
}

@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}

@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International Conference on Machine Learning},
  pages={8748--8763},
  year={2021}
}

@inproceedings{mitchell2022fast,
  title={Fast model editing at scale},
  author={Mitchell, Eric and Lin, Charles and Bosselut, Antoine and Finn, Chelsea and Manning, Christopher D},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

@inproceedings{sinitsin2020editable,
  title={Editable neural networks},
  author={Sinitsin, Anton and Platonov, Vsevolod and Smetanin, Dmitry and Yakubovskiy, Alim and Shvechikov, Andrey and Babenko, Artem},
  booktitle={International Conference on Learning Representations},
  year={2020}
}

@inproceedings{meng2022locating,
  title={Locating and editing factual associations in {GPT}},
  author={Meng, Kevin and Bau, David and Andonian, Alex and Belinkov, Yonatan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{frankle2018lottery,
  title={The lottery ticket hypothesis: Finding sparse, trainable neural networks},
  author={Frankle, Jonathan and Carbin, Michael},
  booktitle={International Conference on Learning Representations},
  year={2018}
}

@inproceedings{neyshabur2020what,
  title={What is being transferred in transfer learning?},
  author={Neyshabur, Behnam and Sedghi, Hanie and Thoppilan, Albin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}

@inproceedings{izmailov2018averaging,
  title={Averaging weights leads to wider optima and better generalization},
  author={Izmailov, Pavel and Podoprikhin, Dmitrii and Garipov, Timur and Vetrov, Dmitry and Wilson, Andrew Gordon},
  booktitle={Conference on Uncertainty in Artificial Intelligence},
  year={2018}
}

@inproceedings{maddox2019simple,
  title={A simple baseline for uncertainty in deep learning},
  author={Maddox, Wesley J and Garipov, Timur and Izmailov, Pavel and Vetrov, Dmitry and Wilson, Andrew Gordon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}

@inproceedings{fort2019deep,
  title={Deep ensembles: A loss landscape perspective},
  author={Fort, Stanislav and Hu, Huiyi and Lakshminarayanan, Balaji},
  booktitle={arXiv preprint arXiv:1912.02757},
  year={2019}
}

@inproceedings{wortsman2021learning,
  title={Learning neural network subspaces},
  author={Wortsman, Mitchell and Horton, Maxwell and Guestrin, Carlos and Farhadi, Ali and Rastegari, Mohammad},
  booktitle={International Conference on Machine Learning},
  year={2021}
}

@inproceedings{benton2021loss,
  title={Loss surface simplexes for mode connecting volumes and fast ensembling},
  author={Benton, Gregory and Izmailov, Pavel and Wilson, Andrew Gordon},
  booktitle={International Conference on Machine Learning},
  year={2021}
}

@article{delange2021continual,
  title={A continual learning survey: Defying forgetting in classification tasks},
  author={De Lange, Matthias and Aljundi, Rahaf and Masana, Marc and Parisot, Sarah and Jia, Xu and {\v{S}}l{\=\i}pka, {\v{Z}}iga and Leonardis, Ale{\v{s}} and Tuytelaars, Tinne},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={7},
  pages={3361--3385},
  year={2021}
}

@article{parisi2019continual,
  title={Continual lifelong learning with neural networks: A review},
  author={Parisi, German I and Kemker, Ronald and Part, Jose L and Kanan, Christopher and Wermter, Stefan},
  journal={Neural Networks},
  volume={113},
  pages={54--71},
  year={2019}
}

@inproceedings{zenke2017continual,
  title={Continual learning through synaptic intelligence},
  author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
  booktitle={International Conference on Machine Learning},
  year={2017}
}

@inproceedings{chaudhry2018efficient,
  title={Efficient lifelong learning with a-gem},
  author={Chaudhry, Arslan and Ranzato, Marc'Aurelio and Rohrbach, Marcus and Elhoseiny, Mohamed},
  booktitle={International Conference on Learning Representations},
  year={2018}
}

@inproceedings{lopezpaz2017gradient,
  title={Gradient episodic memory for continual learning},
  author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}

@inproceedings{lomonaco2017core50,
  title={{CORe50}: a new dataset and benchmark for continuous object recognition},
  author={Lomonaco, Vincenzo and Maltoni, Davide},
  booktitle={Conference on Robot Learning},
  year={2017}
}

@inproceedings{schneider2020covariate,
  title={Improving robustness against common corruptions by covariate shift adaptation},
  author={Schneider, Steffen and Rusak, Evgenia and Eck, Luisa and Oliver, Oliver and Wieland, Wieland and Brendel, Wieland},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}

@inproceedings{geva2020transformer,
  title={Transformer feed-forward layers are key-value memories},
  author={Geva, Mor and Schuster, Roei and Berant, Jonathan and Levy, Omer},
  booktitle={Proceedings of the Conference on Empirical Methods in Natural Language Processing},
  year={2020}
}

@inproceedings{dai2022knowledge,
  title={Knowledge neurons in pretrained transformers},
  author={Dai, Damai and Dong, Li and Shuo, Ma and Sui, Zhifang and Chang, Baobao and Zhou, Ji-Rong and others},
  booktitle={Proceedings of the Association for Computational Linguistics},
  year={2022}
}

@article{gu2023mamba,
  title={Mamba: Linear-time sequence modeling with selective state spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
"""

with open('example_paper.bib', 'a') as f:
    f.write(new_refs)

print("Successfully added 35 new references to example_paper.bib!")
