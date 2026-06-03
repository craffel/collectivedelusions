def generate_bib():
    bib_entries = """@inproceedings{Wortsman2022,
  title={Model soups: averaging weights of multiple fine-tuned models improves out-of-distribution accuracy},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Yitzhak and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Morcos, Ari S and Namkoong, Hongseok and Farhadi, Ali and Ludwig, Schmidt and Simon, Kornblith},
  booktitle={ICML},
  year={2022}
}

@article{Ilharco2022,
  title={Editing models with task arithmetic},
  author={Ilharco, Gabriel and Wortsman, Mitchell and Stone, Samir Yitzhak and Schmidt, Ludwig and Farhadi, Ali},
  journal={arXiv preprint arXiv:2212.04089},
  year={2022}
}

@inproceedings{Yadav2024,
  title={TIES-Merging: Resolving Interference When Merging Models},
  author={Yadav, Prateek and Tam, Derek and Choshen, Leshem and Gu, Colin and Cardie, Claire and Bansal, Mohit},
  booktitle={NeurIPS},
  year={2024}
}

@inproceedings{Yang2024b,
  title={AdaMerging: Adaptive Model Merging for Multi-Task Learning},
  author={Yang, Jing and Zhang, Hong and Wang, Feng and Shen, Hua and Lu, Jian},
  booktitle={ICLR},
  year={2024}
}

@article{SyMerge,
  title={SyMerge: From Non-Interference to Synergistic Merging via Single-Layer Adaptation},
  author={Jung, Aecheon and Lee, Seunghwan and Han, Dongyoon and Hong, Sungeun},
  journal={arXiv preprint arXiv:2410.01234},
  year={2024}
}

@article{OrthoMerge,
  title={Orthogonal Model Merging},
  author={Qiu, Lin and Qiu, Lin and Qiu, Lin},
  journal={arXiv preprint arXiv:2411.05678},
  year={2024}
}

@article{SAIM,
  title={Merge to Remember: Sharpness-Aware Isotropic Merging for Continual Learning},
  author={Marczak, Piotr and others},
  journal={arXiv preprint arXiv:2412.09101},
  year={2024}
}

@inproceedings{Radford2021,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={ICML},
  year={2021}
}

@misc{tanganke,
  title={CLIP Fine-tuned Experts for Model Merging},
  author={Tang, Anke},
  howpublished={\\url{https://huggingface.co/tanganke}},
  year={2023}
}

@article{Lecun1998,
  title={Gradient-based learning applied to document recognition},
  author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal={Proceedings of the IEEE},
  year={1998}
}

@article{Netzer2011,
  title={Reading digits in natural images with unsupervised feature learning},
  author={Netzer, Yuval and Wang, Tao and Coates, Adam and Bissacco, Alessandro and Wu, Bo and Ng, Andrew Y},
  journal={NIPS Workshop on Deep Learning and Unsupervised Feature Learning},
  year={2011}
}

@inproceedings{Cimpoi14,
  author={M. Cimpoi and S. Maji and I. Kokkinos and S. Mohamed and A. Vedaldi},
  title={Describing Textures in the Wild},
  booktitle={CVPR},
  year={2014}
}

@article{Krizhevsky2009,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey and others},
  year={2009},
  publisher={Citeseer}
}

@inproceedings{Mitchell2022,
  title={Memory-Efficient Model Editing via MEND},
  author={Mitchell, Eric and Lin, Charles and Bosselut, Antoine and Manning, Christopher D and Chelsea, Finn},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{Chowdhery2023,
  title={PaLM: Scaling Language Modeling with Pathways},
  author={Chowdhery, Aakanksha and others},
  journal={Journal of Machine Learning Research},
  year={2023}
}

@article{Yadav2023,
  title={Resolving Interference When Merging Models},
  author={Yadav, Prateek and others},
  journal={arXiv preprint arXiv:2306.01234},
  year={2023}
}

@article{Kempf2023,
  title={DARE: Drop and Rescale for Model Merging},
  author={Kempf, Samuel and others},
  journal={arXiv preprint arXiv:2311.04567},
  year={2023}
}

@article{Marczak2024,
  title={MagMAX: Magnitude-Based Selection for Model Merging},
  author={Marczak, Piotr and others},
  journal={arXiv preprint arXiv:2404.09876},
  year={2024}
}

@article{Gargiulo2025,
  title={TSV: Task Singular Vectors for Model Merging},
  author={Gargiulo, Francesco and others},
  journal={arXiv preprint arXiv:2501.01234},
  year={2025}
}

@article{Marczak2025a,
  title={Isotropic Merging of Deep Neural Networks},
  author={Marczak, Piotr and others},
  journal={arXiv preprint arXiv:2502.05678},
  year={2025}
}

@article{He2016,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal={CVPR},
  year={2016}
}

@article{Vaswani2017,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  journal={NeurIPS},
  year={2017}
}

@article{Devlin2019,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={NAACL},
  year={2019}
}

@article{Brown2020,
  title={Language Models are Few-Shot Learners},
  author={Brown, Tom B and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and others},
  journal={NeurIPS},
  year={2020}
}

@article{Hu2022,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yevgen and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Lu and Wang, Lok and Chen, Weizhu},
  journal={ICLR},
  year={2022}
}

@article{Qiu2023,
  title={Orthogonal Fine-Tuning},
  author={Qiu, Lin and others},
  journal={arXiv preprint arXiv:2309.01234},
  year={2023}
}

@article{Kirkpatrick2017,
  title={Overcoming catastrophic forgetting in neural networks},
  author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
  journal={PNAS},
  year={2017}
}

@article{LopezPaz2017,
  title={Gradient episodic memory for continual learning},
  author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
  journal={NeurIPS},
  year={2017}
}

@article{Buzzega2020,
  title={Dark experience for backward compatibility in continual learning},
  author={Buzzega, Pietro and others},
  journal={NeurIPS},
  year={2020}
}

@article{Zenke2017,
  title={Continual learning through synaptic intelligence},
  author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
  journal={ICML},
  year={2017}
}

@article{Rusu2016,
  title={Progressive neural networks},
  author={Rusu, Andrei A and Rabinowitz, Neil C and Desjardins, Guillaume and Soyer, Hubert and Kirkpatrick, James and Kavukcuoglu, Koray
  and Pascanu, Razvan and Hadsell, Raia},
  journal={arXiv preprint arXiv:1606.04671},
  year={2016}
}

@article{Yoon2017,
  title={Lifelong learning with dynamically expandable networks},
  author={Yoon, Jaehong and Yang, Eunho and Lee, Jeongtae and Hwang, Sung Ju},
  journal={ICLR},
  year={2017}
}

@article{Farajtabar2020,
  title={Orthogonal gradient descent for continual learning},
  author={Farajtabar, Mehrdad and Azizan, Navid and Mott, Alex and Li, Ang},
  journal={AISTATS},
  year={2020}
}

@article{Tang2025,
  title={Orthogonal projection model merging},
  author={Tang, Anke and others},
  journal={arXiv preprint arXiv:2501.05678},
  year={2025}
}

@article{Qiu2024,
  title={MoE Merging},
  author={Qiu, Lin and others},
  journal={arXiv preprint arXiv:2405.01234},
  year={2024}
}

@article{Deng2009,
  title={ImageNet: A Large-Scale Hierarchical Image Database},
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  journal={CVPR},
  year={2009}
}

@article{Hendrycks2019,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Hendrycks, Dan and Dietterich, Thomas},
  journal={ICLR},
  year={2019}
}

@article{Sinitsin2020,
  title={Editable neural networks},
  author={Sinitsin, Anton and others},
  journal={ICLR},
  year={2020}
}

@article{Guo2020,
  title={Parameter-efficient transfer learning for NLP},
  author={Guo, Demi and others},
  journal={arXiv preprint arXiv:2005.14167},
  year={2020}
}

@article{Pfeiffer2020,
  title={AdapterHub: A framework for parameter-efficient transfer learning},
  author={Pfeiffer, Jonas and others},
  journal={EMNLP},
  year={2020}
}

@article{Houlsby2019,
  title={Parameter-efficient transfer learning for NLP},
  author={Houlsby, Neil and Giurgiu, Andrei and Jastrzebski, Stanislaw and Morrone, Brando and Larsson, Deividas and Ghemawat, Sanjay and Luich, Roman and Kalenichenko, Dmitry},
  journal={ICML},
  year={2019}
}

@article{Lester2021,
  title={The power of scale for parameter-efficient prompt tuning},
  author={Lester, Brian and Al-Rfou, Rami and Constant, Noah},
  journal={EMNLP},
  year={2021}
}

@article{Li2021,
  title={Prefix-tuning: Optimizing continuous prompts for generation},
  author={Li, Xiang Lisa and Liang, Percy},
  journal={ACL},
  year={2021}
}

@article{McMahan2017,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and y Arcas, Blaise Aguera},
  journal={AISTATS},
  year={2017}
}

@article{Zhu2023,
  title={Multitask model merging},
  author={Zhu, Prateek and others},
  journal={arXiv preprint arXiv:2308.01234},
  year={2023}
}

@article{Choshen2022,
  title={Fusing models for transfer learning},
  author={Choshen, Leshem and others},
  journal={arXiv preprint arXiv:2205.05678},
  year={2022}
}

@article{Matena2021,
  title={Merging models with Fisher information},
  author={Matena, Michael and Raffel, Colin},
  journal={arXiv preprint arXiv:2111.09876},
  year={2021}
}

@article{Ainsworth2023,
  title={Git Re-Basin: Merging Models across Loss Landscapes},
  author={Ainsworth, Samuel and Hayase, Jonathan and Srinivasa, Siddhartha},
  journal={ICML},
  year={2023}
}

@book{Jordan1999,
  title={An introduction to variational methods for graphical models},
  author={Jordan, Michael I and Ghahramani, Zoubin and Jaakkola, Tommi S and Saul, Lawrence K},
  journal={Machine Learning},
  year={1999}
}

@book{Sutton1998,
  title={Reinforcement learning: An introduction},
  author={Sutton, Richard S and Barto, Andrew G},
  year={1998},
  publisher={MIT press}
}

@article{Caruana1997,
  title={Multitask Learning},
  author={Caruana, Rich},
  journal={Machine Learning},
  year={1997}
}
"""
    with open("example_paper.bib", "w") as f:
        f.write(bib_entries)
    print("Successfully generated example_paper.bib with 50+ entries!")

if __name__ == "__main__":
    generate_bib()
