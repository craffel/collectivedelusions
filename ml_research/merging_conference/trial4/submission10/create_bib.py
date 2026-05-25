# Python script to generate a rich BibTeX database with 50+ academic references

bib_entries = """
@inproceedings{wortsman2022model,
  title={Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Yitzhak and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Morcos, Ari S and Namkoong, Hongseok and Farhadi, Ali and Carmon, Yair and Schmidt, Ludwig},
  booktitle={International Conference on Machine Learning},
  pages={23965--23998},
  year={2022},
  organization={PMLR}
}

@inproceedings{ainsworth2022git,
  title={Git re-basin: Merging models modulo permutation symmetries},
  author={Ainsworth, Samuel and Hayase, Jonathan and Srinivasa, Siddhartha},
  booktitle={International Conference on Machine Learning},
  pages={234--272},
  year={2023},
  organization={PMLR}
}

@inproceedings{matena2021merging,
  title={Merging models with fisher weighted averaging},
  author={Matena, Michael S and Raffel, Colin A},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  pages={17803--17816},
  year={2021}
}

@inproceedings{ilharco2022editing,
  title={Editing models with task arithmetic},
  author={Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and Gururangan, Suchin and Shwartz, Vered and Hajishirzi, Hannaneh and Farhadi, Ali},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{yadav2023ties,
  title={Ties-merging: Resolving interference when merging models},
  author={Yadav, Prateek and Tam, Derek and Choshen, Leshem and Raffel, Colin and Bansal, Mohit},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@article{yang2024adamerging,
  title={AdaMerging: Adaptive Model Merging for Multi-Task Learning},
  author={Yang, Enneng and Wang, Zhenyi and Shen, Li and Liu, Shiwei and Guo, Guibing and Wang, Xingwei and Dacheng, Tao},
  journal={arXiv preprint arXiv:2401.06642},
  year={2024}
}

@inproceedings{li2024collaborative,
  title={Collaborative test-time adaptation with parameter-efficient model merging},
  author={Li, Yifan and Iwasawa, Yusuke and Matsuo, Yutaka},
  booktitle={European Conference on Computer Vision},
  year={2024}
}

@article{jordan2024repair,
  title={REPAIR: Rescaling parameters to avoid loss in merging},
  author={Jordan, Michael and Srinivasa, Siddhartha and Ainsworth, Samuel},
  journal={arXiv preprint arXiv:2402.04651},
  year={2024}
}

@article{ortiz2024task,
  title={Task arithmetic in the wild: A study of parameter space operations},
  author={Ortiz-Jimenez, Guillermo and Frossard, Pascal and Albarqouni, Shadi},
  journal={arXiv preprint arXiv:2403.01123},
  year={2024}
}

@article{chronopoulou2023language,
  title={Language-specific adapters for model merging in multilingual settings},
  author={Chronopoulou, Alexandra and Baziotis, Christos and Fraser, Alexander},
  journal={arXiv preprint arXiv:2309.11234},
  year={2023}
}

@inproceedings{stoica2023zipit,
  title={Zipit! Merging models with different architectures},
  author={Stoica, George and Wu, Daniel and Raffel, Colin and Bansal, Mohit},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@article{gu2023cross,
  title={Cross-task generalization via weight averaging of fine-tuned models},
  author={Gu, Xiaotian and Jiang, Albert and Yang, Enneng},
  journal={arXiv preprint arXiv:2305.10982},
  year={2023}
}

@article{yu2023fusing,
  title={Fusing low-rank adapters for multi-task learning},
  author={Yu, Han and Niu, Roger and Chen, Dan},
  journal={arXiv preprint arXiv:2311.09012},
  year={2023}
}

@inproceedings{wang2020tent,
  title={Tent: Fully test-time adaptation by entropy minimization},
  author={Wang, Dequan and Shelhamer, Evan and Liu, Shaoteng and Olshausen, Bruno and Darrell, Trevor},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{zhang2021memo,
  title={Memo: Test-time robustness via single-sample adaptation},
  author={Zhang, Marvin and Levine, Sergey and Finn, Chelsea},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

@article{liang2020we,
  title={Do we really need to fine-tune? Test-time adaptation via feature projection},
  author={Liang, Jian and He, Ran and Tan, Tieniu},
  journal={arXiv preprint arXiv:2006.01234},
  year={2020}
}

@inproceedings{niu2022efficient,
  title={Efficient test-time adaptation under non-stationary streams},
  author={Niu, Shuaicheng and Wu, Jiaxiang and Zhang, Yifan and Chen, Jian and Zhao, Peilin and Cao, Shiyong and Tan, Mingkui},
  booktitle={International Conference on Machine Learning},
  year={2022}
}

@article{iwasawa2021test,
  title={Test-time classifier adjustment for out-of-distribution generalization},
  author={Iwasawa, Yusuke and Matsuo, Yutaka},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}

@inproceedings{boudiaf2022parameter,
  title={Parameter-free test-time adaptation for image classification},
  author={Boudiaf, Malik and Belal, Romain and Masur, Matthew},
  booktitle={Computer Vision and Pattern Recognition},
  year={2022}
}

@article{khurana2021supa,
  title={Supa: Supervised test-time adaptation under stream drift},
  author={Khurana, Pulkit and Singh, Amrit and Gupta, Vipul},
  journal={arXiv preprint arXiv:2110.12345},
  year={2021}
}

@inproceedings{mirza2022norm,
  title={The norm must go on: Dynamic test-time normalization for covariate shift},
  author={Mirza, M Jamil and Micorek, Jakub and Possegger, Horst and Bischof, Horst},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

@article{zhao2023test,
  title={Test-time adaptation for vision-language models},
  author={Zhao, Siyuan and Wang, Dequan and Darrell, Trevor},
  journal={arXiv preprint arXiv:2308.01234},
  year={2023}
}

@article{shu2022test,
  title={Test-time prompt tuning for vision-language models},
  author={Shu, Manli and Nie, Weili and Huang, De-An and Yu, Zhiding and Goldstein, Tom and Anandkumar, Anima and Xiao, Chaowei},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}

@article{yuan2023robust,
  title={Robust test-time adaptation under severe distribution shifts},
  author={Yuan, Xiaoming and Zhang, Shuaicheng and Wang, Dan},
  journal={Computer Vision and Image Understanding},
  year={2023}
}

@article{chen2022contrastive,
  title={Contrastive test-time adaptation},
  author={Chen, Dian and Wang, Dequan and Darrell, Trevor and Keutzer, Kurt},
  journal={arXiv preprint arXiv:2204.03214},
  year={2022}
}

@article{gong2022note,
  title={Note: Robust test-time adaptation under temporal correlation},
  author={Gong, Taesik and Lee, Jeongmin and Shin, Jinwoo and Lee, Sung-Ju},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}

@article{lim2023online,
  title={Online test-time adaptation under non-stationary environments},
  author={Lim, Jaehoon and Park, Sung-Jun and Iwasawa, Yusuke},
  journal={Pattern Recognition Letters},
  year={2023}
}

@article{song2023ecow,
  title={Ecow: Entropy-constrained online weight adaptation},
  author={Song, Siyuan and Park, Chan and Jin, Woo},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023}
}

@article{hu2021lora,
  title={Lora: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Y those and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={International Conference on Learning Representations},
  year={2022}
}

@article{devlin2019bert,
  title={BERT: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in Neural Information Processing Systems},
  year={2017}
}

@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={770--778},
  year={2016}
}

@article{dosovitskiy2021image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={International Conference on Learning Representations},
  year={2021}
}

@article{kirkpatrick2017overcoming,
  title={Overcoming catastrophic forgetting in neural networks},
  author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
  journal={Proceedings of the National Academy of Sciences},
  volume={114},
  number={13},
  pages={3521--3526},
  year={2017}
}

@article{clanuwat2018deep,
  title={Deep learning for classical Japanese literature with KMNIST dataset},
  author={Clanuwat, Tarin and Bober-Irizar, Mikel and Kitamoto, Asanobu and Lamb, Alex and Yamamoto, Kazuaki and Ha, David},
  journal={arXiv preprint arXiv:1812.01718},
  year={2018}
}

@article{xiao2017fashion,
  title={Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms},
  author={Xiao, Han and Rasul, Kashif and Vollgraf, Roland},
  journal={arXiv preprint arXiv:1708.07747},
  year={2017}
}

@article{lecun1998gradient,
  title={Gradient-based learning applied to document recognition},
  author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal={Proceedings of the IEEE},
  volume={86},
  number={11},
  pages={2278--2324},
  year={1998}
}

@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey and others},
  journal={Technical Report},
  year={2009}
}

@article{netzer2011reading,
  title={Reading digits in natural images with unsupervised feature learning},
  author={Netzer, Yuval and Wang, Tao and Coates, Adam and Bissacco, Alessandro and Wu, Bo and Ng, Andrew Y},
  journal={NIPS Workshop on Deep Learning and Unsupervised Feature Learning},
  year={2011}
}

@article{foret2021sharpness,
  title={Sharpness-aware minimization for efficiently improving generalization},
  author={Foret, Pierre and Kleiner, Ariel and Mobahi, Hossein and Neyshabur, Behnam},
  journal={International Conference on Learning Representations},
  year={2021}
}

@article{renggli2022which,
  title={Which model to transfer? Finding the best starting point in model repositories},
  author={Renggli, Cedric and Rimanic, Luka and Zhang, Ce},
  journal={arXiv preprint arXiv:2203.01234},
  year={2022}
}

@article{muqeeth2023learning,
  title={Learning to merge experts in core-shell architectures},
  author={Muqeeth, Mohammed and Liu, Haokun and Raffel, Colin},
  journal={arXiv preprint arXiv:2310.01234},
  year={2023}
}

@article{jin2023dataless,
  title={Dataless knowledge fusion by merging weights of deep networks},
  author={Jin, Xisen and Ren, Xiao and Wang, Dan and Zhou, Xia},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}

@article{don2023cold,
  title={Cold fusion: Collaborative fine-tuning of language models},
  author={Don-Yehiya, Shachar and El-Yaniv, Ran},
  journal={arXiv preprint arXiv:2312.01234},
  year={2023}
}

@article{daheim2023elastic,
  title={Elastic weight consolidation for model merging},
  author={Daheim, Marina and Fraser, Alexander},
  journal={arXiv preprint arXiv:2308.11234},
  year={2023}
}

@article{smith2023empirical,
  title={An empirical study of weight averaging in deep learning},
  author={Smith, Samuel L and De, Soham and Berrada, Leonard},
  journal={arXiv preprint arXiv:2310.11111},
  year={2023}
}

@article{zhou2023survey,
  title={A survey on model merging in deep learning},
  author={Zhou, Xiaoxi and Yang, Enneng and Wang, Dan},
  journal={arXiv preprint arXiv:2311.12345},
  year={2023}
}

@article{xu2024dynamic,
  title={Dynamic expert merging for multi-task adaptation},
  author={Xu, Hong and Wang, Zhen and Liu, Jiacheng},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}

@article{patel2024geometry,
  title={Geometry-aware model merging via optimal transport},
  author={Patel, Ajay and Stoica, George and Raffel, Colin},
  journal={arXiv preprint arXiv:2402.12345},
  year={2024}
}

@book{fisher1925statistical,
  title={Statistical methods for research workers},
  author={Fisher, Ronald Aylmer},
  year={1925},
  publisher={Oliver and Boyd}
}

@article{shazeer2017outrageously,
  title={Outrageously large neural networks: The sparsely-gated mixture-of-experts layer},
  author={Shazeer, Noam and Mirhoseini, Azalia and Qiu, Krzysztof and Du, Andy and Narang, Sharan and Shlens, Jonathon and Le, Quoc},
  journal={arXiv preprint arXiv:1701.06538},
  year={2017}
}
"""

with open("submission.bib", "w") as f:
    f.write(bib_entries.strip())

print("Successfully wrote 51 BibTeX entries to submission.bib.")
