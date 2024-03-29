## Stroke-cCycleGAN for realistic offline signature generation.

This repository presents the inference code of Stroke-cCycleGAN [1] for stroke-informed online-to-offline signature conversion. The Stroke-cCycleGAN was inspired from CycleGAN [2,4]. Before using this repository to generate realistic handwritten signatures, you should first download the DeepSignDB online signature database [3], and render the data into skeleton images using OpenCV library. 

### Entrypoint

test.py

### Environment Setup

See requirements.txt.

### References

[1] Jiang J, Lai S, Jin L, et al. Forgery-free signature verification with stroke-aware cycle-consistent generative adversarial network[J]. Neurocomputing, 2022, 507: 345-357.  

[2] Zhu J Y, Park T, Isola P, et al. Unpaired image-to-image translation using cycle-consistent adversarial networks[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2223-2232.

[3] Tolosana R, Vera-Rodriguez R, Fierrez J, et al. DeepSign: Deep on-line signature verification[J]. IEEE Transactions on Biometrics, Behavior, and Identity Science, 2021, 3(2): 229-239.

[4] https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.


### Citation and Contacts

Please consider to cite our paper if you find the code useful.

```
@article{jiang2022forgery,
  title={Forgery-free signature verification with stroke-aware cycle-consistent generative adversarial network},
  author={Jiang, Jiajia and Lai, Songxuan and Jin, Lianwen and Zhu, Yecheng and Zhang, Jiaxin and Chen, Bangdong},
  journal={Neurocomputing},
  volume={507},
  pages={345--357},
  year={2022},
  publisher={Elsevier}
}
```

For any questions about the codes, please contact the authors by sending emails to Prof. Jin (eelwjin@scut.edu.cn) or Jiajia Jiang (jiajiajiang123@qq.com).

### Copyright

This code is free to the academic community for research purpose only. For commercial purpose usage, please contact Dr. Lianwen Jin: eelwjin@scut.edu.cn.


