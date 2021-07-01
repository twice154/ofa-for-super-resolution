# Image Downscaling & Super-Resolution based on Once-for-All

This repository contains image downscaling & super-resolution project code based on the paper ["Once-for-All: Train One Network and Specialize it for Efficient Deployment"](https://arxiv.org/abs/1908.09791) (ICLR 2020).

I tried to implement (the paper don't mention explicitly.)
* Neural Network Architecture
* Training Details
* Data Augmentation Strategy

## License and Citation

```BibTex
@inproceedings{
  cai2020once,
  title={Once for All: Train One Network and Specialize it for Efficient Deployment},
  author={Han Cai and Chuang Gan and Tianzhe Wang and Zhekai Zhang and Song Han},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://arxiv.org/pdf/1908.09791.pdf}
}
```

# Data Preparation

Follow the instruction of [Official xR-EgoPose Dataset Repository](https://github.com/facebookresearch/xR-EgoPose) for downloading dataset.  
After downloading, open **data/config.yml**, and write your own dataset path.

Here is a visualization of LeftArm's distribution in 3D space (Just watch trianglular points.).
![img](LeftArm.png)  
You can check LeftArm.png, LeftElbow.png, LeftHand.png, RightFoot.png, RightKnee.png, RightLeg.png for more joint visualizations.

# Results of implemented Model in terms of Full-body Average Evaluation Error (mm).

Tricks I tried.
* Pre-training & Fine-tuning
* Add BN in Lifting Module ([Reference](https://arxiv.org/abs/1705.03098))
* Add Dropout
* Change Root Joint from Neck to Hip ([Reference1](https://arxiv.org/abs/2008.09047), [Reference2](https://arxiv.org/abs/2008.03713))
* Improve Lifting Module Architecture
* Data Augmentation


</ul>
<table>
<thead>
<tr>
<th align="center">Dataset</th>
<th align="center">Naive Architecture & Training</th>
<th align="center">Pre-training & Fine-tuning</th>
<th align="center">BN in Lifting Module</th>
<th align="center">Dropout</th>
<th align="center">Change Root Joint</th>
<th align="center">Improve Lifting Module</th>
<th align="center">Data Augmentation</th>
<th align="center">ICCV19</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">xR-EgoPose</td>
<td align="center">280</td>
<td align="center">230</td>
<td align="center">110</td>
<td align="center">110</td>
<td align="center">110</td>
<td align="center">100</td>
<td align="center">100</td>
<td align="center">58</td>
</tr>
</tbody></table>

# Training

Open **data/config.yml**, and write your own settings.

Run
```
python train.py
```
