#  Learning Shape Abstractions by Assembling Volumetric Primitives
Shubham Tulsiani, Hao Su, Leonidas J. Guibas, Alexei A. Efros, Jitendra Malik. In CVPR, 2017.
[Project Page](https://shubhtuls.github.io/volumetricPrimitives/)

![Teaser Image](https://shubhtuls.github.io/volumetricPrimitives/resources/images/teaser.png)

### 1) Demo
Please check out the [interactive notebook](demo/demo.ipynb) which shows how to compute the primitive based representation for an input shape. You'll need to - 
- Install a working implementation of torch and itorch.
- Edit the path to the blender executable in the demo script.

### 2) Training
We provide code to train the abstraction models on ShapeNet categories.

#### a) Preprocessing
We'll first need to preprocess the ShapeNet models to compute voxelizations required as inputs as well as data required to implement the loss functions.
- Install [gptoolbox](https://github.com/alecjacobson/gptoolbox) in external/gptoolbox. You'll need to compile the mex file for ```point_mesh_sqaured_distance```. You can first try [this precompiled version](https://people.eecs.berkeley.edu/~shubhtuls/cachedir/primitives/point_mesh_squared_distance.mexa64). If that does not work, you will have to compile it yourself - some helpful steps as required on my machine are pointed out [here](external/gptInstall.sh).
- Modify the path to ShapeNet dataset (v1) in the [startup file](preprocess/shapenet/startup.m)
- Specify the synsets of interest in the [preprocessing script](preprocess/shapenet/precomputeShapeData.m) and then run it.

#### b) Learning
The training takes place in two stages. In the first we use all cuboids while biasing them to be small and then allow the network to choose to use fewer cuboids. Sample scripts for the synset corresponding to chairs are below.
```
# Stage 1
cd experiments;
disp=0 gpu=1 nParts=20 nullReward=0 probLrDecay=0.0001 shapeLrDecay=0.01 synset=3001627 usePretrain=0 numTrainIter=20000 name=chairChamferSurf_null_small_init_prob0pt0001_shape0pt01 th cadAutoEncCuboids/primSelTsdfChamfer.lua
```

After the first network is trained, we allow the learning of primitive existence probabilities.
```
# Stage 2
cd experiments;
pretrainNet=chairChamferSurf_null_small_init_prob0pt0001_shape0pt01 pretrainIter=20000 disp=0 gpu=1 nParts=20 nullReward=8e-5 shapeLrDecay=0.5   synset=3001627 probLrDecay=0.2 usePretrain=1  numTrainIter=30000 name=chairChamferSurf_null_small_ft_prob0pt2_shape0pt5_null8em5 th cadAutoEncCuboids/primSelTsdfChamfer.lua
```
### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{abstractionTulsiani17,
  title={Learning Shape Abstractions by Assembling Volumetric Primitives},
  author = {Shubham Tulsiani
  and Hao Su
  and Leonidas J. Guibas
  and Alexei A. Efros
  and Jitendra Malik},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2017}
}
```
