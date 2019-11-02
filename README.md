# Robot-progress
Repository to track the progress in robot grasping, including the datasets and the current state-of-the-art.

# Generalising to Novel Objects


## Detecting Grasp poses

### 2008 - [Robotic Grasping of Novel Objects using Vision](http://pr.cs.cornell.edu/grasping/IJRR_saxena_etal_roboticgraspingofnovelobjects.pdf)
**Summary**: Given two or more images, this algorithm tries to find a few points which indicate good grasping locations. These points are then triangulated to compute a 3D grasping position. It is a super-vised learning method, trained on synthetic data. Effectively grasps wide range of (unseen) objects. 

### 2013 - [Deep Learning for Detecting Robotic Grasps](https://arxiv.org/abs/1301.3592)
**Summary**: Introduces 5-dimensional grasp representation. Presents two-step cascaded system. First network has fewer features and can effectively prune unlikely grasps. Second network only handles those few good grasps. The input is a single RGB-D image. A small network is used to evaluate potential grasps. The best grasps are inputs for the second larger network, that outputs the best grasp. This is then converted to a robot grasp that includes a grasping point and an approach vector. It uses the rectangle's parameters and the surface normal at the rectangle's center to compute this. The network is trained on the Cornell Dataset, which is particulary set up for parellel gripper robots.

### 2015 - [Real-Time Grasp Detection Using Convolutional Neural Networks](https://arxiv.org/abs/1412.3128)
**Summary**: Presents single-stage regression to grasp bounding boxes, not using sliding-window methods. Runs in 13fps on GPU. Can also predict multiple grasps; works better, especially with objects that can be grasped in multiple ways. Also uses 5D representation. Standard ConvNetwork that outputs 6 neurons, trained on Cornell Dataset, pretrained on ImageNet. Best so far: 88 procent accuracy. 

### 2016 - [Robotic Grasp Detection using Deep Convolutional Neural Networks](https://arxiv.org/abs/1611.08036)
**Summary**: Implements ResNet50. Cornell Dataset; pretrained on ImageNet; 5D pose. Best so far: 89.1 procent accuracy. Does not test with real robot.

### 2018 - [Fully Convolutional Grasp Detection Network with Oriented Anchor Box](https://arxiv.org/abs/1803.02209)
**Summary**: Predicts multiple-grasp poses. Network has two parts: feature extractor (DNN) & multi-grap predictor (regresses grasp rectangles from oriented anchor boxes; classifies these to graspable or not). Cornell Dataset. Best so far: 97.74 procent accuracy.  Does not test with real robot.

**Future work**: Detect grasp locations for all objects in an image. Handle overlapping objects. 

### 2019 - [Real-Time, Highly Accurate Robotic Grasp Detection using Fully Convolutional Neural Network with Rotation Ensemble Module](https://arxiv.org/abs/1812.07762)
**Summary**: Proposes a rotation ensemble module (REM): convolutions that rotates network weights. 5D poses; Cornell dataset: 99.2 procent accuracy. Test on real (4-axis) robot: 93.8 succes rate (on 8 small objects).

### 2019 - [Multimodal grasp data set: A novel visual–tactile data set for robotic manipulation](https://journals.sagepub.com/doi/10.1177/1729881418821571)


## Surveys 

### 2016 - [Data-Driven Grasp Synthesis - A Survey](https://arxiv.org/pdf/1309.2660.pdf)

### 2019 - [Vision-based Robotic Grasping from Object Localization, Pose Estimation, Grasp Detection to Motion Planning: A Review](https://arxiv.org/abs/1905.06658)
**Summary**: Talks about object localization; object segmenation; 6D-pose estimation; grasp detection; end2end; motion planning; datasets. 


## Deep Reinforcement Learning

#### 2018 - [Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning](https://arxiv.org/abs/1803.09956)

"Each individual FCN φψ takes as input the heightmap image representation of the state st and outputs a dense pixel-wise map of Q values with the same image size and resolution as that of st, where each individual Q value prediction at a pixel p represents the future expected reward of executing primitive ψ at 3D location q where q 􏰏 p ∈ st. Note that this formulation is a direct amalgamation of Q-learning with visual affordance-based manipulation."


