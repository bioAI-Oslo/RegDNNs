# Regularization of Neural Networks
**Project**: Exploring soft and hard constraints in artificial neural networks  
**Goal**: Understand how different regularization techniques promotes better generalization and robustness in DNNs  
  
This project developed over time, and has three parts. Part one was a collaboration between me and [@Dalsaetra](https://www.github.com/Dalsaetra). All parts were done under supervision by [@Vemundss](https://www.github.com/Vemundss).
  
  
## Content
1. [Setup: Packages and Software](#setup-packages-and-software)  
2. [Part One: Initial Testing, Learning and Toy-Problem](#part-one-initial-testing-learning-and-toy-problem)  
3. [Part Two: Broad Regularization Benchmarking](#part-two-broad-regularization-benchmarking)  
    1. [Overview of Regularization Techniques](#overview-of-regularization-techniques)  
    2. [Overview of Visualization Techniques](#overview-of-visualization-techniques)  
4. [Part Three: Exploring Jacobian and SVB Regularization](#part-three-exploring-jacobian-and-svb-regularization)  
    1. [Takeaways and Further Work](#takeaways-and-further-work)   
   

## Setup: Packages and Software
I use two conda environments for the project. One for running the files under the folder *JAX* in part one of the project, and another for running all other files. This was due to some packages breaking when combined with the jax packages. For both conda environments I installed jupyter via:
```
conda install jupyter
```
Afterwards I installed the packages in the requirement files using pip. The requirements for jax is in the *JAX* folder, while the requirements for everything else is in the main project. For the main project I also installed **ffmpeg** with:
```
conda install -c conda-forge ffmpeg
```
Which I used to make animations during my initial exploration of a toy problem in part one. I also used ChatGPT to provide most of the code documentation, as well as to help in writing many parts of the code.
  
  
## Part One: Initial Testing, Learning and Toy-Problem
The folder *Initial Testing, Learning and Toy-Problem*  contains *JAX*, *Start training*, *UMI* and *UMI RNN*. Here I learned how to build/train NNs and RNNs using pytorch, and started learning how to use jax.  

I tested regularization on a simple toy problem (called UMI). The testing consisted of the following:  
1. Generating four clusters of points with some Gaussian noise  
2. Training networks to classify points to the correct class (cluster)  
3. Learning how to implement different regularization techniques, and test them on the problem  

**Motivation**: The simple toy problem easily shows how regularization makes networks generalize better, which makes it good for learning how to implement neural networks with regularization in pytorch. For Jacobian regularization it is interesting to look at how it preserves areas between I/O, which we get a good intuition for with the toy problem as we can visualize it easily.   
   
**Result**: Tested without regularization, and with L1, L2, SVB and different Jacobian regularizations. Results are stored in the folder *Animations* under *UMI*.  
   
The code in this part is quite unstructured and contains multiple errors as it was made for learning/testing.
  

## Part Two: Regularization Benchmarking
The goal of this part of the project was to investigate deep convolutional neural networks trained with different regularization techniques to build intuition for how the techniques work and how they influence the network. To do this I implemented many different regularization techniques and visualization techniques for DNNs. I tested the implementations on three datasets: MNIST, CIFAR10 and CIFAR100, using the LeNet model (See Lecun et al, 1998) and the DDNet model (See Hoffman 2019). For this part I also learned how to use ssh to connect to and use a data cluster, and how to use parallell computing to train on multiple GPUs using pytorch. See notebooks MNIST_LeNet, CIFAR10_DDNet and CIFAR100_DDNet for results. Following is a list of the regularization and visualization techniques I implemented, with a short description explaining how they work. 
  
### Overview of Regularization Techniques
  
#### Comparison - No Regularization
Without any regularization, the model simply minimizes the loss function on the training data. This can lead to overfitting, especially in models with many parameters that can fit a lot of noise. As a result, it often performs poorly on unseen data. I train models without regularization to use as comparison for the different regularization techniques.
  
#### L1 Regularization
L1 regularization, also known as Lasso regularization, involves adding a term to the loss function that penalizes the absolute value of the weights. This encourages the model to have sparse weights, meaning that many weights are zero. This can lead to a model that is easier to interpret, because it effectively performs feature selection, choosing a subset of the input features to focus on.  
  
#### L2 Regularization
L2 regularization, also known as Ridge regularization, involves adding a term to the loss function that penalizes the square of the weights. This encourages the model to have small weights but does not encourage sparsity. L2 regularization can help prevent overfitting by discouraging the model from relying too much on any single input feature.  
  
#### Elastic Net Regularization
Elastic Net regularization is a compromise between L1 and L2 regularization. It involves adding terms to the loss function with both a L1 penalty and an L2 penalty. This allows the model to have some level of sparsity, like L1 regularization, while also encouraging small weights, like L2 regularization.  
  
#### Soft SVB Regularization
Soft SVB regularization, introduced by Jia et al. 2019, penalizes the model based on the Frobenius norm of the difference between the weights' Gram matrix and the identity matrix. This encourages the model's weights to be orthogonal, which can improve generalization. Soft SVB regularization might introduce additional computational cost due to the need to compute matrix multiplications and norms.  
  
#### Hard SVB Regularization
Hard SVB regularization, similar to Soft SVB, also encourages the model's weights to be orthogonal, but it does so in a more strict manner. It uses a hard constraint instead of a soft penalty, meaning that the model's weights are forced to be orthogonal.  
  
#### Jacobian Regularization
Jacobian regularization introduces a penalty on the norm of the Jacobian matrix of the model's outputs with respect to its inputs. The Jacobian matrix represents the first-order derivatives of the model. By penalizing the norm of the Jacobian, the model is encouraged to have outputs that change linearly or sub-linearly with respect to small changes in the inputs. It can help in achieving more stable models with smoother decision boundaries. However, calculating the Jacobian matrix can be computationally expensive for complex models and large inputs. I solve this issue by approximating it, following an algorithm laid out in Hoffman 2019.  
  
#### Jacobi Determinant Regularization
Jacobi Determinant regularization involves adding a term to the loss function that penalizes the squared difference between the determinant of the Jacobian of the model's outputs with respect to its inputs and one. This regularization approach encourages the model to maintain volume preservation in the input space to the output space. The computational cost of this method is high.  
  
#### Dropout Regularization
Dropout works during training, by randomly setting a fraction of input units to 0 at each update, which helps to prevent overfitting. This introduces noise into the training process that forces the learning algorithm to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.  
  
#### Confidence Penalty Regularization
Confidence penalty regularization adds a penalty to the loss function based on the confidence of the model's predictions. If the model is too confident, it is penalized more heavily. This can encourage the model to be more cautious in its predictions, potentially leading to better calibration and more reliable uncertainty estimates. However, this approach may be inappropriate for certain tasks where high confidence is desirable, and it can also make the optimization problem more challenging.   
  
#### Label Smoothing Regularization
Label Smoothing is a form of regularization where the target labels for a classification problem are replaced with smoothed versions. Instead of having a hard one-hot encoded target, each target will have a small value for each incorrect class and the rest of the value for the correct class. This encourages the model to be less confident, reducing the risk of overfitting and improving generalization.  
  
#### Noise Injection to Inputs
Noise injection to inputs is a regularization technique where random noise is added to the inputs during training. This encourages the model to be robust to small changes in the inputs. This form of regularization can help prevent overfitting by discouraging the model from fitting the noise in the training data, instead focusing on the underlying patterns that are consistent even when noise is added.  
  
#### Noise Injection to Weights
Similar to noise injection to inputs, noise injection to weights involves adding random noise to the weights during training. This can be seen as a form of stochastic regularization, as it adds a source of randomness that the model needs to be robust to. By preventing the weights from settling into a fixed value, it encourages the model to explore different parts of the weight space, which can help it avoid overfitting to the training data.   
  
  
### Overview of Visualization Techniques
  
#### Training and Test Loss Curves
The most straightforward way to visualize the effect of regularization is by plotting the training and test loss over time. The further the gap between the two losses, the more the model might be overfitting the training data. Thus, we can see the effect of regularization on reducing the gap, and also on overall performance.  
  
#### Weight Distributions
For L1, L2, and Elastic Net regularization it is useful to visualize the distribution of the weights in the model. L1 regularization should result in many weights being exactly zero, while L2 regularization will typically result in a distribution with smaller magnitudes.  
  
#### Feature Map Visualizations
Especially in the context of convolutional neural networks (CNNs), visualizing the feature maps - the activations of the convolutional layers - can provide insight into what features the network is learning. This can give a sense of how regularization is affecting the types of features learned. For instance, too much L1/L2 regularization might result in overly simplistic feature maps, while too little might result in feature maps that are overly complex or noisy.  
  
#### Uncertainty Estimates
For regularization methods that affect the model's confidence, like Confidence Penalty and Label Smoothing, one can plot the model's predicted scores. A well-regularized model should show less overconfidence and better-calibrated scores.  
  
#### T-SNE or PCA of Activations
Dimensionality reduction techniques like t-SNE or PCA can be used to visualize the activations of the network. This involves taking the activation values of a particular layer and reducing them to 2 or 3 dimensions for plotting. Different classes should ideally form distinct clusters, and overfitting may manifest as overly complex boundaries between classes.  
  
#### Saliency Maps
A saliency map tries to show which parts of the image contribute most significantly to a neural network's decision. It is created by calculating the gradient of the output category with respect to the input image. This gradient is then visualized as a heatmap overlaying the original image, with high-gradient regions indicating important areas for the model's decision. The intuition behind this is that the gradient measures how much a small change in each pixel's intensity would affect the final prediction. So, large gradient values suggest important pixels.  
  
#### Occlusion Sensitivity 
Occlusion sensitivity is a method that involves systematically occluding different parts of the input image with a grey square (or other "occluder"), and monitoring the effect on the classifier's output. The output is then visualized as a heatmap showing how much the classifier's confidence decreased when each region was occluded, highlighting important regions in the input image for the model's decision.  
  
  
## Part Three: Exploring Jacobian and SVB Regularization
After doing a lot of testing, I was ready to face the problem of generalization and robustness in models head on. For this part I train LeNet models on MNIST, DDNet models on CIFAR10 and ResNet18 models on CIFAR100. For MNIST and CIFAR10 I train models with dropout regularization and no, l2, jacobian and svb regularization. I also train models without any regularization, and with jacobian regularization without dropout. For the ResNet18 models I use ResNet18 from torchvision which comes without dropout, so I only train four models: no, l2, jacobian and svb regularization, all without dropout.   
   
I started by reproducing most of Hoffman 2019, and using the Singular Value Bounding regularization scheme from Jia 2019. I trained models for MNIST and CIFAR10 as described in Hoffman 2019. I also implemented decision boundary visualization, and introduced isotropic and anisotropic total variation as a measure of ruggedness in plots of decision boundaries. Lastly, I attack the models using FGSM and PGD attacks. The results and plots can be found in the results and visualization notebooks for each dataset.   
   
The following is a more detailed description of the problem, and techniques used in this part in terms of implementation and what they show, as well as motivation for why I used them.   
   
### Problem Description   
The problem my project has revolved around and that this part tries to make progress on is two-folded, but connected: there is a problem of generalization, and a problem of robustness. The problem of generalization consists in the fact that DNNs often perform a lot better on the training data than on test data. Models are often not good at distributional shifts of input data, making them less useful on a range of tasks for which one cannot be completely certain that the model will always be used on data similar to those it was trained on. The problem of robustness is closely related: it is the fact that many DNNs are prone to bad performance on input data with small perturbations. Models are fragile (not robust) when very small changes to data make their predictions break down. Fragile models also generalize worse, and do worse under distributional shifts. Thus, the problems of generalizations and robustness are connected, and we want to be able to consistently train robust models that generalize well to new data.  
     
### Jacobian Regularization   
Jacobian regularization aims to make models more robust and better at generalizing by introducing a penalty term on the norm of the jacobian matrix of the model's outputs with respect to its inputs in the loss function. The jacobian matrix represents the first-order derivatives of the model. By penalizing the norm of the jacobian, the model is encouraged to have outputs that change linearly or sub-linearly with respect to small changes in the inputs. This can help in achieving more stable models with larger and smoother decision boundaries - showing that the model is more robust to small perturbations. However, calculating the jacobian matrix can be computationally expensive for complex models and large inputs. I solve this issue by approximating it, following an algorithm laid out in Hoffman 2019.  
    
### Singular Value Bounding Regularization    
Singular value bounding regularization was developed based on an upper generalization error bound for nonlinear DNNs, which Jia et al showed can be achieved through equal singular values of weight matrices. This can be most easily achieved through orthogonal deep neural networks (all singular values are 1). Since this is expensive, Jia et al developed an algorithm to achieve approximate Orthogonal Deep Neural Networks. To do this, one has to first initialize the weights orthogonally. Then, every N SGD iterations one performs singular value decomposition on the weight matrices to seperate out the singular values, and then bound them in a small band around 1.   
    
### Visualization of Decision Boundaries   
The file plotting_tools.py contains functionality for plotting decision boundaries. These boundaries take a random subsample of the 784-dimensional input space (for MNIST) and plots a 2D plane of decision boundaries. Because of this it is somewhat limited, and will sometimes show the wrong prediction even though the model actually makes the right prediction. It is useful to study the decision boundaries because we can observe how the different regularizers behave. The ruggedness/shape of the decision boundaries tells us something about how the model has been trained and is performing. We expect models that generalize worse and/or that overfits the training data to have more irregular decision cell boundaries (overfitting makes complex decision boundaries to fit single training data). For these models, small perturbations such as those provided by the FGSM attack and other adverserial attacks might be detrimental to model generalization. For models that generalize better and that are regularized in a better way we expect more smooth and larger decision cell boundaries. This would signify robustness, as perturbations in the input data would not easily lead to a change in classification. It would also mean that the model has learnt representations of the high-dimensional image input space that is more clearly separated into regions, and less irregular. Intuitively, models with those properties are needed to solve the problem of instability to perturbations, especially in image classification. In image classification models are constantly faced with many challenges: variations in lighting, scale, viewpoint, background clutter/variation and intra-class variation (for example: some cars look very different from other cars). This makes robustness and generalizability essential features of such models.    
    
### Anisotropic and Isotropic Total Variation   
After implementing decision cell boundaries I realized that it is very subjective to decide which decision boundaries are the best just by looking at them. I wanted to find a numerical way of estimating the roughness/complexity of the decision boundaries, and after searching found that isotropic and/or anisotropic total variation might work well. Total variation is a measure of the "total amount of change" in a function. In image processing, total variation is often used as a regularizer for noise reduction and image reconstruction, as it encourages piecewise constant solutions. Here, we could use it as a measure of the "roughness" of the decision boundaries, as a boundary with lots of twists and turns will have a higher total variation than a smooth one. Isotropic total variation measures the total variation in an image by summing the magnitudes of the gradients (computed using both x and y derivatives) across the image. Anisotropic total variation measures total variation by summing the absolute differences in the x and y directions independently. This gives more weight to horizontal and vertical changes (and so it is anisotropic). These measures are known in the literature for denoising images (see for example Lou and Zeng 2015, A Weighted Difference of Anisotropic and Isotropic Total Variation Model for Image Processing and Isotropic and anisotropic total variation regularization in electrical impedance tomography by Gonzalez et al). For decision boundaries in a classification task, smoothness/ruggedness could be understood as the degree to which the boundary changes direction. This can be quantified using the concept of total variation. I implemented both procedures and applied them to the plots of 2D planes given from the plots of decision boundaries implemented earlier. It is important to notice that the numerical results I get from this are highly reliant on my implementation of getting the image of decision boundaries. It relies especially on the parameter **resolution** in the function. It cannot be used as an absolute measure of how good the different regularization techniques are at reducing the total variance in the images of decision boundaries, but using the same parameters allows for comparison between the models. The results (and test accuracy metrics for the different models) can be found in mnist_lenet_results.ipynb and cifar10_ddnet_results.ipynb.   
   
### Fast Gradient Sign Method (FGSM) Attack   
I have implemented a white-box untargeted fast gradient sign method (FGSM) attack as in Hoffman 2019 (for FGSM, see Goodfellow et al, 2014). White box means that the attacker has full knowledge of the model (including architecture, parameters, training procedure etc.). For FGSM we compute gradients with respect to input data, which requires such white box access. Untargeted means that the attack tries to make the model output any false classification. This is opposed to a targeted attack, which would try to make the model output a specific class. The FGSM is an untargeted attack, as we are maximizing any wrong output.
FGSM perturbs an input image in the direction of the gradient of the loss with respect to the input image. This should make small changes to the image, almost imperceptible to humans, that cause the model to misclassify the image. The strength of the attack depends on the epsilon-parameter controlling the magnitude of the perturbation. In the visualization notebooks I plot how the different models do at classifications for different levels of epsilon. As for the problem of generalization and robustness, more robust models will be better at resisting the FGSM attack, as they are less prone to getting wrong prediction based on small perturbations. It is therefore of interest to observe how the different models do against the FSGM attack, and see if this correlates to smoothness of decision boundaries or to the total variation metrics.   
   
### Projected Gradient Descent (PGD) Attack   
I have also implemented a white-box untargeted Projected Gradient Descent (PGD) attack, as used in Hoffman 2019 and outlined in Madry et al, 2017. PGD is an iterative version of FGSM that applies the same type of perturbation multiple times with small (alpha) step sizes. The input image is shifted along the gradient of the loss function with respect to the input data, but clipped to hold the perturbations within a certain limit (epsilon). This should produce images that look similar to humans, but still leading to a misclassification. According to Hoffman 2019, this is a stronger attack than the non-iterative FGSM. It is stronger because it can search for the optimal perturbation within the epsilon-range, instead of making a single-step one. In the visualization notebook I plot how the models do against the attack for different numbers of iterations with set alpha/epsilon values. It is relevant for the problem of generalization/robustness for the same reasons as FGSM.   
   
   
### Takeaways and Further Work    

The key takeaways from my investigations were:
* Regularization does not effect model generalization (in terms of accuracy on test sets) significantly, but for models implemented in the real world where small differences matter it is important.
* When models are much larger than datasets (in the sense that they can overfit training data), regularization does not effect model robustness much.
* When there is a balance between model and dataset size, regularization markedly increases model robustness, and contributes to models learning more robust representations.
* Based on the two points above, one could say that increasing models size is a type of regularization.
* Jacobian and SVB regularization, and regularization in general, has a positive effect on model learning in terms of building more robust representations, making them more robust to adverserial attacks.  
* Visualizing decision boundaries is useful as an interpretability tool, and the metric of total variation helps in interpreting them.   
 
   
If I were to continue working on the project, I would:   
* Train larger models on larger datasets (for example ImageNet), including learning how to do this  
* Try to control for overfitting (partly by the point above, finding models/datasets where the model cannot overfit everything to solve the problem)  
* Investigate more the correlations between total variation measures and decision boundaries, and between decision boundaries and model performance  
* Implement standard deviation around the graphs in the plots of model performance against adverserial attacks   
* Look into more adverserial attacks, implement them, and test them on my models   
* Look into adverserial training (as in Hoffman 2019)  