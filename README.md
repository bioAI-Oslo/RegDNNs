# Regularization of Neural Networks
**Project**: Exploring soft and hard constraints in artificial neural networks  
**Goal**: Understand how different regularization techniques promotes better generalization in ANNs  
  
This project developed over time, and has three parts. Part one was a collaboration between me and [@Dalsaetra](https://www.github.com/Dalsaetra). All parts were done under supervision by [@Vemundss](https://www.github.com/Vemundss).
  
  
## Content
1. [Setup: Packages and Software](#setup-packages-and-software)  
2. [Part One: Initial Testing, Learning and Toy-Problem](#part-one-initial-testing-learning-and-toy-problem)  
3. [Part Two: Broad Regularization Benchmarking](#part-two-broad-regularization-benchmarking)  
    1. [Overview of Regularization Techniques](#overview-of-regularization-techniques)  
    2. [Overview of Visualization Techniques](#overview-of-visualization-techniques)  
4. [Part Three: Exploring Jacobian and SVB Regularization](#part-three-exploring-jacobian-and-svb-regularization)  
  
  
## Setup: Packages and Software
I used two conda environments for the project. One for running the files under the folder *JAX* in part one of the project, and another for running all other files. This was due to some packages breaking when combined with the jax packages. For both conda environments I installed jupyter via:
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
The goal of this part of the project was to investigate deep convolutional neural networks trained with different regularization techniques and build intuition for how the techniques work and how they influence the network. To do this I implemented many different regularization techniques and visualization techniques for DNNs. I tested the implementations on three datasets: MNIST, CIFAR10 and CIFAR100, using the LeNet model (See Lecun et al, 1998). For this part I also learned how to use ssh to connect to and use a data cluster, and how to use parallell computing to train on multiple GPUs using pytorch. See notebooks for results. Following is a list of the regularization and visualization techniques I implemented, with a short description explaining how they work.  
  

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
For this part I chose to continue with the regularization techniques of the Jacobian Regularization from Hoffman 2019 and the SVB regularization from Jia 2019. I wanted to investigate these techniques further, also using models trained with L2 regularization and no regularization for comparison. I also chose to continue with the following visualization techniques for the investigation: accuracy curves and activation plots through dimensionality reduction (PCA). I also wanted to implement the visualization technique of decision boundaries from Hoffman 2019, and use it on my models to try to understand how they work.

Learned:
To use research rabbit to look for papers and try to get an overview of field.

The folder *Jacobian Regularization* contains functionality for plotting decision boundaries. These boundaries take a random subsample of the 784-dimensional input space and plots a 2D plane of decision boundaries. Because of this it is quite limited, and will often show the wrong prediction even though the model actually makes the right prediction. It is useful to study the decision boundaries because we can observe how the different regularizers behave.


