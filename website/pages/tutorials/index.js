/**
 * Copyright (c) 2019-present, Facebook, Inc.
 *
 * This source code is licensed under the BSD license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const CWD = process.cwd();

const CompLibrary = require(`${CWD}/node_modules/docusaurus/lib/core/CompLibrary.js`);
const Container = CompLibrary.Container;
const MarkdownBlock = CompLibrary.MarkdownBlock;

const TutorialSidebar = require(`${CWD}/core/TutorialSidebar.js`);

class TutorialHome extends React.Component {
  render() {
    return (
      <div className="docMainWrapper wrapper">
        <TutorialSidebar currentTutorialID={null} />
        <Container className="mainContainer documentContainer postContainer">
          <div className="post">
            <header className="postHeader">
              <h1 className="postHeaderTitle">Captum Tutorials</h1>
            </header>
            <body>
              <p>
                The tutorials here will help you understand and use Captum. They assume that you are familiar with PyTorch and its basic features.
              </p>
              <p>
                If you are new to Captum, the easiest way to get started is
                with the{' '}
                <a href="Titanic_Basic_Interpret">
                  Getting started with Captum
                </a>{' '}
                tutorial.
              </p>
              <p>
                If you are new to PyTorch, the easiest way to get started is
                with the{' '}
                <a href="https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py">
                  What is PyTorch?
                </a>{' '}
                tutorial.
              </p>

              <p>
                <h4>Getting started with Captum:</h4>
                In this tutorial we create and train a simple neural network on the Titanic survival dataset.
                We then use Integrated Gradients to analyze feature importance.  We then deep dive the network to assess layer and neuron importance
                using conductance.  Finally, we analyze a specific
                neuron to understand feature importance for that specific neuron.  Find the tutorial <a href="Titanic_Basic_Interpret">here</a>.

                <h3>Attribution</h3>

                <h4>Interpreting text models:</h4>
                In this tutorial we use a pre-trained CNN model for sentiment analysis on an IMDB dataset.
                We use Captum and Integrated Gradients to interpret model predictions by show which specific
                words have highest attribution to the model output.  Find the tutorial <a href="IMDB_TorchText_Interpret">here </a>.

                <h4>Interpreting vision with CIFAR:</h4>
                This tutorial demonstrates how to use Captum for interpreting vision focused models.
                First we create and train (or use a pre-trained) a simple CNN model on the CIFAR dataset.
                We then interpret the output of an example with a series of overlays using Integrated Gradients and DeepLIFT.
                Find the tutorial <a href="CIFAR_TorchVision_Interpret">here</a>.

                <h4>Interpreting vision with Pretrained models:</h4>
                Like the CIFAR based tutorial above, this tutorial demonstrates how to use Captum for interpreting vision-focused models.
                This tutorial begins with a pretrained resnet18 and VGG16 model and demonstrates how to use Intergrated Gradients along with Noise Tunnel,
                GradientShap, Occlusion, and LRP.
                Find the tutorial <a href="TorchVision_Interpret">here</a>.

                <h4>Feature ablation on images:</h4>
                This tutorial demonstrates feature ablation in Captum, applied on images as an example.
                It leverages segmentation masks to define ablation groups over the input features.
                We show how this kind of analysis helps understanding which parts of the input impacts a certain target in the model.
                Find the tutorial <a href="Resnet_TorchVision_Ablation">here</a>.

                <h4>Interpreting multimodal models:</h4>
                To demonstrate interpreting multimodal models we have chosen to look at an open source Visual Question Answer (VQA) model.
                Using Captum and Integrated Gradients we interpret the output of several test questions and analyze the attribution scores
                of the text and visual parts of the model. Find the tutorial <a href="Multimodal_VQA_Interpret">here</a>.

                <h4>Understanding Llama2 with Captum LLM Attribution:</h4>
                This tutorial demonstrates how to easily use the LLM attribution functionality to interpret the large langague models (LLM) in text generation.
                It takes Llama2 as the example and shows the step-by-step improvements from the basic attribution setting to more advanced techniques.
                Find the tutorial <a href="Llama2_LLM_Attribution">here</a>.

                <h4>Interpreting question answering with BERT Part 1:</h4>
                This tutorial demonstrates how to use Captum to interpret a BERT model for question answering.
                We use a pre-trained model from Hugging Face fine-tuned on the SQUAD dataset and show how to use hooks to
                examine and better understand embeddings, sub-embeddings, BERT, and attention layers.
                Find the tutorial <a href="Bert_SQUAD_Interpret">here</a>.

                <h4>Interpreting question answering with BERT Part 2:</h4>
                In the second part of Bert tutorial we analyze attention matrices using attribution algorithms s.a. Integrated Gradients.
                This analysis helps us to identify strong interaction pairs between different tokens for a specific model prediction.
                We compare our findings with the <a href="https://arxiv.org/pdf/2004.10102.pdf">vector norms</a> and show that attribution scores
                are more meaningful compared to the vector norms.
                Find the tutorial <a href="Bert_SQUAD_Interpret2">here</a>.

                <h4>Interpreting a regression model of California house prices:</h4>
                To demonstrate interpreting regression models we have chosen to look at the California house prices dataset.
                Using Captum and a variety of attribution methods, we evaluate feature importance as well as internal attribution to understand
                the network function. Find the tutorial <a href="House_Prices_Regression_Interpret">here</a>.

                <h4>Interpreting a semantic segmentation model:</h4>
                In this tutorial, we demonstrate applying Captum to a semantic segmentation task to understand what pixels
                and regions contribute to the labeling of a particular class. We explore applying GradCAM as well as Feature Ablation
                to a pretrained Fully-Convolutional Network model with a ResNet-101 backbone. Find the tutorial <a href="Segmentation_Interpret">here</a>.

                <h4>Using Captum with torch.distributed:</h4>
                This tutorial provides examples of using Captum with the torch.distributed package and DataParallel,
                allowing attributions to be computed in a distributed manner across processors, machines or GPUs.
                Find the tutorial <a href="Distributed_Attribution">here</a>.

                <h4>Interpreting DLRM models with Captum:</h4>
                This tutorial demonstrates how we use Captum for Deep Learning Recommender Models using
                 <a href="https://github.com/facebookresearch/dlrm">dlrm</a> model published from facebook research and integrated gradients algorithm.
                It showcases feature importance differences for sparse and dense features in predicting clicked and non-clicked Ads. It also analyzes
                the importance of feature interaction layer and neuron importances in the final fully connected layer when predicting clicked Ads.
                Find the tutorial <a href="DLRM_Tutorial">here</a>.

                <h4>Interpreting vision and text models with LIME:</h4>
                This tutorial demonstrates how to interpret computer vision and text classification models using Local Interpretable Model-agnostic
                Explanations (LIME) algorithm. For vision it uses resnet18 model to explain image classification based on super-pixels extracted by
                a segmentation mask. For text it uses a classification model trained on `AG_NEWS` dataset and explains model predictions based
                on the word tokens in the input text.
                Find the tutorial <a href="Image_and_Text_Classification_LIME">here</a>.

                <h3>Robustness</h3>

                <h4>Applying robustness attacks and metrics to CIFAR model and dataset:</h4>
                This tutorial demonstrates how to apply robustness attacks such as FGSM and PGD as well as robustness metrics such as
                MinParamPerturbation and AttackComparator to a model trained on CIFAR dataset. Apart from that it also demonstrates how
                robustness techniques can be used in conjunction with attribution algorithms.
                Find the tutorial <a href="CIFAR_Captum_Robustness">here</a>.

                <h3>Concept</h3>

                <h4>TCAV for image classification for googlenet model:</h4>
                This tutorial demonstrates how to apply Testing with Concept Activation Vectors (TCAV) algorithm on image classification problem.
                It uses googlenet model and imagenet images to showcase the effectiveness of TCAV algorithm on interpreting zebra predictions through
                stripes concepts.
                Find the tutorial <a href="TCAV_Image">here</a>.

                <h4>TCAV for NLP sentiment analysis model:</h4>
                This tutorial demonstrates how to apply TCAV algorithm for a NLP task using movie rating dataset and a CNN-based binary
                sentiment classification model. It showcases that `positive adjectives` concept plays a significant role in predicting
                positive sentiment.
                Find the tutorial <a href="TCAV_NLP">here</a>.

                <h3>Influential Examples</h3>

                <h4>Identifying influential examples and mis-labelled examples with TracInCP:</h4>
                This tutorial demonstrates two use cases of the TracInCP method: providing interpretability by identifying influential training examples
                for a given prediction, and identifying mis-labelled examples. These two use cases are demonstrated using the CIFAR dataset and
                checkpoints obtained from training a simple CNN model on it (which can also be downloaded to avoid training).
                Find the tutorial <a href="TracInCP_Tutorial">here</a>.

                <h3>Captum Insight</h3>

                <h4>Getting Started with Captum Insights:</h4>
                This tutorial demonstrates how to use Captum Insights for a vision model in a notebook setting.  A simple pretrained torchvision
                CNN model is loaded and then used on the CIFAR dataset.  Captum Insights is then loaded to visualize the interpretation of specific examples.
                Find the tutorial <a href="CIFAR_TorchVision_Captum_Insights">here</a>.

                <h4>Using Captum Insights with multimodal models (VQA):</h4>
                This tutorial demonstrates how to use Captum Insights for visualizing attributions of a multimodal model, particularly an open
                source Visual Question Answer (VQA) model.
                Find the tutorial <a href="Multimodal_VQA_Captum_Insights">here</a>.
              </p>
            </body>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
