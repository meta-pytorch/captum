/**
 * Copyright (c) 2019-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
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
                <a href="https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py">
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
                The Captum tutorials are grouped into the following four areas.
              </p>
              <p>
                <h4>Getting started with Captum:</h4>
                In this tutorial we create and train a simple neural network on the Titanic survival dataset.
                We then use Integrated Gradients to analyze feature importance.  We then deep dive the network to assess layer and neuron importance
                using conductance.  Finally, we analyze a specific
                neuron to understand feature importance for that specific neuron.  Find the tutorial <a href="/tutorials/Titanic_Basic_Interpret">here</a>.

                <h4>Interpreting text models:</h4>
                In this tutorial we use a pre-trained CNN model for sentiment analysis on an IMDB dataset.
                We use Captum and Integrated Gradients to interpret model predictions by show which specific
                words have highest attribution to the model output.  Find the tutorial <a href="/tutorials/IMDB_TorchText_Interpret">here </a>.

                <h4>Interpreting vision models:</h4>
                This tutorial demonstrates how to use Captum for interpreting vision focused models.
                First we create and train (or use a pre-trained) a simple CNN model on the CIFAR dataset.
                We then interpret the output of an example with a series of overlays using Integrated Gradients and DeepLIFT.
                Find the tutorial <a href="/tutorials/CIFAR_TorchVision_Interpret">here</a>.

                <h4>Interpreting multimodal models:</h4>
                To demonstrate interpreting multimodal models we have chosen to look at an open source Visual Question Answer (VQA) model.
                Using Captum and Integrated Gradients we interpret the output of several test questions and analyze the attribution scores
                of the text and visual parts of the model. Find the tutorial <a href="/tutorials/Multimodal_VQA_Interpret">here</a>.

              </p>
            </body>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
