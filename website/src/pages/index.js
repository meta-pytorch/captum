/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import CodeBlock from '@theme/CodeBlock';
import Heading from '@theme/Heading';

const features = [
  {
    content:
      'Supports interpretability of models across modalities including vision, text, and more.',
    image: 'img/multi-modal.png',
    title: 'Multi-Modal',
  },
  {
    content:
      'Supports most types of PyTorch models and can be used with minimal modification to the original neural network.',
    image: 'img/pytorch_logo.svg',
    title: 'Built on PyTorch',
  },
  {
    content:
      'Open source, generic library for interpretability research. Easily implement and benchmark new algorithms.',
    image: 'img/expanding_arrows.svg',
    title: 'Extensible',
  },
];

const Feature = ({imageUrl, title, content, image}) => {
  const imgUrl = useBaseUrl(imageUrl);
  const resolvedImage = useBaseUrl(image);
  return (
    <div className="col col--4 feature text--center">
      {imgUrl && (
        <div>
          <img src={imgUrl} alt={title} />
        </div>
      )}
      {resolvedImage && (
        <div>
          <img
            className="margin--md"
            src={resolvedImage}
            alt={title}
            style={{width: '80px', height: '80px'}}
          />
        </div>
      )}
      <h2>{title}</h2>
      <p>{content}</p>
    </div>
  );
};

const Features = () => (
  <div className="padding--xl">
    <h2 className="text--center padding--md">Key Features</h2>
    {features && features.length > 0 && (
      <div className="row">
        {features.map(({title, imageUrl, content, image}) => (
          <Feature
            key={title}
            title={title}
            imageUrl={imageUrl}
            content={content}
            image={image}
          />
        ))}
      </div>
    )}
  </div>
);

const HomeSplash = () => {
  const {siteConfig} = useDocusaurusContext();
  return (
    <div
      className="homeContainer text--center"
      style={{height: '30rem'}}>
      <div className="padding-vert--md">
        <div className="productTitle">{siteConfig.title}</div>
        <p className="hero__subtitle text--secondary margin-vert--md">
          {siteConfig.tagline}
        </p>
      </div>
      <div>
        <Link
          to="/docs/introduction"
          className="button button--lg button--outline button--secondary margin--sm">
          Introduction
        </Link>
        <Link
          to="/#quickstart"
          className="button button--lg button--outline button--secondary margin--sm">
          Get Started
        </Link>
        <Link
          to="/docs/tutorials/Titanic_Basic_Interpret"
          className="button button--lg button--outline button--secondary margin--sm">
          Tutorials
        </Link>
      </div>
    </div>
  );
};

const VideoContainer = () => (
  <div className="container text--center margin-bottom--xl margin-top--lg">
    <div className="row">
      <div className="col" style={{textAlign: 'center'}}>
        <Heading as="h2">Check it out in the intro video</Heading>
        <div>
          <iframe
            width="560"
            height="315"
            src="https://www.youtube.com/embed/YYatLlT3tPI"
            title="Explain Like I'm 5: Captum"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        </div>
      </div>
    </div>
  </div>
);

export default () => {
  const {siteConfig} = useDocusaurusContext();

  const createModelExample = `import numpy as np

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 3)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(3, 2)

        # initialize weights and biases
        self.lin1.weight = nn.Parameter(torch.arange(-4.0, 5.0).view(3, 3))
        self.lin1.bias = nn.Parameter(torch.zeros(1, 3))
        self.lin2.weight = nn.Parameter(torch.arange(-3.0, 3.0).view(2, 3))
        self.lin2.bias = nn.Parameter(torch.ones(1, 2))

    def forward(self, input):
        return self.lin2(self.relu(self.lin1(input)))


model = ToyModel()
model.eval()`;

  const randomSeedsDefinition = `torch.manual_seed(123)
np.random.seed(123)`;

  const defineInputBaseline = `input = torch.rand(2, 3)
baseline = torch.zeros(2, 3)`;

  const instantiateApply = `ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
print('IG Attributions:', attributions)
print('Convergence Delta:', delta)`;

  const igOutput = `IG Attributions: tensor([[-0.5922, -1.5497, -1.0067],
                         [ 0.0000, -0.2219, -5.1991]])
Convergence Delta: tensor([2.3842e-07, -4.7684e-07])`;

  const QuickStart = () => (
    <div
      className="padding--xl"
      style={{backgroundColor: 'var(--ifm-color-emphasis-200)'}}>
      <Heading
        as="h2"
        className="text--center padding--md"
        id="quickstart">
        Get Started
      </Heading>
      <div>
        <ol>
          <li>
            <h3>Install Captum:</h3>
            via conda (recommended):
            <CodeBlock language="bash">conda install captum -c pytorch</CodeBlock>
            via pip:
            <CodeBlock language="bash">pip install captum</CodeBlock>
          </li>
          <li>
            <h3>Create and prepare model:</h3>
            <CodeBlock language="python">{createModelExample}</CodeBlock>
          </li>
          <li>
            <h3>To make computations deterministic, let's fix random seeds:</h3>
            <CodeBlock language="python">{randomSeedsDefinition}</CodeBlock>
          </li>
          <li>
            <h3>Define input and baseline tensors:</h3>
            <CodeBlock language="python">{defineInputBaseline}</CodeBlock>
          </li>
          <li>
            <h3>
              Select algorithm to instantiate and apply (Integrated Gradients in
              this example):
            </h3>
            <CodeBlock language="python">{instantiateApply}</CodeBlock>
          </li>
          <li>
            <h3>View Output:</h3>
            <CodeBlock language="python">{igOutput}</CodeBlock>
          </li>
        </ol>
      </div>
    </div>
  );

  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <HomeSplash />
      <VideoContainer />
      <Features />
      <QuickStart />
    </Layout>
  );
};
