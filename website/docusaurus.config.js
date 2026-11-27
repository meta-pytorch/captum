/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {themes as prismThemes} from 'prism-react-renderer';

// Preview builds (e.g. a fork's GitHub Pages) override the URL and base path
// and must not be indexed by search engines.
const isPreview = process.env.CAPTUM_PREVIEW === 'true';

module.exports = {
  title: 'Captum',
  tagline: 'Model Interpretability for PyTorch',
  url: process.env.CAPTUM_URL || 'https://captum.ai',
  baseUrl: process.env.CAPTUM_BASE_URL || '/',
  noIndex: isPreview,
  organizationName: 'meta-pytorch',
  projectName: 'captum',
  favicon: 'img/captum.ico',
  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',
  trailingSlash: false,
  markdown: {
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          showLastUpdateAuthor: false,
          showLastUpdateTime: false,
          editUrl: 'https://github.com/meta-pytorch/captum/edit/master/docs/',
          path: '../docs',
          // README.md describes the docs/ source layout and is not a published doc.
          exclude: ['README.md'],
          sidebarPath: '../website/sidebars.js',
        },
        blog: false,
        theme: {
          customCss: 'static/css/custom.css',
        },
      },
    ],
  ],
  plugins: [
    [
      '@docusaurus/plugin-client-redirects',
      {
        // Preserve v1 URL parity: v1 served `/tutorials/<id>`,
        // `/tutorials/<id>.html` and `/tutorials/<id>/`; tutorials now live
        // under `/docs/tutorials/<id>` (BoTorch convention).
        createRedirects(existingPath) {
          if (
            existingPath === '/docs/tutorials' ||
            existingPath.startsWith('/docs/tutorials/')
          ) {
            const oldPath = existingPath.replace('/docs/tutorials', '/tutorials');
            return [oldPath, `${oldPath}.html`];
          }
          return undefined;
        },
      },
    ],
  ],
  themeConfig: {
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
    navbar: {
      logo: {
        src: 'img/captum_logo.svg',
        alt: 'Captum',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docs',
          label: 'Docs',
          position: 'left',
        },
        {
          type: 'docSidebar',
          sidebarId: 'tutorials',
          label: 'Tutorials',
          position: 'left',
        },
        {
          // pathname:// prefix bypasses the Docusaurus router, serving the
          // Sphinx-generated HTML at static/api/* directly.
          href: 'pathname:///api/',
          label: 'API Reference',
          position: 'left',
          target: '_blank',
        },
        {
          href: 'https://github.com/meta-pytorch/captum',
          className: 'header-github-link',
          'aria-label': 'GitHub',
          position: 'right',
        },
      ],
    },
    image: 'img/captum-icon.png',
    footer: {
      style: 'dark',
      logo: {
        alt: 'Captum',
        src: 'img/captum_logo.svg',
      },
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Introduction',
              to: 'docs/introduction',
            },
            {
              label: 'Getting Started',
              to: 'docs/getting_started',
            },
            {
              label: 'Tutorials',
              to: '/docs/tutorials/',
            },
            {
              label: 'API Reference',
              href: 'pathname:///api/',
              target: '_blank',
            },
          ],
        },
        {
          title: 'Legal',
          // Please do not remove the privacy and terms, it's a legal requirement.
          items: [
            {
              label: 'Privacy',
              href: 'https://opensource.facebook.com/legal/privacy/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
            {
              label: 'Terms',
              href: 'https://opensource.facebook.com/legal/terms/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
          ],
        },
        {
          title: 'Social',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/meta-pytorch/captum',
            },
            {
              html: `<iframe
                src="https://ghbtns.com/github-btn.html?user=meta-pytorch&amp;repo=captum&amp;type=star&amp;count=true&amp;size=small"
                title="GitHub Stars"
              />`,
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Meta Platforms, Inc.`,
    },
  },
};
