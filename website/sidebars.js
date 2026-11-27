/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD license found in the
 * LICENSE file in the root directory of this source tree.
 */

const tutorials = () => {
  // tutorials.json shape: { <Category>: [ { id, title } | { title, children: [...] }, ... ] }
  const allTutorialMetadata = require('./tutorials.json');
  const buildItem = (tutorial) => ({
    type: 'doc',
    label: tutorial.title,
    id: `tutorials/${tutorial.id}/index`,
  });

  const tutorialsSidebar = [
    {
      type: 'doc',
      id: 'tutorials/index',
      label: 'Overview',
    },
  ];
  for (const category in allTutorialMetadata) {
    const items = [];
    for (const entry of allTutorialMetadata[category]) {
      if (entry.id) {
        items.push(buildItem(entry));
      } else if (entry.children) {
        items.push({
          type: 'category',
          label: entry.title,
          items: entry.children.map(buildItem),
        });
      }
    }
    tutorialsSidebar.push({
      type: 'category',
      label: category,
      collapsed: false,
      items,
    });
  }
  return tutorialsSidebar;
};

export default {
  docs: {
    About: ['introduction', 'overview'],
    General: [
      'getting_started',
      'attribution_algorithms',
      'algorithms_comparison_matrix',
      'faq',
      'contribution_guidelines',
    ],
    Usage: ['extension/integrated_gradients'],
  },
  tutorials: tutorials(),
};
