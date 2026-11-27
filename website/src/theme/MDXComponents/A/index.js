/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Swizzled from @docusaurus/theme-classic/src/theme/MDXComponents/A.
 *
 * Why this exists:
 *   Docusaurus' built-in `@docusaurus/mdx-loader` `transformLinks` remark plugin
 *   visits Markdown `link` AST nodes and, for URLs that resolve against a
 *   `staticDirectories` entry, rewrites them to a `require()` and attaches
 *   `data-noBrokenLinkCheck` so the broken-link checker leaves them alone.
 *   That plugin does NOT visit `mdxJsxFlowElement` / `mdxJsxTextElement` nodes,
 *   so a raw HTML `<a href="/img/foo.png">…</a>` written inside a Markdown
 *   file (Captum's algorithms_comparison_matrix.md does this) bypasses asset
 *   resolution and gets reported as a broken link.
 *
 *   This swizzle mirrors what Docusaurus does for Markdown asset links: when
 *   the href looks like an internal static asset (absolute path with a
 *   non-page file extension), pass `data-noBrokenLinkCheck` through to the
 *   underlying `<Link>` so the checker treats it the same way.
 */

import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {useAnchorTargetClassName} from '@docusaurus/theme-common';

const PAGE_LIKE_EXT = /\.(?:mdx?|html)$/i;

function isInternalStaticAssetHref(href) {
  if (typeof href !== 'string' || !href.startsWith('/')) {
    return false;
  }
  // Strip query/hash for extension detection.
  const pathname = href.split(/[?#]/, 1)[0];
  const dotIdx = pathname.lastIndexOf('.');
  if (dotIdx < 0) {
    return false;
  }
  const ext = pathname.slice(dotIdx);
  // Treat any non-page file extension as a static asset, mirroring the
  // `hasAssetLikeExtension` heuristic in @docusaurus/mdx-loader.
  return !PAGE_LIKE_EXT.test(ext);
}

export default function MDXA(props) {
  const anchorTargetClassName = useAnchorTargetClassName(props.id);
  const skipBrokenLinkCheck =
    isInternalStaticAssetHref(props.href) || undefined;
  return (
    <Link
      {...props}
      data-noBrokenLinkCheck={skipBrokenLinkCheck}
      className={clsx(anchorTargetClassName, props.className)}
    />
  );
}
