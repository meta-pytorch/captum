# Captum Website

This directory contains the source for the [captum.ai](https://captum.ai) website,
built with [Docusaurus 3.9](https://docusaurus.io/).

## Requirements

- **Node.js 22** (Docusaurus 3.9 requires Node 20+; CI uses Node 22)
- **Yarn** (Classic / v1.x)
- For the full build: Python with `pip install -e .[dev]` plus the
  `nbconvert`, `mdformat` and `pandas` versions pinned in
  `.github/workflows/website-deploy.yml`

## Build

From the repository root, `./scripts/build_docs.sh` builds the Sphinx API
reference into `website/static/api/`, converts the tutorial notebooks into
`docs/tutorials/`, and writes the static site to `website/build/`. Then
`python3 scripts/check_website_links.py website/build` checks every internal
link.

For changes to the Docusaurus pages only:

```bash
cd website
yarn install --frozen-lockfile --ignore-scripts
yarn start          # dev server with hot reload (http://localhost:3000)
yarn build          # production build → ./build
yarn serve          # serve the production build
```

## Preview builds

Running the "Deploy to GitHub Pages" workflow manually on a fork deploys a
preview to `https://<owner>.github.io/<repo>/`. The build reads
`CAPTUM_PREVIEW`, `CAPTUM_URL` and `CAPTUM_BASE_URL`; preview builds are
marked `noindex` and ship no `CNAME`.

## Source layout

| Path | Purpose |
|---|---|
| `docusaurus.config.js` | Site configuration |
| `sidebars.js` | Docs and tutorials sidebars (tutorials come from `tutorials.json`) |
| `src/pages/index.js` | Homepage |
| `src/theme/MDXComponents/A/` | Swizzled link component: raw `<a href="/img/...">` links in docs skip the broken-link check, as Markdown asset links do |
| `static/` | Verbatim-served files (CNAME, images, css) |
| `../docs/` | Markdown source for `/docs/*` pages |

## Resolutions

See [`RESOLUTIONS.md`](./RESOLUTIONS.md) for the reason behind each entry in
`package.json#resolutions`.
