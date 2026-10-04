/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the BSD license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const PropTypes = require("prop-types");
const React = require("react");

function SocialFooter(props) {
  const repoUrl = `https://github.com/${props.config.organizationName}/${props.config.projectName}`;
  return (
    <div className="footerSection">
      <h5>Social</h5>
      <div className="social">
        <a
          className="github-button" // part of the https://buttons.github.io/buttons.js script in siteConfig.js
          href={repoUrl}
          data-count-href={`${repoUrl}/stargazers`}
          data-show-count="true"
          data-count-aria-label="# stargazers on GitHub"
          aria-label="Star Captum on GitHub"
        >
          {props.config.projectName}
        </a>
      </div>
    </div>
  );
}

SocialFooter.propTypes = {
  config: PropTypes.object
};

class Footer extends React.Component {
  docUrl(doc, language) {
    const baseUrl = this.props.config.baseUrl;
    const docsUrl = this.props.config.docsUrl;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ""}`;
    const langPart = `${language ? `${language}/` : ""}`;
    return `${baseUrl}${docsPart}${langPart}${doc}`;
  }

  pageUrl(doc, language) {
    const baseUrl = this.props.config.baseUrl;
    return baseUrl + (language ? `${language}/` : "") + doc;
  }

  render() {
    const currentYear = new Date().getFullYear();
    return (
      <footer className="nav-footer" id="footer">
        <section className="sitemap">
          {this.props.config.footerIcon && (
            <a href={this.props.config.baseUrl} className="nav-home">
              <img
                src={`${this.props.config.baseUrl}${this.props.config.footerIcon}`}
                alt={this.props.config.title}
                width="66"
                height="58"
              />
            </a>
          )}
          <div className="footerSection">
            <h5>Docs</h5>
            <a href={this.docUrl("introduction")}>Introduction</a>
            <a href={this.docUrl("getting_started")}>Getting Started</a>
            <a href={`${this.props.config.baseUrl}tutorials/`}>Tutorials</a>
            <a href={`${this.props.config.baseUrl}api/`}>API Reference</a>
          </div>
          <div className="footerSection">
            <h5>Legal</h5>
            <a
              href="https://opensource.facebook.com/legal/privacy/"
              target="_blank"
              rel="noreferrer noopener"
            >
              Privacy
            </a>
            <a
              href="https://opensource.facebook.com/legal/terms/"
              target="_blank"
              rel="noreferrer noopener"
            >
              Terms
            </a>
          </div>
          <SocialFooter config={this.props.config} />
        </section>
        <a
          href="https://opensource.facebook.com/"
          target="_blank"
          rel="noreferrer noopener"
          className="fbOpenSource"
        >
          <img
            src={`${this.props.config.baseUrl}img/oss_logo.png`}
            alt="Facebook Open Source"
            width="170"
            height="45"
          />
        </a>
        <section className="copyright">
          {this.props.config.copyright && (
            <span>{this.props.config.copyright}</span>
          )}{" "}
          Copyright &copy; {currentYear} Meta Platforms, Inc.
        </section>
        {process.env.NODE_ENV !== 'development' &&
          <script dangerouslySetInnerHTML={{
            __html: `
            (function() {
              var BAD_BASE = '/captum/';
              if (window.location.origin !== '${this.props.config.url}') {
                var pathname = window.location.pathname;
                var newPathname = pathname.slice(pathname.indexOf(BAD_BASE) === 0 ? BAD_BASE.length : 1);
                var newLocation = '${this.props.config.url}${this.props.config.baseUrl}' + newPathname;
                console.log('redirecting to ' + newLocation);
                window.location.href = newLocation;
              }
            })();
          `}} />
        }
      </footer>
    );
  }
}

module.exports = Footer;
