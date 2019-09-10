import React from "react";
import "./App.css";

// helper method to convert a POJO into a valid classname
function cx(obj) {
  return Object.keys(obj)
    .filter(k => !!obj[k])
    .join(" ");
}

class Header extends React.Component {
  render() {
    return (
      <header className="header">
        <div className="header__name">Captum Insights</div>
        <nav className="header__nav">
          <ul>
            <li className="header__nav__item header__nav__item--active">
              Instance Attribution
            </li>
            <li className="header__nav__item">Direct Target</li>
            <li className="header__nav__item">Export</li>
          </ul>
        </nav>
      </header>
    );
  }
}

class FilterContainer extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      instance_type: "all",
      approximation_steps: 50
    };
  }

  handleInputChange = event => {
    const target = event.target;
    const value = target.type === "checkbox" ? target.checked : target.value;
    const name = target.name;
    this.setState({
      [name]: value
    });
  };

  handleSubmit = event => {
    this.props.fetchData(this.state);
    event.preventDefault();
  };

  render() {
    return (
      <Filter
        instanceType={this.state.instance_type}
        approximationSteps={this.state.approximation_steps}
        onHandleInputChange={this.handleInputChange}
        handleSubmit={this.handleSubmit}
      />
    );
  }
}

class Filter extends React.Component {
  render() {
    return (
      <form onSubmit={this.props.handleSubmit}>
        <div className="filter-panel">
          <div className="filter-panel__column">
            <div className="filter-panel__column__title">Filter by Classes</div>
            <div className="filter-panel__column__body">
              Animal and 2 other classes are selected. <a href="">Edit</a>
            </div>
          </div>
          <div className="filter-panel__column">
            <div className="filter-panel__column__title">
              Filter by Instances
            </div>
            <div className="filter-panel__column__body">
              Instance Type:{" "}
              <select
                className="select"
                name="instance_type"
                value={this.props.isntanceType}
                onChange={this.props.handleInputChange}
              >
                <option value="all">All</option>
                <option value="false_negative">False Negative</option>
                <option value="false_positive">False Positive</option>
              </select>
            </div>
          </div>
          <div className="filter-panel__column">
            <div className="filter-panel__column__title">
              Integrated Gradients
            </div>
            <div className="filter-panel__column__body">
              Approximation steps:{" "}
              <input
                className="input"
                name="approximation_steps"
                type="number"
                value={this.props.approximationSteps}
                onChange={this.props.handleInputChange}
              />
            </div>
          </div>
          <div className="filter-panel__column filter-panel__column--end">
            <button className="btn btn--outline btn--large">Fetch</button>
          </div>
        </div>
      </form>
    );
  }
}

function ImageFeature(props) {
  return (
    <>
      <div className="panel__column__title">{props.feature.name} (Image)</div>
      <div className="panel__column__body">
        <div className="gallery">
          <div className="gallery__item">
            <div className="gallery__item__image">
              <img src={"data:image/png;base64," + props.feature.base} />
            </div>
            <div className="gallery__item__description">Original</div>
          </div>
          <div className="gallery__item">
            <div className="gallery__item__image">
              <img src={"data:image/png;base64," + props.feature.modified} />
            </div>
            <div className="gallery__item__description">Gradient Overlay</div>
          </div>
        </div>
      </div>
    </>
  );
}

function get_feature(f) {
  let feature = null;
  switch (f.type) {
    case "image":
      feature = <ImageFeature feature={f} />;
      break;

    default:
      throw new Error("Unsupported feature visualization type: " + f.type);
  }
  return feature;
}

class Contributions extends React.Component {
  _getColorClassName(percentage) {
    if (percentage > 50) {
      return "bar-blue";
    } else if (percentage > 10) {
      return "bar-light-blue";
    } else if (percentage > -10) {
      return "bar-gray";
    } else if (percentage > -50) {
      return "bar-light-red";
    } else {
      return "bar-red";
    }
  }

  render() {
    return this.props.feature_outputs.map(f => (
      <div className="bar-chart__group">
        <div
          className={cx({
            "bar-chart__group__bar": true,
            [this._getColorClassName(f.contribution)]: true
          })}
          width={f.contribution + "%"}
        />
        <div className="bar-chart__group__title">{f.name}</div>
      </div>
    ));
  }
}

class Visualization extends React.Component {
  render() {
    const data = this.props.data;
    const features = data.feature_outputs.map(f => get_feature(f));

    return (
      <div className="panel panel--long">
        <div className="panel__column">
          <div className="panel__column__title">Predicted</div>
          <div className="panel__column__body">
            {data.predicted.map((p, i) => (
              <div className="row row--padding">
                <div
                  className={cx({
                    btn: true,
                    "btn--solid": i === 0,
                    "btn--outline": i !== 0
                  })}
                >
                  {p.label} ({p.score.toFixed(3)})
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="panel__column">
          <div className="panel__column__title">Label</div>
          <div className="panel__column__body">
            <div className="row row--padding">
              <div className="btn btn--outline">{data.actual}</div>
            </div>
          </div>
        </div>
        <div className="panel__column">
          <div className="panel__column__title">Contribution</div>
          <div className="panel__column__body">
            <div className="bar-chart">
              <Contributions feature_outputs={data.feature_outputs} />
            </div>
          </div>
        </div>
        <div className="panel__column panel__column--stretch">{features}</div>
      </div>
    );
  }
}

function Visualizations(props) {
  if (!props.data || props.data.length === 0) {
    return (
      <div className="viz">
        <div className="panel">
          <div className="filter-panel__column">
            Please press <strong>Fetch</strong> to start loading data.
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="viz">
      {props.data.map((v, i) => (
        <Visualization data={v} key={i} />
      ))}
    </div>
  );
}

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      data: []
    };
  }

  fetchData = filter_config => {
    console.log("filter config: ", filter_config);
    fetch("/fetch", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(filter_config)
    })
      .then(response => response.json())
      .then(response => this.setState({ data: response }));
  };

  render() {
    return (
      <div className="app">
        <Header />
        <FilterContainer fetchData={this.fetchData} />
        <Visualizations data={this.state.data} />
      </div>
    );
  }
}

export default App;
