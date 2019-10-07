from collections import namedtuple
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple, Union

from captum.attr import IntegratedGradients
from captum.attr._utils.batching import _batched_generator
from captum.attr._utils.common import _run_forward
from captum.insights.features import BaseFeature

import torch
from torch import Tensor
from torch.nn import Module

PredictionScore = namedtuple("PredictionScore", "score label")
VisualizationOutput = namedtuple(
    "VisualizationOutput", "feature_outputs actual predicted"
)
Contribution = namedtuple("Contribution", "name percent")


class FilterConfig(NamedTuple):
    steps: int = 25
    prediction: str = "all"
    classes: List[str] = []
    count: int = 4


class Data:
    def __init__(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        labels: Optional[Tensor],
        additional_args=None,
    ):
        self.inputs = inputs
        self.labels = labels
        self.additional_args = additional_args


class AttributionVisualizer(object):
    def __init__(
        self,
        models: Union[List[Module], Module],
        classes: List[str],
        features: Union[List[BaseFeature], BaseFeature],
        dataset: Iterable[Data],
        score_func: Optional[Callable] = None,
    ):
        if not isinstance(models, List):
            models = [models]

        if not isinstance(features, List):
            features = [features]

        self.models = models
        self.classes = classes
        self.features = features
        self.dataset = dataset
        self.score_func = score_func
        self._config = FilterConfig(steps=25, prediction="all", classes=[], count=4)

    def _calculate_attribution(
        self,
        net: Module,
        baselines: Optional[List[Tuple[Tensor, ...]]],
        data: Tuple[Tensor, ...],
        additional_forward_args: Optional[Tuple[Tensor, ...]],
        label: Optional[Tensor],
    ) -> Tensor:
        ig = IntegratedGradients(net)
        # TODO support multiple baselines
        baseline = baselines[0] if len(baselines) > 0 else None
        label = None if label is None or label.nelement() == 0 else label
        attr_ig, _ = ig.attribute(
            data,
            baselines=baseline,
            additional_forward_args=additional_forward_args,
            target=label,
            n_steps=self._config.steps,
        )

        return attr_ig

    def _update_config(self, settings):
        self._config = FilterConfig(
            steps=int(settings["approximation_steps"]),
            prediction=settings["prediction"],
            classes=settings["classes"],
            count=4,
        )

    def render(self, blocking=False, debug=False):
        from IPython.display import IFrame, display
        from captum.insights.server import start_server

        port = start_server(self, blocking, debug)

        display(IFrame(src=f"http://127.0.0.1:{port}", width="100%", height="500px"))

    def _get_labels_from_scores(
        self, scores: Tensor, indices: Tensor
    ) -> List[PredictionScore]:
        pred_scores = []
        for i in range(len(indices)):
            score = scores[i].item()
            pred_scores.append(PredictionScore(score, self.classes[indices[i]]))
        return pred_scores

    def _transform(
        self,
        transforms: Union[Callable, List[Callable]],
        inputs: Tensor,
        batch: bool = False,
    ) -> Tensor:
        transformed_inputs = inputs
        # TODO support batch size > 1
        if batch:
            transformed_inputs = inputs.squeeze()

        if isinstance(transforms, List):
            for t in transforms:
                transformed_inputs = t(transformed_inputs)
        else:
            transformed_inputs = transforms(transformed_inputs)

        if batch:
            transformed_inputs.unsqueeze_(0)

        return transformed_inputs

    def _calculate_net_contrib(self, attrs_per_input_feature: List[Tensor]):
        # get the net contribution per feature (input)
        net_contrib = torch.stack(
            [attrib.flatten().sum() for attrib in attrs_per_input_feature]
        )

        # normalise the contribution, s.t. sum(abs(x_i)) = 1
        norm = torch.norm(net_contrib, p=1)
        if norm > 0:
            net_contrib /= norm

        return net_contrib.tolist()

    def _predictions_matches_labels(
        self,
        predicted_scores: List[PredictionScore],
        actual_labels: Union[str, List[str]],
    ) -> bool:
        if len(predicted_scores) == 0:
            return False

        predicted_label = predicted_scores[0].label

        if isinstance(actual_labels, List):
            return predicted_label in actual_labels

        return actual_labels == predicted_label

    def _should_keep_prediction(
        self, predicted_scores: List[PredictionScore], actual_label: str
    ) -> bool:
        # filter by class
        if len(self._config.classes) != 0:
            if not self._predictions_matches_labels(
                predicted_scores, self._config.classes
            ):
                return False

        # filter by accuracy
        if self._config.prediction == "all":
            pass
        elif self._config.prediction == "correct":
            if not self._predictions_matches_labels(predicted_scores, actual_label):
                return False
        elif self._config.prediction == "incorrect":
            if self._predictions_matches_labels(predicted_scores, actual_label):
                return False
        else:
            raise Exception(f"Invalid prediction config: {self._config.prediction}")

        return True

    def _get_outputs(self) -> List[VisualizationOutput]:
        batch_data = next(self.dataset)
        net = self.models[0]  # TODO process multiple models
        vis_outputs = []

        for inputs, additional_forward_args, label in _batched_generator(
            inputs=batch_data.inputs,
            additional_forward_args=batch_data.additional_args,
            target_ind=batch_data.labels,
            internal_batch_size=1,  # should be 1 until we have batch label support
        ):
            # initialize baselines
            baseline_transforms_len = len(self.features[0].baseline_transforms or [])
            baselines = [
                [None] * len(self.features) for _ in range(baseline_transforms_len)
            ]
            transformed_inputs = list(inputs)

            for feature_i, feature in enumerate(self.features):
                if feature.input_transforms is not None:
                    transformed_inputs[feature_i] = self._transform(
                        feature.input_transforms, transformed_inputs[feature_i], True
                    )
                if feature.baseline_transforms is not None:
                    assert baseline_transforms_len == len(
                        feature.baseline_transforms
                    ), "Must have same number of baselines across all features"

                    for baseline_i, baseline_transform in enumerate(
                        feature.baseline_transforms
                    ):
                        baselines[baseline_i][feature_i] = self._transform(
                            baseline_transform, transformed_inputs[feature_i], True
                        )

            outputs = _run_forward(
                net, tuple(transformed_inputs), additional_forward_args
            )

            if self.score_func is not None:
                outputs = self.score_func(outputs)

            if outputs.nelement() == 1:
                scores = outputs
                predicted = scores.round().to(torch.int)
            else:
                scores, predicted = outputs.topk(min(4, outputs.shape[-1]))

            scores = scores.cpu().squeeze(0)
            predicted = predicted.cpu().squeeze(0)

            actual_label = self.classes[label[0]] if label is not None else None
            predicted_scores = self._get_labels_from_scores(scores, predicted)

            # Filter based on UI configuration
            if not self._should_keep_prediction(predicted_scores, actual_label):
                continue

            baselines = [tuple(b) for b in baselines]

            # attributions are given per input*
            # inputs given to the model are described via `self.features`
            #
            # *an input contains multiple features that represent it
            #   e.g. all the pixels that describe an image is an input
            attrs_per_input_feature = self._calculate_attribution(
                net,
                baselines,
                tuple(transformed_inputs),
                additional_forward_args,
                label,
            )

            net_contrib = self._calculate_net_contrib(attrs_per_input_feature)

            # the features per input given
            features_per_input = [
                feature.visualize(attr, data, contrib)
                for feature, attr, data, contrib in zip(
                    self.features, attrs_per_input_feature, inputs, net_contrib
                )
            ]

            output = VisualizationOutput(
                feature_outputs=features_per_input,
                actual=actual_label,
                predicted=predicted_scores,
            )

            vis_outputs.append(output)

        return vis_outputs

    def visualize(self):
        output_list = []
        while len(output_list) < self._config.count:
            try:
                output_list.extend(self._get_outputs())
            except StopIteration:
                break
        return output_list
