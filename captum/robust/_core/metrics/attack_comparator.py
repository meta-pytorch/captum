#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import warnings
from collections import namedtuple
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from captum._utils.common import (
    _expand_additional_forward_args,
    _format_additional_forward_args,
    _reduce_list,
)
from captum.attr import Max, Mean, Min, Summarizer
from captum.log import log_usage
from captum.robust._core.perturbation import Perturbation
from torch import Tensor

ORIGINAL_KEY = "Original"

MetricResultType = TypeVar(
    "MetricResultType", float, Tensor, Tuple[Union[float, Tensor], ...]
)


class AttackInfo(NamedTuple):
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    attack_fn: Union[Perturbation, Callable]
    name: str
    num_attempts: int
    apply_before_preproc: bool
    attack_kwargs: Dict[str, Any]
    additional_args: List[str]


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def agg_metric(inp):
    if isinstance(inp, Tensor):
        return inp.mean(dim=0)
    elif isinstance(inp, tuple):
        return tuple(agg_metric(elem) for elem in inp)
    return inp


class AttackComparator(Generic[MetricResultType]):
    r"""
    Allows measuring model robustness for a given attack or set of attacks. This class
    can be used with any metric(s) as well as any set of attacks, either based on
    attacks / perturbations from captum.robust such as FGSM or PGD or external
    augmentation methods or perturbations such as torchvision transforms.
    """

    def __init__(
        self,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        forward_func: Callable,
        metric: Callable[..., MetricResultType],
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        preproc_fn: Optional[Callable] = None,
    ) -> None:
        r"""
        Args:
            forward_func (Callable or torch.nn.Module): This can either be an instance
                of pytorch model or any modification of a model's forward
                function.

            metric (Callable): This function is applied to the model output in
                order to compute the desired performance metric or metrics.
                This function should have the following signature::

                    >>> def model_metric(model_out: Tensor, **kwargs: Any)
                    >>>     -> Union[float, Tensor, Tuple[Union[float, Tensor], ...]:

                All kwargs provided to evaluate are provided to the metric function,
                following the model output. A single metric can be returned as
                a float or tensor, and multiple metrics should be returned as either
                a tuple or named tuple of floats or tensors. For a tensor metric,
                the first dimension should match the batch size, corresponding to
                metrics for each example. Tensor metrics are averaged over the first
                dimension when aggregating multiple batch results.
                If tensor metrics represent results for the full batch, the size of the
                first dimension should be 1.

            preproc_fn (Callable, optional): Optional method applied to inputs. Output
                of preproc_fn is then provided as input to model, in addition to
                additional_forward_args provided to evaluate.
                Default: ``None``
        """
        self.forward_func = forward_func
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        self.metric: Callable = metric
        self.preproc_fn = preproc_fn
        self.attacks: Dict[str, AttackInfo] = {}
        self.summary_results: Dict[str, Summarizer] = {}
        # pyre-fixme[4]: Attribute must be annotated.
        self.metric_aggregator = agg_metric
        self.batch_stats = [Mean, Min, Max]
        self.aggregate_stats = [Mean]
        self.summary_results = {}
        # pyre-fixme[4]: Attribute must be annotated.
        self.out_format = None

    def add_attack(
        self,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        attack: Union[Perturbation, Callable],
        name: Optional[str] = None,
        num_attempts: int = 1,
        apply_before_preproc: bool = True,
        attack_kwargs: Optional[Dict[str, Any]] = None,
        additional_attack_arg_names: Optional[List[str]] = None,
    ) -> None:
        r"""
        Adds attack to be evaluated when calling evaluate.

        Args:

            attack (Perturbation or Callable): This can either be an instance
                of a Captum Perturbation / Attack
                or any other perturbation or attack function such
                as a torchvision transform.

            name (str, optional): Name or identifier for attack, used as key for
                attack results. This defaults to attack.__class__.__name__
                if not provided and must be unique for all added attacks.
                Default: ``None``

            num_attempts (int, optional): Number of attempts that attack should be
                repeated. This should only be set to > 1 for non-deterministic
                attacks. The minimum, maximum, and average (best, worst, and
                average case) are tracked for attack attempts.
                Default: ``1``

            apply_before_preproc (bool, optional): Defines whether attack should be
                applied before or after preproc function.
                Default: ``True``

            attack_kwargs (dict, optional): Additional arguments to be provided to
                given attack. This should be provided as a dictionary of keyword
                arguments.
                Default: ``None``

            additional_attack_arg_names (list[str], optional): Any additional
                arguments for the attack which are specific to the particular input
                example or batch. An example of this is target, which is necessary
                for some attacks such as FGSM or PGD. These arguments are included
                if provided as a kwarg to evaluate.
                Default: ``None``
        """
        if name is None:
            name = attack.__class__.__name__

        if attack_kwargs is None:
            attack_kwargs = {}

        if additional_attack_arg_names is None:
            additional_attack_arg_names = []

        if name in self.attacks:
            raise RuntimeError(
                "Cannot add attack with same name as existing attack {}".format(name)
            )

        self.attacks[name] = AttackInfo(
            attack_fn=attack,
            name=name,
            num_attempts=num_attempts,
            apply_before_preproc=apply_before_preproc,
            attack_kwargs=attack_kwargs,
            additional_args=additional_attack_arg_names,
        )

    def _format_summary(
        self,
        # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
        #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting
        #  errors.
        summary: Union[Dict, List[Dict]],
    ) -> Dict[str, MetricResultType]:
        r"""
        This method reformats a given summary; particularly for tuples,
        the Summarizer's summary format is a list of dictionaries,
        each containing the summary for the corresponding elements.
        We reformat this to return a dictionary with tuples containing
        the summary results.
        """
        if isinstance(summary, dict):
            return summary
        else:
            # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
            summary_dict: Dict[str, Tuple] = {}
            for key in summary[0]:
                summary_dict[key] = tuple(s[key] for s in summary)
                if self.out_format:
                    summary_dict[key] = self.out_format(*summary_dict[key])
            return summary_dict  # type: ignore

    def _update_out_format(
        self, out_metric: Union[float, Tensor, Tuple[Union[float, Tensor], ...]]
    ) -> None:
        if (
            not self.out_format
            and isinstance(out_metric, tuple)
            and hasattr(out_metric, "_fields")
        ):
            self.out_format = namedtuple(  # type: ignore
                type(out_metric).__name__, cast(NamedTuple, out_metric)._fields
            )

    def _evaluate_batch(
        self,
        input_list: List[Any],
        # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
        additional_forward_args: Optional[Tuple],
        key_list: List[str],
        batch_summarizers: Dict[str, Summarizer],
        metric_kwargs: Dict[str, Any],
    ) -> None:
        if additional_forward_args is None:
            additional_forward_args = ()
        if len(input_list) == 1:
            model_out = self.forward_func(input_list[0], *additional_forward_args)
            out_metric = self.metric(model_out, **metric_kwargs)
            self._update_out_format(out_metric)
            batch_summarizers[key_list[0]].update(out_metric)
        else:
            batched_inps = _reduce_list(input_list)
            model_out = self.forward_func(batched_inps, *additional_forward_args)
            current_count = 0
            for i in range(len(input_list)):
                batch_size = (
                    input_list[i].shape[0]
                    if isinstance(input_list[i], Tensor)
                    else input_list[i][0].shape[0]
                )
                out_metric = self.metric(
                    model_out[current_count : current_count + batch_size],
                    **metric_kwargs,
                )
                self._update_out_format(out_metric)
                batch_summarizers[key_list[i]].update(out_metric)
                current_count += batch_size

    @log_usage(part_of_slo=False)
    def evaluate(
        self,
        inputs: Any,
        additional_forward_args: Optional[object] = None,
        perturbations_per_eval: int = 1,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> Dict[str, Union[MetricResultType, Dict[str, MetricResultType]]]:
        r"""
        Evaluate model and attack performance on provided inputs

        Args:

            inputs (Any): Input for which attack metrics
                are computed. It can be provided as a tensor, tuple of tensors,
                or any raw input type (e.g. PIL image or text string).
                This input is provided directly as input to preproc function as well
                as any attack applied before preprocessing. If no pre-processing
                function is provided, this input is provided directly to the main
                model and all attacks.

            additional_forward_args (Any, optional): If the forward function
                requires additional arguments other than the preprocessing
                outputs (or inputs if preproc_fn is None), this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a
                tuple containing multiple additional arguments including
                tensors or any arbitrary python types. These arguments
                are provided to forward_func in order following the
                arguments in inputs.
                For a tensor, the first dimension of the tensor must
                correspond to the number of examples. For all other types,
                the given argument is used for all forward evaluations.
                Default: ``None``
            perturbations_per_eval (int, optional): Allows perturbations of multiple
                attacks to be grouped and evaluated in one call of forward_fn
                Each forward pass will contain a maximum of
                perturbations_per_eval * #examples samples.
                For DataParallel models, each batch is split among the
                available devices, so evaluations on each available
                device contain at most
                (perturbations_per_eval * #examples) / num_devices
                samples.
                In order to apply this functionality, the output of preproc_fn
                (or inputs itself if no preproc_fn is provided) must be a tensor
                or tuple of tensors.
                Default: ``1``
            kwargs (Any, optional): Additional keyword arguments provided to metric
                function as well as selected attacks based on chosen additional_args.
                Default: ``None``

        Returns:

        - **attack results** Dict: str -> Dict[str, Union[Tensor, Tuple[Tensor, ...]]]:
                Dictionary containing attack results for provided batch.
                Maps attack name to dictionary,
                containing best-case, worst-case and average-case results for attack.
                Dictionary contains keys "mean", "max" and "min" when num_attempts > 1
                and only "mean" for num_attempts = 1, which contains the (single) metric
                result for the attack attempt.
                An additional key of 'Original' is included with metric results
                without any perturbations.


        Examples::

        >>> def accuracy_metric(model_out: Tensor, targets: Tensor):
        >>>     return torch.argmax(model_out, dim=1) == targets).float()

        >>> attack_metric = AttackComparator(model=resnet18,
                                             metric=accuracy_metric,
                                             preproc_fn=normalize)

        >>> random_rotation = transforms.RandomRotation()
        >>> jitter = transforms.ColorJitter()

        >>> attack_metric.add_attack(random_rotation, "Random Rotation",
        >>>                          num_attempts = 5)
        >>> attack_metric.add_attack((jitter, "Jitter", num_attempts = 1)
        >>> attack_metric.add_attack(FGSM(resnet18), "FGSM 0.1", num_attempts = 1,
        >>>                          apply_before_preproc=False,
        >>>                          attack_kwargs={epsilon: 0.1},
        >>>                          additional_args=["targets"])

        >>> for images, labels in dataloader:
        >>>     batch_results = attack_metric.evaluate(inputs=images, targets=labels)
        """
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        expanded_additional_args = (
            _expand_additional_forward_args(
                additional_forward_args, perturbations_per_eval
            )
            if perturbations_per_eval > 1
            else additional_forward_args
        )

        preproc_input = None
        if self.preproc_fn is not None:
            preproc_input = self.preproc_fn(inputs)
        else:
            preproc_input = inputs

        input_list = [preproc_input]
        key_list = [ORIGINAL_KEY]

        batch_summarizers = {ORIGINAL_KEY: Summarizer([Mean()])}
        if ORIGINAL_KEY not in self.summary_results:
            self.summary_results[ORIGINAL_KEY] = Summarizer(
                [stat() for stat in self.aggregate_stats]
            )

        # pyre-fixme[53]: Captured variable `batch_summarizers` is not annotated.
        # pyre-fixme[53]: Captured variable `expanded_additional_args` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def _check_and_evaluate(input_list, key_list):
            if len(input_list) == perturbations_per_eval:
                self._evaluate_batch(
                    input_list,
                    expanded_additional_args,
                    key_list,
                    batch_summarizers,
                    kwargs,
                )
                return [], []
            return input_list, key_list

        input_list, key_list = _check_and_evaluate(input_list, key_list)

        for attack_key in self.attacks:
            attack = self.attacks[attack_key]
            if attack.num_attempts > 1:
                stats = [stat() for stat in self.batch_stats]
            else:
                stats = [Mean()]
            batch_summarizers[attack.name] = Summarizer(stats)
            additional_attack_args = {}
            for key in attack.additional_args:
                if key not in kwargs:
                    warnings.warn(
                        f"Additional sample arg {key} not provided for {attack_key}",
                        stacklevel=1,
                    )
                else:
                    additional_attack_args[key] = kwargs[key]

            for _ in range(attack.num_attempts):
                if attack.apply_before_preproc:
                    attacked_inp = attack.attack_fn(
                        inputs, **additional_attack_args, **attack.attack_kwargs
                    )
                    preproc_attacked_inp = (
                        self.preproc_fn(attacked_inp)
                        if self.preproc_fn
                        else attacked_inp
                    )
                else:
                    preproc_attacked_inp = attack.attack_fn(
                        preproc_input, **additional_attack_args, **attack.attack_kwargs
                    )

                input_list.append(preproc_attacked_inp)
                key_list.append(attack.name)

                input_list, key_list = _check_and_evaluate(input_list, key_list)

        if len(input_list) > 0:
            final_add_args = _expand_additional_forward_args(
                additional_forward_args, len(input_list)
            )
            self._evaluate_batch(
                input_list, final_add_args, key_list, batch_summarizers, kwargs
            )

        return self._parse_and_update_results(batch_summarizers)

    def _parse_and_update_results(
        self, batch_summarizers: Dict[str, Summarizer]
    ) -> Dict[str, Union[MetricResultType, Dict[str, MetricResultType]]]:
        results: Dict[str, Union[MetricResultType, Dict[str, MetricResultType]]] = {
            ORIGINAL_KEY: self._format_summary(
                # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
                #  `typing.Dict[<key type>, <value type>]` to avoid runtime
                #  subscripting errors.
                # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
                #  `typing.List[<element type>]` to avoid runtime subscripting errors.
                cast(Union[Dict, List], batch_summarizers[ORIGINAL_KEY].summary)
            )["mean"]
        }
        self.summary_results[ORIGINAL_KEY].update(
            self.metric_aggregator(results[ORIGINAL_KEY])
        )
        for attack_key in self.attacks:
            attack = self.attacks[attack_key]
            attack_results = self._format_summary(
                # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
                #  `typing.Dict[<key type>, <value type>]` to avoid runtime
                #  subscripting errors.
                # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
                #  `typing.List[<element type>]` to avoid runtime subscripting errors.
                cast(Union[Dict, List], batch_summarizers[attack.name].summary)
            )
            results[attack.name] = attack_results

            if len(attack_results) == 1:
                key = next(iter(attack_results))
                if attack.name not in self.summary_results:
                    self.summary_results[attack.name] = Summarizer(
                        [stat() for stat in self.aggregate_stats]
                    )
                self.summary_results[attack.name].update(
                    self.metric_aggregator(attack_results[key])
                )
            else:
                for key in attack_results:
                    summary_key = f"{attack.name} {key.title()} Attempt"
                    if summary_key not in self.summary_results:
                        self.summary_results[summary_key] = Summarizer(
                            [stat() for stat in self.aggregate_stats]
                        )
                    self.summary_results[summary_key].update(
                        self.metric_aggregator(attack_results[key])
                    )
        return results

    def summary(self) -> Dict[str, Dict[str, MetricResultType]]:
        r"""
        Returns average results over all previous batches evaluated.

        Returns:

            - **summary** Dict: str -> Dict[str, Union[Tensor, Tuple[Tensor, ...]]]:
                Dictionary containing summarized average attack results.
                Maps attack name (with "Mean Attempt", "Max Attempt" and "Min Attempt"
                suffixes if num_attempts > 1) to dictionary containing a key of "mean"
                maintaining summarized results,
                which is the running mean of results over all batches
                since construction or previous reset call. Tensor metrics are averaged
                over dimension 0 for each batch, in order to aggregte metrics collected
                per batch.
        """
        return {
            key: self._format_summary(
                # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
                #  `typing.Dict[<key type>, <value type>]` to avoid runtime
                #  subscripting errors.
                # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
                #  `typing.List[<element type>]` to avoid runtime subscripting errors.
                cast(Union[Dict, List], self.summary_results[key].summary)
            )
            for key in self.summary_results
        }

    def reset(self) -> None:
        r"""
        Reset stored average summary results for previous batches
        """
        self.summary_results = {}
