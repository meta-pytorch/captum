#!/usr/bin/env python3

# pyre-strict
import typing
import warnings
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_baseline,
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
    _register_backward_hook,
    _run_forward,
    _select_targets,
    ExpansionTypes,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.common import (
    _call_custom_attribution_func,
    _compute_conv_delta_and_format_attrs,
    _format_callable_baseline,
    _tensorize_baseline,
    _validate_input,
)
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module
from torch.overrides import TorchFunctionMode
from torch.utils.hooks import RemovableHandle


def _can_apply_deeplift_tensor_rule(input: object) -> bool:
    # DeepLift runs one combined forward with actual inputs followed by
    # baselines along batch dimension 0. Functional-op rules are valid only
    # for tensors that participate in that paired actual / reference batch.
    return (
        isinstance(input, Tensor)
        and input.requires_grad
        and input.dim() > 0
        and input.shape[0] % 2 == 0
    )


def _can_apply_deeplift_binary_tensor_rule(input: object, other: object) -> bool:
    return (
        _can_apply_deeplift_tensor_rule(input)
        and _can_apply_deeplift_tensor_rule(other)
        and cast(Tensor, input).shape[0] == cast(Tensor, other).shape[0]
    )


def _is_square_power(power: object) -> bool:
    return isinstance(power, (int, float)) and power == 2


def _is_number_or_none(value: object) -> bool:
    return value is None or isinstance(value, (int, float))


def _forward_deeplift_unary_tensor_op(
    input: Tensor, op_name: str, options: Tuple[object, ...]
) -> Tensor:
    if op_name == "exp":
        return torch.exp(input)
    if op_name == "square":
        return input * input
    if op_name == "relu":
        return F.relu(input)
    if op_name == "leaky_relu":
        return F.leaky_relu(input, negative_slope=cast(float, options[0]))
    if op_name == "sigmoid":
        return torch.sigmoid(input)
    if op_name == "tanh":
        return torch.tanh(input)
    if op_name == "softplus":
        return F.softplus(
            input,
            beta=cast(float, options[0]),
            threshold=cast(float, options[1]),
        )
    if op_name == "elu":
        return F.elu(input, alpha=cast(float, options[0]))
    if op_name == "selu":
        return F.selu(input)
    if op_name == "gelu":
        return F.gelu(input, approximate=cast(str, options[0]))
    if op_name == "rsqrt":
        return torch.rsqrt(input)
    if op_name == "clamp":
        return torch.clamp(
            input,
            min=cast(Optional[float], options[0]),
            max=cast(Optional[float], options[1]),
        )
    raise AssertionError("Unsupported DeepLift tensor unary op.")


def _gradient_deeplift_unary_tensor_op(
    input: Tensor, output: Tensor, op_name: str, options: Tuple[object, ...]
) -> Tensor:
    if op_name == "exp":
        return output
    if op_name == "square":
        return 2 * input
    if op_name == "relu":
        return (input > 0).to(input.dtype)
    if op_name == "leaky_relu":
        negative_slope = cast(float, options[0])
        return torch.where(
            input > 0,
            torch.ones_like(input),
            torch.full_like(input, negative_slope),
        )
    if op_name == "sigmoid":
        return output * (1 - output)
    if op_name == "tanh":
        return 1 - output * output
    if op_name == "softplus":
        beta = cast(float, options[0])
        threshold = cast(float, options[1])
        return torch.where(
            input * beta > threshold,
            torch.ones_like(input),
            torch.sigmoid(input * beta),
        )
    if op_name == "elu":
        alpha = cast(float, options[0])
        return torch.where(input > 0, torch.ones_like(input), alpha * torch.exp(input))
    if op_name == "selu":
        scale = 1.0507009873554805
        alpha = 1.6732632423543772
        return torch.where(
            input > 0,
            torch.full_like(input, scale),
            scale * alpha * torch.exp(input),
        )
    if op_name == "gelu":
        approximate = cast(str, options[0])
        if approximate == "tanh":
            inner = 0.7978845608028654 * (input + 0.044715 * input.pow(3))
            tanh_inner = torch.tanh(inner)
            inner_grad = 0.7978845608028654 * (1 + 3 * 0.044715 * input.pow(2))
            return (
                0.5 * (1 + tanh_inner)
                + 0.5 * input * (1 - tanh_inner.pow(2)) * inner_grad
            )
        return (
            0.5 * (1 + torch.erf(input / 1.4142135623730951))
            + input * torch.exp(-0.5 * input.pow(2)) * 0.3989422804014327
        )
    if op_name == "rsqrt":
        return -0.5 * input.pow(-1.5)
    if op_name == "clamp":
        min_value = cast(Optional[float], options[0])
        max_value = cast(Optional[float], options[1])
        grad = torch.ones_like(input)
        if min_value is not None:
            grad = torch.where(input < min_value, torch.zeros_like(grad), grad)
        if max_value is not None:
            grad = torch.where(input > max_value, torch.zeros_like(grad), grad)
        return grad
    raise AssertionError("Unsupported DeepLift tensor unary op.")


def _run_deeplift_max_pool(
    input: Tensor,
    pool_dim: int,
    kernel_size: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    ceil_mode: bool,
) -> Tuple[Tensor, Tensor]:
    if pool_dim == 1:
        return F.max_pool1d(
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            return_indices=True,
        )
    if pool_dim == 2:
        return F.max_pool2d(
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            return_indices=True,
        )
    if pool_dim == 3:
        return F.max_pool3d(
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            return_indices=True,
        )
    raise AssertionError("Unsupported DeepLift tensor max-pool op.")


def _run_deeplift_max_unpool(
    input: Tensor,
    indices: Tensor,
    pool_dim: int,
    kernel_size: Any,
    stride: Any,
    padding: Any,
    output_size: Any,
) -> Tensor:
    if pool_dim == 1:
        return F.max_unpool1d(input, indices, kernel_size, stride, padding, output_size)
    if pool_dim == 2:
        return F.max_unpool2d(input, indices, kernel_size, stride, padding, output_size)
    if pool_dim == 3:
        return F.max_unpool3d(input, indices, kernel_size, stride, padding, output_size)
    raise AssertionError("Unsupported DeepLift tensor max-pool op.")


class _DeepLiftTensorUnaryOp(torch.autograd.Function):
    """
    DeepLift Rescale rule for functional one-input tensor ops.

    Module hooks do not see calls like ``torch.exp(x)`` or ``x.pow(2)``, so this
    autograd function replaces their backward multipliers with
    ``delta_out / delta_in`` while preserving the original forward value.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx: Any,
        input: Tensor,
        eps: float,
        op_name: str,
        options: Tuple[object, ...],
    ) -> Tensor:
        with torch._C.DisableTorchFunction():
            output = _forward_deeplift_unary_tensor_op(input, op_name, options)
        ctx.eps = eps
        ctx.op_name = op_name
        ctx.options = options
        ctx.save_for_backward(input.detach(), output.detach())
        return output

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        inputs, outputs = ctx.saved_tensors
        delta_in, delta_out = _compute_diffs(inputs, outputs)
        grad_input = grad_output * _gradient_deeplift_unary_tensor_op(
            inputs, outputs, ctx.op_name, ctx.options
        )
        new_grad_input = torch.where(
            abs(delta_in) < ctx.eps,
            grad_input,
            grad_output * delta_out / delta_in,
        )
        return new_grad_input, None, None, None


class _DeepLiftTensorBinaryOp(torch.autograd.Function):
    """
    DeepLift rule for functional two-input tensor ops where both inputs vary.

    For ops such as multiplication and division, plain gradients do not satisfy
    completeness when both operands differ from their baselines. This uses the
    same symmetric contribution split as SHAP's DeepExplainer op handlers.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx: Any, input: Tensor, other: Tensor, eps: float, op_name: str
    ) -> Tensor:
        with torch._C.DisableTorchFunction():
            if op_name == "mul":
                output = input * other
            elif op_name == "div":
                output = input / other
            elif op_name == "minimum":
                output = torch.minimum(input, other)
            elif op_name == "maximum":
                output = torch.maximum(input, other)
            else:
                raise AssertionError("Unsupported DeepLift tensor binary op.")
        ctx.eps = eps
        ctx.op_name = op_name
        ctx.input_shape = input.shape
        ctx.other_shape = other.shape
        ctx.save_for_backward(input.detach(), other.detach(), output.detach())
        return output

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Tensor, None, None]:
        input, other, output = ctx.saved_tensors
        input, other = torch.broadcast_tensors(input, other)
        if output.shape != input.shape:
            output = output.expand_as(input)
        if grad_output.shape != input.shape:
            grad_output = grad_output.expand_as(input)

        input_actual, input_ref = input.chunk(2)
        other_actual, other_ref = other.chunk(2)
        output_actual, output_ref = output.chunk(2)
        delta_input = input_actual - input_ref
        delta_other = other_actual - other_ref

        if ctx.op_name == "mul":
            cross_input_actual = input_actual * other_ref
            cross_other_actual = input_ref * other_actual
            grad_input = other
            grad_other = input
        elif ctx.op_name == "div":
            cross_input_actual = input_actual / other_ref
            cross_other_actual = input_ref / other_actual
            grad_input = 1 / other
            grad_other = -input / (other * other)
        elif ctx.op_name == "minimum":
            cross_input_actual = torch.minimum(input_actual, other_ref)
            cross_other_actual = torch.minimum(input_ref, other_actual)
            grad_input = (input <= other).to(input.dtype)
            grad_other = (other < input).to(other.dtype)
        elif ctx.op_name == "maximum":
            cross_input_actual = torch.maximum(input_actual, other_ref)
            cross_other_actual = torch.maximum(input_ref, other_actual)
            grad_input = (input >= other).to(input.dtype)
            grad_other = (other > input).to(other.dtype)
        else:
            raise AssertionError("Unsupported DeepLift tensor binary op.")

        input_contrib = cast(
            Tensor,
            0.5
            * (output_actual - cross_other_actual + cross_input_actual - output_ref),
        )
        other_contrib = cast(
            Tensor,
            0.5
            * (output_actual - cross_input_actual + cross_other_actual - output_ref),
        )

        delta_input = torch.cat([delta_input, delta_input])
        delta_other = torch.cat([delta_other, delta_other])
        input_contrib = torch.cat([input_contrib, input_contrib])
        other_contrib = torch.cat([other_contrib, other_contrib])

        input_multiplier = torch.where(
            abs(delta_input) < ctx.eps,
            grad_input,
            input_contrib / delta_input,
        )
        other_multiplier = torch.where(
            abs(delta_other) < ctx.eps,
            grad_other,
            other_contrib / delta_other,
        )
        input_grad = (grad_output * input_multiplier).sum_to_size(ctx.input_shape)
        other_grad = (grad_output * other_multiplier).sum_to_size(ctx.other_shape)
        return input_grad, other_grad, None, None


class _DeepLiftTensorSingleInputBinaryOp(torch.autograd.Function):
    """
    DeepLift rule for binary nonlinear ops where only one tensor input varies.

    Elementwise min/max with a constant threshold is nonlinear in the varying
    input, so the plain PyTorch gradient can violate completeness across a
    reference crossing. This applies the same Rescale multiplier used for unary
    nonlinearities to the varying side of the binary op.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx: Any,
        input: Tensor,
        other: Tensor,
        eps: float,
        op_name: str,
        input_is_left: bool,
    ) -> Tensor:
        with torch._C.DisableTorchFunction():
            if op_name == "minimum":
                output = torch.minimum(input, other)
            elif op_name == "maximum":
                output = torch.maximum(input, other)
            else:
                raise AssertionError("Unsupported DeepLift tensor binary op.")
        ctx.eps = eps
        ctx.op_name = op_name
        ctx.input_shape = input.shape
        ctx.input_is_left = input_is_left
        ctx.save_for_backward(input.detach(), other.detach(), output.detach())
        return output

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tuple[Tensor, None, None, None, None]:
        input, other, output = ctx.saved_tensors
        input, other = torch.broadcast_tensors(input, other)
        if output.shape != input.shape:
            output = output.expand_as(input)
        if grad_output.shape != input.shape:
            grad_output = grad_output.expand_as(input)

        if ctx.op_name == "minimum":
            grad_input = (input <= other if ctx.input_is_left else input < other).to(
                input.dtype
            )
        elif ctx.op_name == "maximum":
            grad_input = (input >= other if ctx.input_is_left else input > other).to(
                input.dtype
            )
        else:
            raise AssertionError("Unsupported DeepLift tensor binary op.")

        delta_in, delta_out = _compute_diffs(input, output)
        new_grad_input = torch.where(
            abs(delta_in) < ctx.eps,
            grad_output * grad_input,
            grad_output * delta_out / delta_in,
        )
        return new_grad_input.sum_to_size(ctx.input_shape), None, None, None, None


class _DeepLiftTensorMatmulOp(torch.autograd.Function):
    """
    DeepLift rule for matrix multiplication when both operands vary.

    This is the matmul analogue of the symmetric two-input rule: each operand
    receives the linear multiplier induced by the average of the other operand's
    actual and reference values.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(ctx: Any, input: Tensor, other: Tensor, eps: float) -> Tensor:
        with torch._C.DisableTorchFunction():
            output = torch.matmul(input, other)
        ctx.input_shape = input.shape
        ctx.other_shape = other.shape
        ctx.save_for_backward(input.detach(), other.detach())
        return output

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Tensor, None]:
        input, other = ctx.saved_tensors
        input_actual, input_ref = input.chunk(2)
        other_actual, other_ref = other.chunk(2)
        average_input_half = cast(Tensor, 0.5 * (input_actual + input_ref))
        average_other_half = cast(Tensor, 0.5 * (other_actual + other_ref))
        average_input = torch.cat([average_input_half, average_input_half])
        average_other = torch.cat([average_other_half, average_other_half])
        input_grad = torch.matmul(
            grad_output, average_other.transpose(-2, -1)
        ).sum_to_size(ctx.input_shape)
        other_grad = torch.matmul(
            average_input.transpose(-2, -1), grad_output
        ).sum_to_size(ctx.other_shape)
        return input_grad, other_grad, None


class _DeepLiftTensorMaxReductionOp(torch.autograd.Function):
    """
    DeepLift rule for max reductions over a non-batch dimension.

    This mirrors the max-pool rule: compare actual/reference max outputs,
    route the resulting output differences to the winning input positions, and
    divide by the input differences to obtain multipliers.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(ctx: Any, input: Tensor, dim: int, keepdim: bool, eps: float) -> Tensor:
        with torch._C.DisableTorchFunction():
            output, indices = torch.max(input, dim=dim, keepdim=True)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.eps = eps
        ctx.save_for_backward(input.detach(), output.detach(), indices.detach())
        return output if keepdim else output.squeeze(dim)

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        input, output, indices = ctx.saved_tensors
        dim = ctx.dim
        grad_output = grad_output if ctx.keepdim else grad_output.unsqueeze(dim)

        input_actual, input_ref = input.chunk(2)
        output_actual, output_ref = output.chunk(2)
        cross_max = torch.maximum(output_actual, output_ref)
        output_diffs = torch.cat([cross_max - output_ref, output_actual - cross_max])
        max_positions = torch.zeros_like(input).scatter_add(
            dim, indices, grad_output * output_diffs
        )
        max_positions_actual, max_positions_ref = max_positions.chunk(2)
        max_positions = torch.cat(2 * [max_positions_actual + max_positions_ref])

        delta_input = torch.cat(2 * [input_actual - input_ref])
        grad_input = torch.zeros_like(input).scatter_add(dim, indices, grad_output)
        new_grad_input = torch.where(
            abs(delta_input) < ctx.eps,
            grad_input,
            max_positions / delta_input,
        )
        return new_grad_input, None, None, None


class _DeepLiftTensorMaxPoolOp(torch.autograd.Function):
    """
    DeepLift rule for functional max-pool operations.

    ``nn.MaxPool*`` modules already use the same routing rule through module
    hooks. Functional max-pool calls need a private autograd wrapper because
    they do not create a module boundary for Captum to hook.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx: Any,
        input: Tensor,
        eps: float,
        pool_dim: int,
        kernel_size: object,
        stride: object,
        padding: object,
        dilation: object,
        ceil_mode: bool,
        return_indices: bool,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        with torch._C.DisableTorchFunction():
            output, indices = _run_deeplift_max_pool(
                input,
                pool_dim,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            )
        ctx.eps = eps
        ctx.pool_dim = pool_dim
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.input_shape = input.shape
        ctx.return_indices = return_indices
        ctx.save_for_backward(input.detach(), output.detach(), indices.detach())
        if return_indices:
            ctx.mark_non_differentiable(indices)
            return output, indices
        return output

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    def backward(
        ctx: Any, grad_output: Tensor, grad_indices: Optional[Tensor] = None
    ) -> Tuple[
        Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        input, output, indices = ctx.saved_tensors
        with torch.no_grad():
            input_actual, input_ref = input.chunk(2)
            output_actual, output_ref = output.chunk(2)
            delta_input = torch.cat(2 * [input_actual - input_ref])
            cross_max = torch.maximum(output_actual, output_ref)
            output_diffs = torch.cat(
                [cross_max - output_ref, output_actual - cross_max]
            )
            output_size = list(cast(torch.Size, ctx.input_shape))
            unpool_delta, unpool_ref_delta = torch.chunk(
                _run_deeplift_max_unpool(
                    grad_output * output_diffs,
                    indices,
                    ctx.pool_dim,
                    ctx.kernel_size,
                    ctx.stride,
                    ctx.padding,
                    output_size,
                ),
                2,
            )
            unpool_delta = torch.cat(2 * [unpool_delta + unpool_ref_delta])
            grad_input = _run_deeplift_max_unpool(
                grad_output,
                indices,
                ctx.pool_dim,
                ctx.kernel_size,
                ctx.stride,
                ctx.padding,
                output_size,
            )

        new_grad_input = torch.where(
            abs(delta_input) < ctx.eps,
            grad_input,
            unpool_delta / delta_input,
        )
        return new_grad_input, None, None, None, None, None, None, None, None


class _DeepLiftTensorSoftmaxShift(torch.autograd.Function):
    """
    Rescale the numerically stable softmax shift back to the original input.

    Functional softmax is decomposed into shifted input, exp, sum, and division
    so the existing unary / binary rules can handle it. This op accounts for
    the data-dependent max subtraction introduced for stability.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(ctx: Any, input: Tensor, dim: int, eps: float) -> Tensor:
        with torch._C.DisableTorchFunction():
            output = input - torch.max(input, dim=dim, keepdim=True).values
        ctx.eps = eps
        ctx.save_for_backward(input.detach(), output.detach())
        return output

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        inputs, outputs = ctx.saved_tensors
        delta_in, delta_out = _compute_diffs(inputs, outputs)
        new_grad_input = torch.where(
            abs(delta_in) < ctx.eps,
            grad_output,
            grad_output * delta_out / delta_in,
        )
        return new_grad_input, None, None


class _DeepLiftTensorOpMode(TorchFunctionMode):
    """
    Intercepts selected functional tensor ops during DeepLift's model forward.

    Captum's historical DeepLift support is module-hook based. This mode adds
    equivalent private handling for common functional ops that have no
    ``nn.Module`` boundary, while leaving unrelated tensor operations unchanged.
    """

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def __torch_function__(
        self,
        func: Callable[..., object],
        types: Tuple[Type[object], ...],
        args: Tuple[object, ...] = (),
        kwargs: Optional[Dict[str, object]] = None,
    ) -> object:
        kwargs = kwargs or {}
        func_name = getattr(func, "__name__", None)

        if (
            func_name == "exp"
            and len(args) > 0
            and _can_apply_deeplift_tensor_rule(args[0])
        ):
            return _DeepLiftTensorUnaryOp.apply(args[0], self.eps, "exp", ())

        if (
            func_name == "square"
            and len(args) > 0
            and _can_apply_deeplift_tensor_rule(args[0])
        ):
            return _DeepLiftTensorUnaryOp.apply(args[0], self.eps, "square", ())

        if (
            func_name == "pow"
            and len(args) > 1
            and _can_apply_deeplift_tensor_rule(args[0])
            and _is_square_power(args[1])
        ):
            return _DeepLiftTensorUnaryOp.apply(args[0], self.eps, "square", ())

        if (
            func_name
            in (
                "relu",
                "sigmoid",
                "tanh",
                "selu",
                "rsqrt",
            )
            and len(args) > 0
            and _can_apply_deeplift_tensor_rule(args[0])
        ):
            return _DeepLiftTensorUnaryOp.apply(args[0], self.eps, func_name, ())

        if (
            func_name == "leaky_relu"
            and len(args) > 0
            and _can_apply_deeplift_tensor_rule(args[0])
        ):
            negative_slope = kwargs.get(
                "negative_slope", args[1] if len(args) > 1 else 0.01
            )
            if isinstance(negative_slope, (int, float)):
                return _DeepLiftTensorUnaryOp.apply(
                    args[0], self.eps, "leaky_relu", (float(negative_slope),)
                )

        if (
            func_name == "softplus"
            and len(args) > 0
            and _can_apply_deeplift_tensor_rule(args[0])
        ):
            beta = kwargs.get("beta", args[1] if len(args) > 1 else 1)
            threshold = kwargs.get("threshold", args[2] if len(args) > 2 else 20)
            if isinstance(beta, (int, float)) and isinstance(threshold, (int, float)):
                return _DeepLiftTensorUnaryOp.apply(
                    args[0],
                    self.eps,
                    "softplus",
                    (float(beta), float(threshold)),
                )

        if (
            func_name == "elu"
            and len(args) > 0
            and _can_apply_deeplift_tensor_rule(args[0])
        ):
            alpha = kwargs.get("alpha", args[1] if len(args) > 1 else 1)
            if isinstance(alpha, (int, float)):
                return _DeepLiftTensorUnaryOp.apply(
                    args[0], self.eps, "elu", (float(alpha),)
                )

        if (
            func_name == "gelu"
            and len(args) > 0
            and _can_apply_deeplift_tensor_rule(args[0])
        ):
            approximate = kwargs.get("approximate", "none")
            if isinstance(approximate, str):
                return _DeepLiftTensorUnaryOp.apply(
                    args[0], self.eps, "gelu", (approximate,)
                )

        if (
            func_name in ("clamp", "clip")
            and len(args) > 0
            and _can_apply_deeplift_tensor_rule(args[0])
        ):
            min_value = kwargs.get("min", args[1] if len(args) > 1 else None)
            max_value = kwargs.get("max", args[2] if len(args) > 2 else None)
            if _is_number_or_none(min_value) and _is_number_or_none(max_value):
                return _DeepLiftTensorUnaryOp.apply(
                    args[0],
                    self.eps,
                    "clamp",
                    (
                        None if min_value is None else float(cast(float, min_value)),
                        None if max_value is None else float(cast(float, max_value)),
                    ),
                )

        if (
            func_name in ("mul", "div", "truediv")
            and len(args) > 1
            and _can_apply_deeplift_binary_tensor_rule(args[0], args[1])
        ):
            return _DeepLiftTensorBinaryOp.apply(
                args[0],
                args[1],
                self.eps,
                "mul" if func_name == "mul" else "div",
            )

        if (
            func_name in ("minimum", "maximum", "min", "max")
            and len(args) > 1
            and _can_apply_deeplift_binary_tensor_rule(args[0], args[1])
        ):
            return _DeepLiftTensorBinaryOp.apply(
                args[0],
                args[1],
                self.eps,
                "minimum" if func_name in ("minimum", "min") else "maximum",
            )

        if func_name in ("minimum", "maximum", "min", "max") and len(args) > 1:
            op_name = "minimum" if func_name in ("minimum", "min") else "maximum"
            if (
                _can_apply_deeplift_tensor_rule(args[0])
                and isinstance(args[1], Tensor)
                and not _can_apply_deeplift_tensor_rule(args[1])
            ):
                return _DeepLiftTensorSingleInputBinaryOp.apply(
                    args[0], args[1], self.eps, op_name, True
                )
            if (
                isinstance(args[0], Tensor)
                and not _can_apply_deeplift_tensor_rule(args[0])
                and _can_apply_deeplift_tensor_rule(args[1])
            ):
                return _DeepLiftTensorSingleInputBinaryOp.apply(
                    args[1], args[0], self.eps, op_name, False
                )

        if (
            func_name in ("matmul", "mm", "bmm")
            and len(args) > 1
            and _can_apply_deeplift_binary_tensor_rule(args[0], args[1])
            and cast(Tensor, args[0]).dim() >= 2
            and cast(Tensor, args[1]).dim() >= 2
        ):
            return _DeepLiftTensorMatmulOp.apply(args[0], args[1], self.eps)

        if (
            func_name == "amax"
            and len(args) > 0
            and _can_apply_deeplift_tensor_rule(args[0])
        ):
            dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
            keepdim = kwargs.get("keepdim", args[2] if len(args) > 2 else False)
            if isinstance(dim, int) and isinstance(keepdim, bool):
                input_dim = cast(Tensor, args[0]).dim()
                dim = dim if dim >= 0 else input_dim + dim
                if dim != 0:
                    return _DeepLiftTensorMaxReductionOp.apply(
                        args[0], dim, keepdim, self.eps
                    )

        if (
            func_name == "max"
            and len(args) > 0
            and _can_apply_deeplift_tensor_rule(args[0])
        ):
            dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
            keepdim = kwargs.get("keepdim", args[2] if len(args) > 2 else False)
            if isinstance(dim, int) and isinstance(keepdim, bool):
                input = cast(Tensor, args[0])
                input_dim = input.dim()
                dim = dim if dim >= 0 else input_dim + dim
                if dim != 0:
                    values = _DeepLiftTensorMaxReductionOp.apply(
                        input, dim, keepdim, self.eps
                    )
                    with torch._C.DisableTorchFunction():
                        _, indices = torch.max(input, dim=dim, keepdim=keepdim)
                    # pyre-fixme[16]: `torch.return_types` has no attribute `max`.
                    return torch.return_types.max((values, indices))

        if (
            func_name
            in (
                "max_pool1d",
                "max_pool1d_with_indices",
                "max_pool2d",
                "max_pool2d_with_indices",
                "max_pool3d",
                "max_pool3d_with_indices",
            )
            and len(args) > 0
            and _can_apply_deeplift_tensor_rule(args[0])
        ):
            pool_dim = 1 if "1d" in func_name else 2 if "2d" in func_name else 3
            kernel_size = args[1] if len(args) > 1 else kwargs.get("kernel_size")
            stride = kwargs.get("stride", args[2] if len(args) > 2 else None)
            padding = kwargs.get("padding", args[3] if len(args) > 3 else 0)
            dilation = kwargs.get("dilation", args[4] if len(args) > 4 else 1)
            ceil_mode = kwargs.get("ceil_mode", args[5] if len(args) > 5 else False)
            return_indices = kwargs.get(
                "return_indices",
                args[6] if len(args) > 6 else func_name.endswith("_with_indices"),
            )
            if (
                kernel_size is not None
                and isinstance(ceil_mode, bool)
                and isinstance(return_indices, bool)
            ):
                return _DeepLiftTensorMaxPoolOp.apply(
                    args[0],
                    self.eps,
                    pool_dim,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    return_indices,
                )

        if (
            func_name == "softmax"
            and len(args) > 0
            and _can_apply_deeplift_tensor_rule(args[0])
        ):
            dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
            if isinstance(dim, int):
                input = cast(Tensor, args[0])
                input_dim = input.dim()
                normalized_dim = dim if dim >= 0 else input_dim + dim
                if normalized_dim != 0:
                    dtype = kwargs.get("dtype")
                    if isinstance(dtype, torch.dtype):
                        input = input.to(dtype=dtype)
                    shifted_input = _DeepLiftTensorSoftmaxShift.apply(
                        input, normalized_dim, self.eps
                    )
                    exp_input = _DeepLiftTensorUnaryOp.apply(
                        shifted_input, self.eps, "exp", ()
                    )
                    exp_sum = exp_input.sum(dim=normalized_dim, keepdim=True)
                    return _DeepLiftTensorBinaryOp.apply(
                        exp_input, exp_sum, self.eps, "div"
                    )

        return func(*args, **kwargs)


class DeepLift(GradientAttribution):
    r"""
    Implements DeepLIFT algorithm based on the following paper:
    Learning Important Features Through Propagating Activation Differences,
    Avanti Shrikumar, et. al.
    https://arxiv.org/abs/1704.02685

    and the gradient formulation proposed in:
    Towards better understanding of gradient-based attribution methods for
    deep neural networks,  Marco Ancona, et.al.
    https://openreview.net/pdf?id=Sy21R9JAW

    This implementation supports only Rescale rule. RevealCancel rule will
    be supported in later releases.
    In addition to that, in order to keep the implementation cleaner, DeepLIFT
    for internal neurons and layers extends current implementation and is
    implemented separately in LayerDeepLift and NeuronDeepLift.
    Although DeepLIFT's(Rescale Rule) attribution quality is comparable with
    Integrated Gradients, it runs significantly faster than Integrated
    Gradients and is preferred for large datasets.

    Currently we only support a limited number of non-linear activations
    but the plan is to expand the list in the future.

    Note: As we know, currently we cannot access the building blocks,
    of PyTorch's built-in LSTM, RNNs and GRUs such as Tanh and Sigmoid.
    Nonetheless, it is possible to build custom LSTMs, RNNS and GRUs
    with performance similar to built-in ones using TorchScript.
    More details on how to build custom RNNs can be found here:
    https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
    """

    def __init__(
        self,
        model: Module,
        multiply_by_inputs: bool = True,
        eps: float = 1e-10,
    ) -> None:
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
            multiply_by_inputs (bool, optional): Indicates whether to factor
                        model inputs' multiplier in the final attribution scores.
                        In the literature this is also known as local vs global
                        attribution. If inputs' multiplier isn't factored in
                        then that type of attribution method is also called local
                        attribution. If it is, then that type of attribution
                        method is called global.
                        More detailed can be found here:
                        https://arxiv.org/abs/1711.06104

                        In case of DeepLift, if `multiply_by_inputs`
                        is set to True, final sensitivity scores
                        are being multiplied by (inputs - baselines).
                        This flag applies only if `custom_attribution_func` is
                        set to None.

            eps (float, optional): A value at which to consider output/input change
                        significant when computing the gradients for non-linear layers.
                        This is useful to adjust, depending on your model's bit depth,
                        to avoid numerical issues during the gradient computation.
                        Default: 1e-10
        """
        GradientAttribution.__init__(self, model)
        self.model: nn.Module = model
        self.eps = eps
        self.forward_handles: List[RemovableHandle] = []
        self.backward_handles: List[RemovableHandle] = []
        self._multiply_by_inputs = multiply_by_inputs

    @typing.overload
    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        *,
        return_convergence_delta: Literal[True],
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]: ...

    @typing.overload
    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        return_convergence_delta: Literal[False] = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> TensorOrTupleOfTensorsGeneric: ...

    @log_usage(part_of_slo=True)
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        return_convergence_delta: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which
                        attributions are computed. If model takes a single
                        tensor as input, a single input tensor should be provided.
                        If model takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores DeepLift
                        computes the differences between the inputs/outputs and
                        corresponding references.
                        Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:

                          - either a tensor with matching dimensions to
                            corresponding tensor in the inputs' tuple
                            or the first dimension is one and the remaining
                            dimensions match with the corresponding
                            input tensor.

                          - or a scalar, corresponding to a tensor in the
                            inputs' tuple. This scalar value is broadcasted
                            for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        model in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False
            custom_attribution_func (Callable, optional): A custom function for
                        computing final attribution scores. This function can take
                        at least one and at most three arguments with the
                        following signature:

                        - custom_attribution_func(multipliers)
                        - custom_attribution_func(multipliers, inputs)
                        - custom_attribution_func(multipliers, inputs, baselines)

                        In case this function is not provided, we use the default
                        logic defined as: multipliers * (inputs - baselines)
                        It is assumed that all input arguments, `multipliers`,
                        `inputs` and `baselines` are provided in tuples of same
                        length. `custom_attribution_func` returns a tuple of
                        attribution tensors that have the same length as the
                        `inputs`.

                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                Attribution score computed based on DeepLift rescale rule with respect
                to each input feature. Attributions will always be
                the same size as the provided inputs, with each value
                providing the attribution of the corresponding input index.
                If a single tensor is provided as inputs, a single tensor is
                returned. If a tuple is provided for inputs, a tuple of
                corresponding sized tensors is returned.
            - **delta** (*Tensor*, returned if return_convergence_delta=True):
                This is computed using the property that
                the total sum of model(inputs) - model(baselines)
                must equal the total sum of the attributions computed
                based on DeepLift's rescale rule.
                Delta is calculated per example, meaning that the number of
                elements in returned delta tensor is equal to the number of
                examples in input.
                Note that the logic described for deltas is guaranteed when the
                default logic for attribution computations is used, meaning that the
                `custom_attribution_func=None`, otherwise it is not guaranteed and
                depends on the specifics of the `custom_attribution_func`.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> dl = DeepLift(net)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes deeplift attribution scores for class 3.
            >>> attribution = dl.attribute(input, target=3)
        """

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs_tuple = _format_tensor_into_tuples(inputs)
        baselines = _format_baseline(baselines, inputs_tuple)

        gradient_mask = apply_gradient_requirements(inputs_tuple)

        _validate_input(inputs_tuple, baselines)

        # set hooks for baselines
        warnings.warn(
            """Setting forward, backward hooks and attributes on non-linear
               activations. The hooks and attributes will be removed
            after the attribution is finished""",
            stacklevel=2,
        )
        baselines = _tensorize_baseline(inputs_tuple, baselines)
        main_model_hooks = []
        try:
            main_model_hooks = self._hook_main_model()

            self.model.apply(self._register_hooks)

            additional_forward_args = _format_additional_forward_args(
                additional_forward_args
            )

            expanded_target = _expand_target(
                target, 2, expansion_type=ExpansionTypes.repeat
            )

            wrapped_forward_func = self._construct_forward_func(
                self.model,
                (inputs_tuple, baselines),
                expanded_target,
                additional_forward_args,
            )
            gradients = self.gradient_func(wrapped_forward_func, inputs_tuple)
            if custom_attribution_func is None:
                if self.multiplies_by_inputs:
                    attributions = tuple(
                        (input - baseline) * gradient
                        for input, baseline, gradient in zip(
                            inputs_tuple, baselines, gradients
                        )
                    )
                else:
                    attributions = gradients
            else:
                attributions = _call_custom_attribution_func(
                    custom_attribution_func,
                    gradients,
                    inputs_tuple,
                    baselines,
                )
        finally:
            # Even if any error is raised, remove all hooks before raising
            self._remove_hooks(main_model_hooks)

        undo_gradient_requirements(inputs_tuple, gradient_mask)
        return cast(
            TensorOrTupleOfTensorsGeneric,
            _compute_conv_delta_and_format_attrs(
                self,
                return_convergence_delta,
                attributions,
                baselines,
                inputs_tuple,
                additional_forward_args,
                target,
                is_inputs_tuple,
            ),
        )

    def attribute_future(self) -> None:
        r"""
        This method is not implemented for DeepLift.
        """
        raise NotImplementedError("attribute_future is not implemented for DeepLift")

    def _construct_forward_func(
        self,
        forward_func: Callable[..., Tensor],
        inputs: Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
    ) -> Callable[[], Tensor]:
        def forward_fn() -> Tensor:
            with _DeepLiftTensorOpMode(self.eps):
                model_out = cast(
                    Tensor,
                    _run_forward(forward_func, inputs, None, additional_forward_args),
                )
            return _select_targets(
                torch.cat((model_out[:, 0], model_out[:, 1])),
                target,
            )

        if hasattr(forward_func, "device_ids"):
            forward_fn.device_ids = forward_func.device_ids  # type: ignore
        return forward_fn

    def _is_non_linear(self, module: Module) -> bool:
        return type(module) in SUPPORTED_NON_LINEAR.keys()

    def _forward_pre_hook_ref(
        self, module: Module, inputs: Union[Tensor, Tuple[Tensor, ...]]
    ) -> None:
        inputs = _format_tensor_into_tuples(inputs)
        module.input_ref = tuple(  # type: ignore
            input.clone().detach() for input in inputs
        )

    def _forward_pre_hook(
        self, module: Module, inputs: Union[Tensor, Tuple[Tensor, ...]]
    ) -> None:
        """
        For the modules that perform in-place operations such as ReLUs, we cannot
        use inputs from forward hooks. This is because in that case inputs
        and outputs are the same. We need access the inputs in pre-hooks and
        set necessary hooks on inputs there.
        """
        inputs = _format_tensor_into_tuples(inputs)
        module.input = inputs[0].clone().detach()  # type: ignore

    def _forward_hook(
        self,
        module: Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        outputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        r"""
        we need forward hook to access and detach the inputs and
        outputs of a neuron
        """
        outputs = _format_tensor_into_tuples(outputs)
        module.output = outputs[0].clone().detach()  # type: ignore

    def _backward_hook(
        self,
        module: Module,
        grad_input: Tensor,
        grad_output: Tensor,
    ) -> Tensor:
        r"""
        `grad_input` is the gradient of the neuron with respect to its input
        `grad_output` is the gradient of the neuron with respect to its output
         we can override `grad_input` according to chain rule with.
        `grad_output` * delta_out / delta_in.

        """
        # before accessing the attributes from the module we want
        # to ensure that the properties exist, if not, then it is
        # likely that the module is being reused.
        attr_criteria = self.satisfies_attribute_criteria(module)
        if not attr_criteria:
            raise RuntimeError(
                "A Module {} was detected that does not contain some of "
                "the input/output attributes that are required for DeepLift "
                "computations. This can occur, for example, if "
                "your module is being used more than once in the network."
                "Please, ensure that module is being used only once in the "
                "network.".format(module)
            )

        multipliers = SUPPORTED_NON_LINEAR[type(module)](
            module,
            module.input,
            module.output,
            grad_input,
            grad_output,
            eps=self.eps,
        )
        # remove all the properies that we set for the inputs and output
        del module.input
        del module.output

        return multipliers

    def satisfies_attribute_criteria(self, module: Module) -> bool:
        return hasattr(module, "input") and hasattr(module, "output")

    def _can_register_hook(self, module: Module) -> bool:
        # TODO find a better way of checking if a module is a container or not
        module_fullname = str(type(module))
        has_already_hooks = len(module._backward_hooks) > 0  # type: ignore
        return not (
            "nn.modules.container" in module_fullname
            or has_already_hooks
            or not self._is_non_linear(module)
        )

    def _register_hooks(
        self, module: Module, attribute_to_layer_input: bool = True
    ) -> None:
        if not self._can_register_hook(module) or (
            not attribute_to_layer_input and module is self.layer  # type: ignore
        ):
            return
        # adds forward hook to leaf nodes that are non-linear
        forward_handle = module.register_forward_hook(self._forward_hook)
        pre_forward_handle = module.register_forward_pre_hook(self._forward_pre_hook)
        backward_handles = _register_backward_hook(module, self._backward_hook, self)
        self.forward_handles.append(forward_handle)
        self.forward_handles.append(pre_forward_handle)
        self.backward_handles.extend(backward_handles)

    def _remove_hooks(self, extra_hooks_to_remove: List[RemovableHandle]) -> None:
        for handle in extra_hooks_to_remove:
            handle.remove()
        for forward_handle in self.forward_handles:
            forward_handle.remove()
        for backward_handle in self.backward_handles:
            backward_handle.remove()

    def _hook_main_model(self) -> List[RemovableHandle]:
        def pre_hook(
            module: Module,
            baseline_inputs_add_args: Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
        ) -> Tuple[object, ...]:
            inputs = baseline_inputs_add_args[0]
            baselines = baseline_inputs_add_args[1]
            additional_args = None
            if len(baseline_inputs_add_args) > 2:
                additional_args = baseline_inputs_add_args[2:]

            baseline_input_tsr = tuple(
                torch.cat([input, baseline])
                for input, baseline in zip(inputs, baselines)
            )
            if additional_args is not None:
                expanded_additional_args = cast(
                    Tuple[object],
                    _expand_additional_forward_args(
                        additional_args, 2, ExpansionTypes.repeat
                    ),
                )
                return (*baseline_input_tsr, *expanded_additional_args)
            return baseline_input_tsr

        def forward_hook(
            module: Module, inputs: Tuple[Tensor, ...], outputs: Tensor
        ) -> Tensor:
            return torch.stack(torch.chunk(outputs, 2), dim=1)

        if isinstance(
            self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
        ):
            return [
                self.model.module.register_forward_pre_hook(pre_hook),  # type: ignore
                self.model.module.register_forward_hook(forward_hook),  # type: ignore
            ]  # type: ignore
        else:
            return [
                self.model.register_forward_pre_hook(pre_hook),  # type: ignore
                self.model.register_forward_hook(forward_hook),
            ]  # type: ignore

    def has_convergence_delta(self) -> bool:
        return True

    @property
    def multiplies_by_inputs(self) -> bool:
        return self._multiply_by_inputs


class DeepLiftShap(DeepLift):
    r"""
    Extends DeepLift algorithm and approximates SHAP values using Deeplift.
    For each input sample it computes DeepLift attribution with respect to
    each baseline and averages resulting attributions.
    More details about the algorithm can be found here:

    https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf

    Note that the explanation model:

        1. Assumes that input features are independent of one another
        2. Is linear, meaning that the explanations are modeled through
            the additive composition of feature effects.

    Although, it assumes a linear model for each explanation, the overall
    model across multiple explanations can be complex and non-linear.
    """

    def __init__(self, model: Module, multiply_by_inputs: bool = True) -> None:
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
            multiply_by_inputs (bool, optional): Indicates whether to factor
                        model inputs' multiplier in the final attribution scores.
                        In the literature this is also known as local vs global
                        attribution. If inputs' multiplier isn't factored in
                        then that type of attribution method is also called local
                        attribution. If it is, then that type of attribution
                        method is called global.
                        More detailed can be found here:
                        https://arxiv.org/abs/1711.06104

                        In case of DeepLiftShap, if `multiply_by_inputs`
                        is set to True, final sensitivity scores
                        are being multiplied by (inputs - baselines).
                        This flag applies only if `custom_attribution_func` is
                        set to None.
        """
        DeepLift.__init__(self, model, multiply_by_inputs=multiply_by_inputs)

    # There's a mismatch between the signatures of DeepLift.attribute and
    # DeepLiftShap.attribute, so we ignore typing here
    @typing.overload  # type: ignore
    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: Union[
            TensorOrTupleOfTensorsGeneric, Callable[..., TensorOrTupleOfTensorsGeneric]
        ],
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        *,
        return_convergence_delta: Literal[True],
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]: ...

    @typing.overload
    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: Union[
            TensorOrTupleOfTensorsGeneric, Callable[..., TensorOrTupleOfTensorsGeneric]
        ],
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        return_convergence_delta: Literal[False] = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> TensorOrTupleOfTensorsGeneric: ...

    @log_usage(part_of_slo=True)
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: Union[
            TensorOrTupleOfTensorsGeneric, Callable[..., TensorOrTupleOfTensorsGeneric]
        ],
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        return_convergence_delta: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which
                        attributions are computed. If model takes a single
                        tensor as input, a single input tensor should be provided.
                        If model takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.
            baselines (Tensor, tuple[Tensor, ...], or Callable):
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores DeepLift
                        computes the differences between the inputs/outputs and
                        corresponding references. Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          the first dimension equal to the number of examples
                          in the baselines' distribution. The remaining dimensions
                          must match with input tensor's dimension starting from
                          the second dimension.

                        - a tuple of tensors, if inputs is a tuple of tensors,
                          with the first dimension of any tensor inside the tuple
                          equal to the number of examples in the baseline's
                          distribution. The remaining dimensions must match
                          the dimensions of the corresponding input tensor
                          starting from the second dimension.

                        - callable function, optionally takes `inputs` as an
                          argument and either returns a single tensor
                          or a tuple of those.

                        It is recommended that the number of samples in the baselines'
                        tensors is larger than one.
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        model in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False
            custom_attribution_func (Callable, optional): A custom function for
                        computing final attribution scores. This function can take
                        at least one and at most three arguments with the
                        following signature:

                        - custom_attribution_func(multipliers)
                        - custom_attribution_func(multipliers, inputs)
                        - custom_attribution_func(multipliers, inputs, baselines)

                        In case this function is not provided we use the default
                        logic defined as: multipliers * (inputs - baselines)
                        It is assumed that all input arguments, `multipliers`,
                        `inputs` and `baselines` are provided in tuples of same
                        length. `custom_attribution_func` returns a tuple of
                        attribution tensors that have the same length as the
                        `inputs`.
                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Attribution score computed based on DeepLift rescale rule with
                        respect to each input feature. Attributions will always be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
            - **delta** (*Tensor*, returned if return_convergence_delta=True):
                        This is computed using the property that the
                        total sum of model(inputs) - model(baselines)
                        must be very close to the total sum of attributions
                        computed based on approximated SHAP values using
                        Deeplift's rescale rule.
                        Delta is calculated for each example input and baseline pair,
                        meaning that the number of elements in returned delta tensor
                        is equal to the
                        `number of examples in input` * `number of examples
                        in baseline`. The deltas are ordered in the first place by
                        input example, followed by the baseline.
                        Note that the logic described for deltas is guaranteed
                        when the default logic for attribution computations is used,
                        meaning that the `custom_attribution_func=None`, otherwise
                        it is not guaranteed and depends on the specifics of the
                        `custom_attribution_func`.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> dl = DeepLiftShap(net)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes shap values using deeplift for class 3.
            >>> attribution = dl.attribute(input, target=3)
        """
        formatted_baselines = _format_callable_baseline(baselines, inputs)

        assert (
            isinstance(formatted_baselines[0], torch.Tensor)
            and formatted_baselines[0].shape[0] > 1
        ), (
            "Baselines distribution has to be provided in form of a torch.Tensor"
            " with more than one example but found: {}."
            " If baselines are provided in shape of scalars or with a single"
            " baseline example, `DeepLift`"
            " approach can be used instead.".format(formatted_baselines[0])
        )

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs_tuple = _format_tensor_into_tuples(inputs)

        # batch sizes
        inp_bsz = inputs_tuple[0].shape[0]
        base_bsz = formatted_baselines[0].shape[0]

        (
            exp_inp,
            exp_base,
            exp_tgt,
            exp_addit_args,
        ) = self._expand_inputs_baselines_targets(
            formatted_baselines,
            inputs_tuple,
            target,
            additional_forward_args,
        )
        attributions = super().attribute.__wrapped__(  # type: ignore
            self,
            exp_inp,
            exp_base,
            target=exp_tgt,
            additional_forward_args=exp_addit_args,
            return_convergence_delta=cast(
                Literal[True, False],
                return_convergence_delta,
            ),
            custom_attribution_func=custom_attribution_func,
        )
        delta: Tensor = torch.tensor(0)
        if return_convergence_delta:
            attributions, delta = cast(Tuple[Tuple[Tensor, ...], Tensor], attributions)

        attributions = tuple(
            self._compute_mean_across_baselines(
                inp_bsz, base_bsz, cast(Tensor, attribution)
            )
            for attribution in attributions
        )

        if return_convergence_delta:
            return (
                cast(
                    TensorOrTupleOfTensorsGeneric,
                    _format_output(is_inputs_tuple, attributions),
                ),
                delta,
            )
        else:
            return cast(
                TensorOrTupleOfTensorsGeneric,
                _format_output(is_inputs_tuple, attributions),
            )

    def _expand_inputs_baselines_targets(
        self,
        baselines: Tuple[Tensor, ...],
        inputs: Tuple[Tensor, ...],
        target: TargetType,
        additional_forward_args: Optional[Tuple[object, ...]],
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], TargetType, object]:
        inp_bsz = inputs[0].shape[0]
        base_bsz = baselines[0].shape[0]

        expanded_inputs = tuple(
            [
                input.repeat_interleave(base_bsz, dim=0).requires_grad_()
                for input in inputs
            ]
        )
        expanded_baselines = tuple(
            [
                baseline.repeat(
                    (inp_bsz,) + tuple([1] * (len(baseline.shape) - 1))
                ).requires_grad_()
                for baseline in baselines
            ]
        )
        expanded_target = _expand_target(
            target, base_bsz, expansion_type=ExpansionTypes.repeat_interleave
        )
        input_additional_args = (
            _expand_additional_forward_args(
                additional_forward_args,
                base_bsz,
                expansion_type=ExpansionTypes.repeat_interleave,
            )
            if additional_forward_args is not None
            else None
        )
        return (
            expanded_inputs,
            expanded_baselines,
            expanded_target,
            input_additional_args,
        )

    def _compute_mean_across_baselines(
        self, inp_bsz: int, base_bsz: int, attribution: Tensor
    ) -> Tensor:
        # Average for multiple references
        attr_shape: Tuple[int, ...] = (inp_bsz, base_bsz)
        if len(attribution.shape) > 1:
            attr_shape += tuple(attribution.shape[1:])
        return torch.mean(attribution.view(attr_shape), dim=1, keepdim=False)


def nonlinear(
    module: Module,
    inputs: Tensor,
    outputs: Tensor,
    grad_input: Tensor,
    grad_output: Tensor,
    eps: float = 1e-10,
) -> Tensor:
    r"""
    grad_input: (dLoss / dprev_layer_out, dLoss / wij, dLoss / bij)
    grad_output: (dLoss / dlayer_out)
    https://github.com/pytorch/pytorch/issues/12331
    """
    delta_in, delta_out = _compute_diffs(inputs, outputs)

    new_grad_inp = torch.where(
        abs(delta_in) < eps, grad_input, grad_output * delta_out / delta_in
    )

    return new_grad_inp


def softmax(
    module: Module,
    inputs: Tensor,
    outputs: Tensor,
    grad_input: Tensor,
    grad_output: Tensor,
    eps: float = 1e-10,
) -> Tensor:
    delta_in, delta_out = _compute_diffs(inputs, outputs)

    grad_input_unnorm = torch.where(
        abs(delta_in) < eps, grad_input, grad_output * delta_out / delta_in
    )
    return grad_input_unnorm


def maxpool1d(
    module: Module,
    inputs: Tensor,
    outputs: Tensor,
    grad_input: Tensor,
    grad_output: Tensor,
    eps: float = 1e-10,
) -> Tensor:
    return maxpool(
        module,
        F.max_pool1d,
        F.max_unpool1d,
        inputs,
        outputs,
        grad_input,
        grad_output,
        eps=eps,
    )


def maxpool2d(
    module: Module,
    inputs: Tensor,
    outputs: Tensor,
    grad_input: Tensor,
    grad_output: Tensor,
    eps: float = 1e-10,
) -> Tensor:
    return maxpool(
        module,
        F.max_pool2d,
        F.max_unpool2d,
        inputs,
        outputs,
        grad_input,
        grad_output,
        eps=eps,
    )


def maxpool3d(
    module: Module,
    inputs: Tensor,
    outputs: Tensor,
    grad_input: Tensor,
    grad_output: Tensor,
    eps: float = 1e-10,
) -> Tensor:
    return maxpool(
        module,
        F.max_pool3d,
        F.max_unpool3d,
        inputs,
        outputs,
        grad_input,
        grad_output,
        eps=eps,
    )


def maxpool(
    module: Module,
    pool_func: Callable[..., Tuple[Tensor, Tensor]],
    unpool_func: Callable[..., Tensor],
    inputs: Tensor,
    outputs: Tensor,
    grad_input: Tensor,
    grad_output: Tensor,
    eps: float = 1e-10,
) -> Tensor:
    with torch.no_grad():
        input, input_ref = inputs.chunk(2)
        output, output_ref = outputs.chunk(2)

        delta_in = input - input_ref
        delta_in = torch.cat(2 * [delta_in])
        # Extracts cross maximum between the outputs of maxpool for the
        # actual inputs and its corresponding references. In case the delta outputs
        # for the references are larger the method relies on the references and
        # corresponding gradients to compute the multiplies and contributions.
        delta_out_xmax = torch.max(output, output_ref)
        delta_out = torch.cat([delta_out_xmax - output_ref, output - delta_out_xmax])

        _, indices = pool_func(
            module.input,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.ceil_mode,
            True,
        )
        grad_output_updated = grad_output
        unpool_grad_out_delta, unpool_grad_out_ref_delta = torch.chunk(
            unpool_func(
                grad_output_updated * delta_out,
                indices,
                module.kernel_size,
                module.stride,
                module.padding,
                list(cast(torch.Size, module.input.shape)),
            ),
            2,
        )

    unpool_grad_out_delta = unpool_grad_out_delta + unpool_grad_out_ref_delta
    unpool_grad_out_delta = torch.cat(2 * [unpool_grad_out_delta])

    if grad_input.shape != inputs.shape:
        raise AssertionError(
            "A problem occurred during maxpool modul's backward pass. "
            "The gradients with respect to inputs include only a "
            "subset of inputs. More details about this issue can "
            "be found here: "
            "https://pytorch.org/docs/stable/"
            "nn.html#torch.nn.Module.register_backward_hook "
            "This can happen for example if you attribute to the outputs of a "
            "MaxPool. As a workaround, please, attribute to the inputs of "
            "the following layer."
        )

    new_grad_inp = torch.where(
        abs(delta_in) < eps, grad_input, unpool_grad_out_delta / delta_in
    )
    return new_grad_inp


def _compute_diffs(inputs: Tensor, outputs: Tensor) -> Tuple[Tensor, Tensor]:
    input, input_ref = inputs.chunk(2)
    # if the model is a single non-linear module and we apply Rescale rule on it
    # we might not be able to perform chunk-ing because the output of the module is
    # usually being replaced by model output.
    output, output_ref = outputs.chunk(2)
    delta_in = input - input_ref
    delta_out = output - output_ref

    return torch.cat(2 * [delta_in]), torch.cat(2 * [delta_out])


SUPPORTED_NON_LINEAR: Dict[Type[Module], Callable[..., Tensor]] = {
    nn.ReLU: nonlinear,
    nn.ELU: nonlinear,
    nn.LeakyReLU: nonlinear,
    nn.Sigmoid: nonlinear,
    nn.Tanh: nonlinear,
    nn.Softplus: nonlinear,
    nn.SELU: nonlinear,
    nn.GELU: nonlinear,
    nn.MaxPool1d: maxpool1d,
    nn.MaxPool2d: maxpool2d,
    nn.MaxPool3d: maxpool3d,
    nn.Softmax: softmax,
}
