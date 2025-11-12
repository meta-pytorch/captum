#!/usr/bin/env python3

# pyre-strict

from typing import Dict, List, no_type_check, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum._utils.typing import PassThroughOutputType
from torch import Tensor
from torch.futures import Future

"""
@no_type_check annotation is applied to type-hinted models to avoid errors
related to mismatch with parent (nn.Module) signature. # type_ignore is not
possible here, since it causes errors in JIT scripting code which parses
the relevant type hints.
"""


class BasicLinearReLULinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int = 5, bias: bool = False
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features, bias=bias)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(out_features, 1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class MixedKwargsAndArgsModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        if y is not None:
            return x + y
        return x


class BasicModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        input = 1 - F.relu(1 - input)
        return input


class BasicModel2(nn.Module):
    """
    Example model one from the paper
    https://arxiv.org/pdf/1703.01365.pdf

    f(x1, x2) = RELU(ReLU(x1) - 1 - ReLU(x2))
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        relu_out1 = F.relu(input1)
        relu_out2 = F.relu(input2)
        return F.relu(relu_out1 - 1 - relu_out2)


class BasicModel3(nn.Module):
    """
    Example model two from the paper
    https://arxiv.org/pdf/1703.01365.pdf

    f(x1, x2) = RELU(ReLU(x1 - 1) - ReLU(x2))
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        relu_out1 = F.relu(input1 - 1)
        relu_out2 = F.relu(input2)
        return F.relu(relu_out1 - relu_out2)


class BasicModel4_MultiArgs(nn.Module):
    """
    Slightly modified example model from the paper
    https://arxiv.org/pdf/1703.01365.pdf
    f(x1, x2) = RELU(ReLU(x1 - 1) - ReLU(x2) / x3)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input1: Tensor,
        input2: Tensor,
        additional_input1: Union[bool, float, int, Tensor],
        additional_input2: int = 0,
    ) -> Tensor:
        relu_out1 = F.relu(input1 - 1)
        relu_out2 = F.relu(input2)
        relu_out2 = relu_out2.div(additional_input1)
        return F.relu(relu_out1 - relu_out2)[:, additional_input2]


class BasicModel5_MultiArgs(nn.Module):
    """
    Slightly modified example model from the paper
    https://arxiv.org/pdf/1703.01365.pdf
    f(x1, x2) = RELU(ReLU(x1 - 1) * x3[0] - ReLU(x2) * x3[1])
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input1: Tensor,
        input2: Tensor,
        additional_input1: List[int],
        additional_input2: int = 0,
    ) -> Tensor:
        relu_out1 = F.relu(input1 - 1) * additional_input1[0]
        relu_out2 = F.relu(input2)
        relu_out2 = relu_out2 * additional_input1[1]
        return F.relu(relu_out1 - relu_out2)[:, additional_input2]


class BasicModel6_MultiTensor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        input = input1 + input2
        return 1 - F.relu(1 - input)[:, 1]


class BasicLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(7, 1)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.linear(torch.cat((x1, x2), dim=-1))


class BasicLinearModel2(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        return self.linear(input)


class BasicLinearModel_Multilayer(nn.Module):
    def __init__(self, in_features: int, hidden_nodes: int, out_features: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_nodes, bias=False)
        self.linear2 = nn.Linear(hidden_nodes, out_features, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        x = self.linear1(input)
        return self.linear2(x)


class ReLUDeepLiftModel(nn.Module):
    r"""
    https://www.youtube.com/watch?v=f_iAM0NPwnM
    """

    def __init__(self) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x1: Tensor, x2: Tensor, x3: int = 2) -> int:
        return 2 * self.relu1(x1) + x3 * self.relu2(x2 - 1.5)


class LinearMaxPoolLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # kernel size -> 4
        self.lin1 = nn.Linear(4, 4, bias=False)
        self.lin1.weight = nn.Parameter(torch.eye(4, 4))
        self.pool1 = nn.MaxPool1d(4)
        self.lin2 = nn.Linear(1, 1, bias=False)
        self.lin2.weight = nn.Parameter(torch.ones(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        return self.lin2(self.pool1(self.lin1(x))[:, 0, :])


class BasicModelWithReusedModules(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(3, 2)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(2, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.relu(self.lin2(self.relu(self.lin1(inputs))))


class BasicModelWithReusedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(3, 3)
        self.relu = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.relu(self.lin1(self.relu(self.lin1(inputs))))


class BasicModelWithSparseInputs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(3, 1)
        self.lin1.weight = nn.Parameter(torch.tensor([[3.0, 1.0, 2.0]]))
        self.lin1.bias = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: Tensor, sparse_list: Tensor) -> Tensor:
        return (
            self.lin1(inputs) + (sparse_list[0] if torch.numel(sparse_list) > 0 else 0)
        ).sum()


class BasicModel_MaxPool_ReLU(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool1d(3)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.maxpool(x)).sum(dim=1)


class TanhDeepLiftModel(nn.Module):
    r"""
    Same as the ReLUDeepLiftModel, but with activations
    that can have negative outputs
    """

    def __init__(self) -> None:
        super().__init__()
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()

    def forward(self, x1: Tensor, x2: Tensor) -> int:
        return 2 * self.tanh1(x1) + 2 * self.tanh2(x2 - 1.5)


class ReLULinearModel(nn.Module):
    r"""
    Simple architecture similar to:
    https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tests/test_tensorflow.py#L65
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.l1 = nn.Linear(3, 1, bias=False)
        self.l2 = nn.Linear(3, 1, bias=False)
        self.l1.weight = nn.Parameter(torch.tensor([[3.0, 1.0, 0.0]]))  # type: ignore
        self.l2.weight = nn.Parameter(torch.tensor([[2.0, 3.0, 0.0]]))  # type: ignore
        self.relu = nn.ReLU(inplace=inplace)
        self.l3 = nn.Linear(2, 1, bias=False)
        self.l3.weight = nn.Parameter(torch.tensor([[1.0, 1.0]]))  # type: ignore

    @no_type_check
    def forward(self, x1: Tensor, x2: Tensor, x3: int = 1) -> Tensor:
        return self.l3(self.relu(torch.cat([self.l1(x1), x3 * self.l2(x2)], dim=1)))


class SimpleLRPModel(nn.Module):
    def __init__(self, inplace: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)
        self.linear.weight.data.fill_(2.0)
        self.relu = torch.nn.ReLU(inplace=inplace)
        self.linear2 = nn.Linear(3, 1, bias=False)
        self.linear2.weight.data.fill_(3.0)
        self.dropout = torch.nn.Dropout(p=0.01)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.linear2(self.relu(self.linear(x))))


class Conv1dSeqModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(nn.Conv1d(4, 2, 1), nn.ReLU(), nn.Linear(1000, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.seq(inputs)


class TextModule(nn.Module):
    r"""Basic model that has inner embedding layer. This layer can be pluged
    into a larger network such as `BasicEmbeddingModel` and help us to test
    nested embedding layers
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        second_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.inner_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.second_embedding = second_embedding
        if self.second_embedding:
            self.inner_embedding2: nn.Embedding = nn.Embedding(
                num_embeddings, embedding_dim
            )

    def forward(
        self,
        input: Optional[Tensor] = None,
        another_input: Optional[Tensor] = None,
    ) -> Tensor:
        assert input is not None, "The inputs to embedding module must be specified"
        embedding = self.inner_embedding(input)
        if self.second_embedding:
            another_embedding = self.inner_embedding2(
                input if another_input is None else another_input
            )
        # pyre-fixme[61]: `another_embedding` is undefined, or not always defined.
        return embedding if another_input is None else embedding + another_embedding


class BasicEmbeddingModel(nn.Module):
    r"""
    Implements basic model with nn.Embedding layer. This simple model
    will help us to test nested InterpretableEmbedding layers
    The model has the following structure:
    BasicEmbeddingModel(
      (embedding1): Embedding(30, 100)
      (embedding2): TextModule(
        (inner_embedding): Embedding(30, 100)
      )
      (linear1): Linear(in_features=100, out_features=256, bias=True)
      (relu): ReLU()
      (linear2): Linear(in_features=256, out_features=1, bias=True)
    )
    """

    def __init__(
        self,
        num_embeddings: int = 30,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 1,
        nested_second_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.embedding1 = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding2 = TextModule(
            num_embeddings, embedding_dim, nested_second_embedding
        )
        self.linear1 = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.linear1.weight = nn.Parameter(torch.ones(hidden_dim, embedding_dim))
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.linear2.weight = nn.Parameter(torch.ones(output_dim, hidden_dim))

    def forward(
        self, input1: Tensor, input2: Tensor, input3: Optional[Tensor] = None
    ) -> Tensor:
        embedding1 = self.embedding1(input1)
        embedding2 = self.embedding2(input2, input3)
        embeddings = embedding1 + embedding2
        return self.linear2(self.relu(self.linear1(embeddings))).sum(1)


class PassThroughLayerOutput(nn.Module):
    """
    This layer is used to test the case where the model returns a layer that
    is not supported by the gradient computation.
    """

    def __init__(self) -> None:
        super().__init__()

    @no_type_check
    def forward(self, output: PassThroughOutputType) -> PassThroughOutputType:
        return output


class BasicModel_GradientLayerAttribution(nn.Module):
    def __init__(
        self,
        inplace: bool = False,
        unsupported_layer_output: Optional[PassThroughOutputType] = None,
        dict_output: bool = True,
    ) -> None:
        super().__init__()

        self.dict_output = dict_output
        # Linear 0 is simply identity transform
        self.unsupported_layer_output = unsupported_layer_output
        self.linear0 = nn.Linear(3, 3)
        self.linear0.weight = nn.Parameter(torch.eye(3))
        self.linear0.bias = nn.Parameter(torch.zeros(3))
        self.linear1 = nn.Linear(3, 4)
        self.linear1.weight = nn.Parameter(torch.ones(4, 3))
        self.linear1.bias = nn.Parameter(torch.tensor([-10.0, 1.0, 1.0, 1.0]))

        self.linear1_alt = nn.Linear(3, 4)
        self.linear1_alt.weight = nn.Parameter(torch.ones(4, 3))
        self.linear1_alt.bias = nn.Parameter(torch.tensor([-10.0, 1.0, 1.0, 1.0]))

        self.relu = nn.ReLU(inplace=inplace)
        self.relu_alt = nn.ReLU(inplace=False)
        self.unsupported_layer = PassThroughLayerOutput()

        self.linear2 = nn.Linear(4, 2)
        self.linear2.weight = nn.Parameter(torch.ones(2, 4))
        self.linear2.bias = nn.Parameter(torch.tensor([-1.0, 1.0]))

        self.linear3 = nn.Linear(4, 2)
        self.linear3.weight = nn.Parameter(torch.ones(2, 4))
        self.linear3.bias = nn.Parameter(torch.tensor([-1.0, 1.0]))

        self.list_output_layer = PassThroughLayerOutput()

        self.int_layer = PassThroughLayerOutput()  # sample layer with an int output

    @no_type_check
    def forward(
        self, x: Tensor, add_input: Optional[Tensor] = None
    ) -> Union[Dict[str, Tensor], Tensor]:
        input = x if add_input is None else x + add_input
        lin0_out = self.linear0(input)
        lin1_out = self.linear1(lin0_out)
        lin1_out_alt = self.linear1_alt(lin0_out)

        if self.unsupported_layer_output is not None:
            self.unsupported_layer(self.unsupported_layer_output)
            # unsupportedLayer is unused in the forward func.
        self.relu_alt(
            lin1_out_alt
        )  # relu_alt's output is supported but it's unused in the forward func.

        relu_out = self.relu(lin1_out)
        lin2_out = self.linear2(relu_out)
        list_out = self.list_output_layer([nn.Linear(2, 2)(lin2_out) for _ in range(2)])
        resized_list_out = torch.cat(list_out, dim=1)

        lin3_out = self.linear3(lin1_out_alt)
        int_output = self.int_layer(lin3_out.to(torch.int64))

        output_tensors = torch.cat((resized_list_out, int_output), dim=1)

        return (
            output_tensors
            if not self.dict_output
            else {
                "task {}".format(i + 1): output_tensors[:, i]
                for i in range(output_tensors.shape[1])
            }
        )
        # we return a dictionary of tensors as an output to test the case
        # where an output accessor is required


class MultiRelu(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.relu1 = nn.ReLU(inplace=inplace)
        self.relu2 = nn.ReLU(inplace=inplace)

    @no_type_check
    def forward(self, arg1: Tensor, arg2: Tensor) -> Tuple[Tensor, Tensor]:
        return (self.relu1(arg1), self.relu2(arg2))


class BasicModel_MultiLayer(nn.Module):
    def __init__(
        self,
        inplace: bool = False,
        multi_input_module: bool = False,
    ) -> None:
        super().__init__()
        # Linear 0 is simply identity transform
        self.multi_input_module = multi_input_module
        self.linear0 = nn.Linear(3, 3)
        self.linear0.weight = nn.Parameter(torch.eye(3))
        self.linear0.bias = nn.Parameter(torch.zeros(3))
        self.linear1 = nn.Linear(3, 4)
        self.linear1.weight = nn.Parameter(torch.ones(4, 3))
        self.linear1.bias = nn.Parameter(torch.tensor([-10.0, 1.0, 1.0, 1.0]))

        self.linear1_alt = nn.Linear(3, 4)
        self.linear1_alt.weight = nn.Parameter(torch.ones(4, 3))
        self.linear1_alt.bias = nn.Parameter(torch.tensor([-10.0, 1.0, 1.0, 1.0]))
        self.multi_relu = MultiRelu(inplace=inplace)
        self.relu = nn.ReLU(inplace=inplace)

        self.linear2 = nn.Linear(4, 2)
        self.linear2.weight = nn.Parameter(torch.ones(2, 4))
        self.linear2.bias = nn.Parameter(torch.tensor([-1.0, 1.0]))

    @no_type_check
    def forward(
        self,
        x: Tensor,
        add_input: Optional[Tensor] = None,
        multidim_output: bool = False,
    ) -> Tensor:
        input = x if add_input is None else x + add_input
        lin0_out = self.linear0(input)
        lin1_out = self.linear1(lin0_out)

        if self.multi_input_module:
            relu_out1, relu_out2 = self.multi_relu(lin1_out, self.linear1_alt(input))
            relu_out = relu_out1 + relu_out2
            # relu is not used when multi_input_module set to True,
            # so this is to set an unsued layer intentionally for testing
            # and it won't be part of return
            self.relu(lin1_out)
        else:
            relu_out = self.relu(lin1_out)
        lin2_out = self.linear2(relu_out)
        if multidim_output:
            stack_mid = torch.stack((lin2_out, 2 * lin2_out), dim=2)
            return torch.stack((stack_mid, 4 * stack_mid), dim=3)
        else:
            return lin2_out


class BasicModel_MultiLayer_with_Future(nn.Module):
    # This model is used to test the case where the model returns a future
    def __init__(self, inplace: bool = False, multi_input_module: bool = False) -> None:
        super().__init__()
        # Linear 0 is simply identity transform
        self.multi_input_module = multi_input_module
        self.linear0 = nn.Linear(3, 3)
        self.linear0.weight = nn.Parameter(torch.eye(3))
        self.linear0.bias = nn.Parameter(torch.zeros(3))
        self.linear1 = nn.Linear(3, 4)
        self.linear1.weight = nn.Parameter(torch.ones(4, 3))
        self.linear1.bias = nn.Parameter(torch.tensor([-10.0, 1.0, 1.0, 1.0]))

        self.linear1_alt = nn.Linear(3, 4)
        self.linear1_alt.weight = nn.Parameter(torch.ones(4, 3))
        self.linear1_alt.bias = nn.Parameter(torch.tensor([-10.0, 1.0, 1.0, 1.0]))
        self.multi_relu = MultiRelu(inplace=inplace)
        self.relu = nn.ReLU(inplace=inplace)

        self.linear2 = nn.Linear(4, 2)
        self.linear2.weight = nn.Parameter(torch.ones(2, 4))
        self.linear2.bias = nn.Parameter(torch.tensor([-1.0, 1.0]))

    @no_type_check
    def forward(
        self,
        x: Tensor,
        add_input: Optional[Tensor] = None,
        multidim_output: bool = False,
    ) -> Future[Tensor]:
        input = x if add_input is None else x + add_input
        lin0_out = self.linear0(input)
        lin1_out = self.linear1(lin0_out)
        if self.multi_input_module:
            relu_out1, relu_out2 = self.multi_relu(lin1_out, self.linear1_alt(input))
            relu_out = relu_out1 + relu_out2
            # relu is not used when multi_input_module set to True,
            # so this is to set an unsued layer intentionally for testing
            # and it won't be part of return
            self.relu(lin1_out)
        else:
            relu_out = self.relu(lin1_out)
        # pyre-fixme [29]: `typing.Type[Future]` is not a function
        result = Future()
        lin2_out = self.linear2(relu_out)
        if multidim_output:
            stack_mid = torch.stack((lin2_out, 2 * lin2_out), dim=2)
            result.set_result(torch.stack((stack_mid, 4 * stack_mid), dim=3))
            return result
        else:
            result.set_result(lin2_out)
            return result


class BasicModelBoolInput_with_Future(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = BasicModel_MultiLayer_with_Future()

    def forward(
        self,
        x: Tensor,
        add_input: Optional[Tensor] = None,
        mult: float = 10.0,
    ) -> Future[Tensor]:
        assert x.dtype is torch.bool, "Input must be boolean"
        return self.mod(x.float() * mult, add_input)


class BasicModelBoolInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = BasicModel_MultiLayer()

    def forward(
        self,
        x: Tensor,
        add_input: Optional[Tensor] = None,
        mult: float = 10.0,
    ) -> Tensor:
        assert x.dtype is torch.bool, "Input must be boolean"
        return self.mod(x.float() * mult, add_input)


class BasicModel_MultiLayer_MultiInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = BasicModel_MultiLayer()

    @no_type_check
    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor, scale: int) -> Tensor:
        return self.model(scale * (x1 + x2 + x3))


class BasicModel_MultiLayer_TupleInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = BasicModel_MultiLayer()

    @no_type_check
    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        return self.model(x[0] + x[1] + x[2])


class BasicModel_MultiLayer_MultiInput_with_Future(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = BasicModel_MultiLayer_with_Future()

    @no_type_check
    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor, scale: int) -> Future[Tensor]:
        return self.model(scale * (x1 + x2 + x3))


class BasicModel_MultiLayer_TrueMultiInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m1 = BasicModel_MultiLayer()
        self.m234 = BasicModel_MultiLayer_MultiInput()

    @no_type_check
    def forward(
        self, x1: Tensor, x2: Tensor, x3: Tensor, x4: Optional[Tensor] = None
    ) -> Tensor:
        a = self.m1(x1)
        if x4 is None:
            b = self.m234(x2, x3, x1, scale=1)
        else:
            b = self.m234(x2, x3, x4, scale=1)
        return a + b


class BasicModel_ConvNet_One_Conv(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1)
        self.relu1 = nn.ReLU(inplace=inplace)
        self.fc1 = nn.Linear(8, 4)
        self.conv1.weight = nn.Parameter(torch.ones(2, 1, 3, 3))  # type: ignore
        self.conv1.bias = nn.Parameter(torch.tensor([-50.0, -75.0]))  # type: ignore
        self.fc1.weight = nn.Parameter(  # type: ignore
            torch.cat([torch.ones(4, 5), -1 * torch.ones(4, 3)], dim=1)
        )
        self.fc1.bias = nn.Parameter(torch.zeros(4))  # type: ignore
        self.relu2 = nn.ReLU(inplace=inplace)

    @no_type_check
    def forward(self, x: Tensor, x2: Optional[Tensor] = None) -> Tensor:
        if x2 is not None:
            x = x + x2
        x = self.relu1(self.conv1(x))
        x = x.view(-1, 8)
        return self.relu2(self.fc1(x))


class BasicModel_ConvNetWithPaddingDilation(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=3, stride=2, dilation=2)
        self.relu1 = nn.ReLU(inplace=inplace)
        self.fc1 = nn.Linear(16, 4)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        x = self.relu1(self.conv1(x))
        x = x.reshape(bsz, 2, -1)
        return self.fc1(x).reshape(bsz, -1)


class BasicModel_ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(2, 4, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4, 8)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(8, 10)
        self.softmax = nn.Softmax(dim=1)

        self.fc1.weight = nn.Parameter(torch.ones(8, 4))
        self.fc2.weight = nn.Parameter(torch.ones(10, 8))

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 4)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


class BasicModel_ConvNet_MaxPool1d(nn.Module):
    """Same as above, but with the MaxPool2d replaced
    with a MaxPool1d. This is useful because the MaxPool modules
    behave differently to other modules from the perspective
    of the DeepLift Attributions
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 2, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(2, 4, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(4, 8)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(8, 10)
        self.softmax = nn.Softmax(dim=1)

        self.fc1.weight = nn.Parameter(torch.ones(8, 4))
        self.fc2.weight = nn.Parameter(torch.ones(10, 8))

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 4)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


class BasicModel_ConvNet_MaxPool3d(nn.Module):
    """Same as above, but with the MaxPool1d replaced
    with a MaxPool3d. This is useful because the MaxPool modules
    behave differently to other modules from the perspective
    of the DeepLift Attributions
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(1, 2, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(2, 4, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(4, 8)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(8, 10)
        self.softmax = nn.Softmax(dim=1)

        self.fc1.weight = nn.Parameter(torch.ones(8, 4))
        self.fc2.weight = nn.Parameter(torch.ones(10, 8))

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 4)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


class BasicModel_ConvNet_Optim(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Conv2d(3, 2, 1, bias=True)
        # Initialize weights and biases for
        # easy reproducibility
        self.layer.weight.data.fill_(0.1)
        if self.layer.bias is not None:
            self.layer.bias.data.fill_(1)

    @no_type_check
    def forward(self, x: Tensor):
        return self.layer(x)
