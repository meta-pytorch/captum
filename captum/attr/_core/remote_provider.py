# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from captum._utils.typing import TokenizerLike

logger: logging.Logger = logging.getLogger(__name__)


class RemoteLLMProvider(ABC):
    """All remote LLM providers that offer logprob via API
    (like vLLM) extends this class."""

    api_url: str

    @abstractmethod
    def generate(self, prompt: str, **gen_args: Any) -> str:
        """
        Args:
            prompt: The input prompt to generate from
            gen_args: Additional generation arguments

        Returns:
            The generated text.
        """
        ...

    @abstractmethod
    def get_logprobs(
        self,
        input_prompt: str,
        target_str: str,
        tokenizer: Optional[TokenizerLike] = None,
    ) -> List[float]:
        """
        Get the log probabilities for all tokens in the target string.

        Args:
            input_prompt: The input prompt
            target_str: The target string
            tokenizer: The tokenizer to use

        Returns:
            A list of log probabilities corresponding to
            each token in the target prompt.
            For a `target_str` of `t` tokens,
            this method returns a list of logprobs of length `k`.
        """
        ...


class VLLMProvider(RemoteLLMProvider):
    model_name: str

    def __init__(self, api_url: str, model_name: Optional[str] = None) -> None:
        """
        Initialize a vLLM provider.

        Args:
            api_url: The URL of the vLLM API
            model_name: The name of the model to use. If None, the first model from
                        the API's model list will be used.

        Environment Variables:
            OPENAI_API_KEY: If not set, "EMPTY" will be used as the API key.

        Raises:
            ValueError: If api_url is empty or model_name is not in the API's model list
            ConnectionError: If API connection fails
            ImportError: If the openai package is not installed
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required to use the VLLMProvider."
                "You can install it by either:\n"
                "1. Installing captum with remote dependencies: "
                "`pip install captum[remote]` OR\n"
                "2. Installing openai directly: `pip install openai`"
            )

        if not api_url.strip():
            raise ValueError("API URL is required")

        self.api_url = api_url

        try:
            self.client: OpenAI = OpenAI(
                base_url=self.api_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY")
            )

            # If model_name is not provided, get the first available model from the API
            if model_name is None:
                models = self.client.models.list().data
                if not models:
                    raise ValueError("No models available from the vLLM API")
                self.model_name = models[0].id
                logger.info(
                    f"No model_name is specified for VLLMProvider."
                    f" Using first available model: {self.model_name}"
                )
            else:
                self.model_name = model_name

        except ValueError:
            raise
        except ConnectionError as e:
            raise ConnectionError(f"Failed to connect to vLLM API: {str(e)}")
        except Exception as e:
            raise Exception(
                f"Unexpected error while initializing vLLM provider: {str(e)}"
            )

    def generate(self, prompt: str, **gen_args: Any) -> str:
        """
        Generate text using the vLLM API.

        Args:
            prompt: The input prompt for text generation
            **gen_args: Additional generation arguments. Supported arguments include:
                - max_tokens: Maximum number of tokens to generate (default: 25)
                - max_new_tokens: Alternative to max_tokens
                                    (will be converted to max_tokens)
                - temperature, top_p, etc.: Other generation parameters
                                    supported by the OpenAI API

        Returns:
            str: The generated text

        Raises:
            KeyError: If API response is missing expected data
            ConnectionError: If connection to API fails
        """
        # Parameter normalization
        if "max_tokens" not in gen_args:
            gen_args["max_tokens"] = gen_args.pop("max_new_tokens", 25)
        if "do_sample" in gen_args:
            gen_args.pop("do_sample")

        try:
            response = self.client.completions.create(
                model=self.model_name, prompt=prompt, **gen_args
            )
            if not hasattr(response, "choices") or not response.choices:
                raise KeyError("API response missing expected 'choices' data")

            return response.choices[0].text

        except KeyError:
            raise
        except ConnectionError as e:
            raise ConnectionError(f"Failed to connect to vLLM API: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during text generation: {str(e)}")

    def get_logprobs(
        self,
        input_prompt: str,
        target_str: str,
        tokenizer: Optional[TokenizerLike] = None,
    ) -> List[float]:
        """
        Get the log probabilities for all tokens in the target string.

        Args:
            input_prompt: The input prompt
            target_str: The target string
            tokenizer: The tokenizer to use

        Returns:
            A list of log probabilities corresponding to each token
            in the target prompt.
            For a `target_str` of `t` tokens, this method returns
            a list of logprobs of length `t`.

        Raises:
            ValueError: If tokenizer is None or target_str is empty
                        or response format is invalid
            KeyError: If API response is missing expected data
            IndexError: If response format is unexpected
            ConnectionError: If connection to API fails
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required for vLLM provider")
        if not target_str:
            raise ValueError("Target string cannot be empty")

        num_target_str_tokens = len(
            tokenizer.encode(target_str, add_special_tokens=False)
        )

        prompt = input_prompt + target_str

        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=0.0,
                max_tokens=1,
                extra_body={"prompt_logprobs": 0},
            )

            if not hasattr(response, "choices") or not response.choices:
                raise KeyError("API response missing expected 'choices' data")

            if (
                not hasattr(response.choices[0], "prompt_logprobs")
                or not response.choices[0].prompt_logprobs
            ):
                raise KeyError("API response missing 'prompt_logprobs' data")

            prompt_logprobs = []
            try:
                for probs in response.choices[0].prompt_logprobs[
                    -num_target_str_tokens:
                ]:
                    if not probs:
                        raise ValueError("Empty probability data in API response")
                    assert len(probs) == 1, "Expected exactly one token in logprobs"
                    prompt_logprobs.append(next(iter(probs.values()))["logprob"])
            except (IndexError, KeyError) as e:
                raise IndexError(f"Unexpected format in log probability data: {str(e)}")

            if len(prompt_logprobs) != num_target_str_tokens:
                raise ValueError(
                    f"Not enough logprobs received:"
                    f"expected {num_target_str_tokens}, got {len(prompt_logprobs)}"
                )

            return prompt_logprobs

        except (KeyError, IndexError, ValueError):
            raise
        except ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to vLLM API when getting logprobs: {str(e)}"
            )
        except Exception as e:
            raise Exception(
                f"Unexpected error while getting log probabilities: {str(e)}"
            )
