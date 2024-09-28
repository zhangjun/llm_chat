import time
from argparse import Namespace
from typing import Any, Dict, List, Literal, Optional, Union

import torch

from typing import Any, Dict, List, Literal, Optional, Union
from typing_extensions import Required, TypedDict, TypeAlias, Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator

from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam)
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam)

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from tensorrt_llm.hlapi import SamplingParams

try:
    from vllm.transformers_utils.tokenizers import (BaichuanTokenizer,
                                                    MistralTokenizer)
except ModuleNotFoundError:
    print("module not found")

# torch is mocked during docs generation,
# so we have to provide the values as literals
_MOCK_LONG_INFO = Namespace(min=-9223372036854775808, max=9223372036854775807)
_LONG_INFO: Union["torch.iinfo", Namespace]

try:
    from sphinx.ext.autodoc.mock import _MockModule

    if isinstance(torch, _MockModule):
        _LONG_INFO = _MOCK_LONG_INFO
    else:
        _LONG_INFO = torch.iinfo(torch.long)
except ModuleNotFoundError:
    _LONG_INFO = torch.iinfo(torch.long)

assert _LONG_INFO.min == _MOCK_LONG_INFO.min
assert _LONG_INFO.max == _MOCK_LONG_INFO.max

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

class CustomChatCompletionContentPartParam(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    type: Required[str]
    """The type of the content part."""

ChatCompletionContentPartParam: TypeAlias = Union[
    OpenAIChatCompletionContentPartParam, CustomChatCompletionContentPartParam, ]

class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""
    role: Required[str]
    """The role of the message's author."""

    content: Union[str, List[ChatCompletionContentPartParam]]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """

ChatCompletionMessageParam = Union[OpenAIChatCompletionMessageParam,
                                   CustomChatCompletionMessageParam]

class OpenAIBaseModel(BaseModel):
    # OpenAI API does not allow extra fields
    model_config = ConfigDict(extra="forbid")

class JsonSchemaResponseFormat(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: Optional[Dict[str, Any]] = Field(default=None, alias='schema')
    strict: Optional[bool] = None

class ResponseFormat(OpenAIBaseModel):
    # type must be "json_schema", "json_object" or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None

class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = True

class CompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    # doc: begin-completion-sampling-params
    use_beam_search: bool = False
    top_k: int = 0 # -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    allowed_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[int] = None
    # doc: end-completion-sampling-params

    # doc: begin-completion-extra-params
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."),
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=
        ("Similar to chat completion, this parameter specifies the format of "
         "output. Only {'type': 'json_object'} or {'type': 'text' } is "
         "supported."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be one of "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))

    # doc: end-completion-extra-params

    def to_sampling_params(
            self,
            # tokenizer: AnyTokenizer,
            # guided_decode_logits_processor: Optional[LogitsProcessor],
            default_max_tokens: int) -> SamplingParams:
        max_tokens = self.max_tokens
        if max_tokens is None:
            max_tokens = default_max_tokens

        prompt_logprobs = self.prompt_logprobs
        if prompt_logprobs is None and self.echo:
            prompt_logprobs = self.logprobs

        echo_without_generation = self.echo and self.max_tokens == 0

        # logits_processors = get_logits_processors(
        #     logit_bias=self.logit_bias,
        #     allowed_token_ids=self.allowed_token_ids,
        #     tokenizer=tokenizer,
        # )
        # if guided_decode_logits_processor:
        #     logits_processors.append(guided_decode_logits_processor)
        # tensorrt_llm/hlapi/utils.py
        sampling_params = SamplingParams()
        sampling_params.max_new_tokens = max_tokens if not echo_without_generation else 1
        sampling_params.stop = self.stop
        sampling_params.presence_penalty=self.presence_penalty
        sampling_params.frequency_penalty=self.frequency_penalty
        sampling_params.repetition_penalty=self.repetition_penalty
        sampling_params.temperature=self.temperature
        sampling_params.top_p=self.top_p
        sampling_params.top_k=self.top_k
        sampling_params.random_seed=self.seed
        sampling_params.stop_token_ids=self.stop_token_ids
        sampling_params.min_length=self.min_tokens
        sampling_params.early_stopping=self.early_stopping
        sampling_params.length_penalty=self.length_penalty

        return sampling_params
        # return SamplingParams.from_optional(
        #     n=self.n,
        #     best_of=self.best_of,
        #     presence_penalty=self.presence_penalty,
        #     frequency_penalty=self.frequency_penalty,
        #     repetition_penalty=self.repetition_penalty,
        #     temperature=self.temperature,
        #     top_p=self.top_p,
        #     top_k=self.top_k,
        #     min_p=self.min_p,
        #     seed=self.seed,
        #     stop=self.stop,
        #     stop_token_ids=self.stop_token_ids,
        #     logprobs=self.logprobs,
        #     ignore_eos=self.ignore_eos,
        #     max_tokens=max_tokens if not echo_without_generation else 1,
        #     min_tokens=self.min_tokens,
        #     use_beam_search=self.use_beam_search,
        #     early_stopping=self.early_stopping,
        #     prompt_logprobs=prompt_logprobs,
        #     skip_special_tokens=self.skip_special_tokens,
        #     spaces_between_special_tokens=self.spaces_between_special_tokens,
        #     include_stop_str_in_output=self.include_stop_str_in_output,
        #     length_penalty=self.length_penalty,
        #     # logits_processors=logits_processors,
        #     truncate_prompt_tokens=self.truncate_prompt_tokens,
        # )

    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and prompt_logprobs > 0:
                raise ValueError(
                    "`prompt_logprobs` are not available when `stream=True`.")

            if prompt_logprobs < 0:
                raise ValueError("`prompt_logprobs` must be a positive value.")

        if (logprobs := data.get("logprobs")) is not None and logprobs < 0:
            raise ValueError("`logprobs` must be a positive value.")

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when `stream=True`.")

        return data



class ErrorResponse(OpenAIBaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int

class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class ResponseFormat(OpenAIBaseModel):
    # type must be "json_schema", "json_object" or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


# class CompletionLogProbs(OpenAIBaseModel):
#     text_offset: List[int] = Field(default_factory=list)
#     token_logprobs: List[Optional[float]] = Field(default_factory=list)
#     tokens: List[str] = Field(default_factory=list)
#     top_logprobs: List[Optional[Dict[str,
#                                      float]]] = Field(default_factory=list)


class CompletionResponseChoice(OpenAIBaseModel):
    index: int
    text: str
    # logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )
    # prompt_logprobs: Optional[List[Optional[Dict[int, Logprob]]]] = None


class CompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    text: str
    # logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )


class CompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)

class TokenizeCompletionRequest(OpenAIBaseModel):
    model: str
    prompt: str

    add_special_tokens: bool = Field(default=True)


class TokenizeChatRequest(OpenAIBaseModel):
    model: str
    messages: List[ChatCompletionMessageParam]

    add_generation_prompt: bool = Field(default=True)
    add_special_tokens: bool = Field(default=False)


TokenizeRequest = Union[TokenizeCompletionRequest, TokenizeChatRequest]


class TokenizeResponse(OpenAIBaseModel):
    count: int
    max_model_len: int
    tokens: List[int]


class DetokenizeRequest(OpenAIBaseModel):
    model: str
    tokens: List[int]


class DetokenizeResponse(OpenAIBaseModel):
    prompt: str
