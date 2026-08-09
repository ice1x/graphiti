"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import json
import logging
import typing
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig, ModelSize
from .errors import RateLimitError

if TYPE_CHECKING:
    import groq
    from groq import AsyncGroq
    from groq.types.chat import ChatCompletionMessageParam
else:
    # Defer the optional-SDK requirement to construction time: importing this module
    # must never fail just because the `groq` extra is not installed (issue #18).
    try:
        import groq
        from groq import AsyncGroq
        from groq.types.chat import ChatCompletionMessageParam
    except ImportError:
        groq = None
        AsyncGroq = None
        ChatCompletionMessageParam = None

_GROQ_IMPORT_ERROR = ImportError(
    'groq is required for GroqClient. Install it with: pip install graphiti-core[groq]'
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'llama-3.1-70b-versatile'
DEFAULT_MAX_TOKENS = 2048


class GroqClient(LLMClient):
    def __init__(self, config: LLMConfig | None = None, cache: bool = False):
        if groq is None:
            raise _GROQ_IMPORT_ERROR

        if config is None:
            config = LLMConfig(max_tokens=DEFAULT_MAX_TOKENS)
        elif config.max_tokens is None:
            config.max_tokens = DEFAULT_MAX_TOKENS
        super().__init__(config, cache)

        self.client = AsyncGroq(api_key=config.api_key)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        msgs: list[ChatCompletionMessageParam] = []
        for m in messages:
            if m.role == 'user':
                msgs.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                msgs.append({'role': 'system', 'content': m.content})
        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=msgs,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format={'type': 'json_object'},
            )
            result = response.choices[0].message.content or ''
            return json.loads(result)
        except groq.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise
