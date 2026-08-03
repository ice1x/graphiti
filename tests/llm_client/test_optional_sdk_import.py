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

# Running tests: pytest -xvs tests/llm_client/test_optional_sdk_import.py

import importlib
import sys

import pytest

# (module path, client class name, top-level SDK modules to block)
OPTIONAL_CLIENTS = [
    ('graphiti_core.llm_client.anthropic_client', 'AnthropicClient', ['anthropic']),
    ('graphiti_core.llm_client.groq_client', 'GroqClient', ['groq']),
    ('graphiti_core.llm_client.gemini_client', 'GeminiClient', ['google.genai']),
    ('graphiti_core.llm_client.gliner2_client', 'GLiNER2Client', ['gliner2']),
]


def _import_with_blocked_sdks(module_name: str, block_modules: list[str]):
    """Import ``module_name`` fresh while forcing ``block_modules`` to be unimportable.

    Setting ``sys.modules[name] = None`` makes any ``import name`` raise ImportError,
    which faithfully simulates the optional provider SDK not being installed.
    Everything is restored afterwards so the blocking does not leak into other tests.
    """
    saved: dict[str, object] = {}
    for name in block_modules:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = None  # type: ignore[assignment]
    saved[module_name] = sys.modules.pop(module_name, None)
    try:
        return importlib.import_module(module_name)
    finally:
        for name in block_modules:
            if saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]  # type: ignore[assignment]
        sys.modules.pop(module_name, None)
        if saved[module_name] is not None:
            sys.modules[module_name] = saved[module_name]  # type: ignore[assignment]


@pytest.mark.parametrize(('module_name', 'class_name', 'block_modules'), OPTIONAL_CLIENTS)
def test_submodule_imports_without_provider_sdk(module_name, class_name, block_modules):
    """Importing an LLM client submodule must not raise when its SDK is absent (issue #18)."""
    module = _import_with_blocked_sdks(module_name, block_modules)
    assert hasattr(module, class_name)


@pytest.mark.parametrize(('module_name', 'class_name', 'block_modules'), OPTIONAL_CLIENTS)
def test_constructing_client_without_sdk_raises_importerror(module_name, class_name, block_modules):
    """The SDK requirement should surface as a clear ImportError at construction time."""
    module = _import_with_blocked_sdks(module_name, block_modules)
    client_cls = getattr(module, class_name)
    with pytest.raises(ImportError):
        client_cls()
