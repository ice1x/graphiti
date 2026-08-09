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

# Running tests: pytest -xvs tests/driver/test_drevo_native_query_embedding.py
#
# graphiti#20 (Phase 2): route the query embedding through drevo.semantic.embed
# so the drevo-native path drops the client embedder without losing Cypher-level
# filters (the server-embedded vector flows through the existing filtered cosine
# search unchanged).

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.capabilities import (
    GraphCapabilities,
    uses_native_query_embedding,
)
from graphiti_core.search.search_utils import resolve_query_vector

try:
    from graphiti_core.driver.drevo_driver import DrevoDriver

    HAS_NEO4J = True
except ImportError:
    DrevoDriver = None
    HAS_NEO4J = False


class TestNativeQueryEmbeddingCapability:
    def test_default_is_false(self):
        assert GraphCapabilities().native_query_embedding is False

    def test_helper_reads_capability(self):
        on = SimpleNamespace(capabilities=GraphCapabilities(native_query_embedding=True))
        off = SimpleNamespace(capabilities=GraphCapabilities(native_query_embedding=False))
        assert uses_native_query_embedding(on) is True
        assert uses_native_query_embedding(off) is False

    def test_helper_tolerates_missing_capabilities(self):
        assert uses_native_query_embedding(SimpleNamespace()) is False


class TestResolveQueryVector:
    """The single seam that decides where the query vector comes from."""

    @pytest.mark.asyncio
    async def test_explicit_vector_passthrough(self):
        embedder = MagicMock()
        embedder.create = AsyncMock(side_effect=AssertionError('must not embed'))
        driver = SimpleNamespace(capabilities=GraphCapabilities(native_query_embedding=True))
        out = await resolve_query_vector(driver, embedder, 'q', [0.9, 0.1])
        assert out == [0.9, 0.1]

    @pytest.mark.asyncio
    async def test_native_uses_driver_embed_query_not_client(self):
        embedder = MagicMock()
        embedder.create = AsyncMock(
            side_effect=AssertionError('client embedder must not be called')
        )
        driver = SimpleNamespace(
            capabilities=GraphCapabilities(native_query_embedding=True),
            embed_query=AsyncMock(return_value=[1.0, 2.0, 3.0]),
        )
        out = await resolve_query_vector(driver, embedder, 'hello\nworld', None)
        assert out == [1.0, 2.0, 3.0]
        # newlines normalized before embedding
        driver.embed_query.assert_awaited_once_with('hello world')

    @pytest.mark.asyncio
    async def test_non_native_uses_client_embedder(self):
        embedder = MagicMock()
        embedder.create = AsyncMock(return_value=[0.5, 0.5])
        driver = SimpleNamespace(capabilities=GraphCapabilities(native_query_embedding=False))
        out = await resolve_query_vector(driver, embedder, 'hi', None)
        assert out == [0.5, 0.5]
        embedder.create.assert_awaited_once()


@pytest.mark.skipif(not HAS_NEO4J, reason='neo4j driver package is not installed')
class TestDrevoEmbedQuery:
    def _make_driver(self) -> 'DrevoDriver':
        with patch('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase') as mock_gdb:
            mock_gdb.driver.return_value = MagicMock()
            return DrevoDriver(uri='bolt://localhost:7687', user='neo4j', password='password')

    @pytest.mark.asyncio
    async def test_embed_query_calls_semantic_embed(self):
        driver = self._make_driver()
        exec_mock = AsyncMock(return_value=([{'vector': [0.1, 0.2, 0.3]}], None, None))
        with patch.object(driver, 'execute_query', exec_mock):
            vector = await driver.embed_query('anxious thoughts')
        assert vector == [0.1, 0.2, 0.3]
        query = exec_mock.await_args.args[0]
        assert 'drevo.semantic.embed' in query

    @pytest.mark.asyncio
    async def test_negotiation_enables_query_embedding_when_embed_probe_succeeds(self):
        driver = self._make_driver()
        info_record = {'embedder_present': True, 'model': 'm', 'dimension': 1536}

        async def fake_exec(query, *args, **kwargs):
            if 'semantic.info' in query:
                return ([info_record], None, None)
            if 'semantic.embed' in query:
                return ([{'vector': [0.0] * 4}], None, None)
            return ([{'label': 'Entity'}], None, None)  # register / registerRel

        with patch.object(driver, 'execute_query', AsyncMock(side_effect=fake_exec)):
            await driver.build_indices_and_constraints()

        assert driver.capabilities.native_auto_embedding is True
        assert driver.capabilities.native_query_embedding is True

    @pytest.mark.asyncio
    async def test_query_embedding_stays_off_when_embed_absent(self):
        """Old drevo with auto-embed writes but no semantic.embed: writes stay
        native, but the query path must fall back to the client embedder."""
        driver = self._make_driver()
        info_record = {'embedder_present': True, 'model': 'm', 'dimension': 1536}

        async def fake_exec(query, *args, **kwargs):
            if 'semantic.info' in query:
                return ([info_record], None, None)
            if 'semantic.embed' in query:
                raise Exception('no such procedure `drevo.semantic.embed`')
            return ([{'label': 'Entity'}], None, None)

        with patch.object(driver, 'execute_query', AsyncMock(side_effect=fake_exec)):
            await driver.build_indices_and_constraints()

        assert driver.capabilities.native_auto_embedding is True
        assert driver.capabilities.native_query_embedding is False
