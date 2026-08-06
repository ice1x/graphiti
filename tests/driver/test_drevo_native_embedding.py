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

# Running tests: pytest -xvs tests/driver/test_drevo_native_embedding.py
#
# Covers graphiti#16 "drevo-native auto-embedding" foundation slice: capability
# negotiation, semantic-target registration, and the client-side write-embedding
# no-op that lets drevo own stored embeddings.

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.capabilities import (
    GraphCapabilities,
    uses_native_auto_embedding,
)
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_save_query
from graphiti_core.models.nodes.node_db_queries import get_entity_node_save_query

try:
    from graphiti_core.driver.drevo_driver import DrevoDriver

    HAS_NEO4J = True
except ImportError:
    DrevoDriver = None
    HAS_NEO4J = False


# ---------------------------------------------------------------------------
# Capability flag + helper
# ---------------------------------------------------------------------------


class TestNativeAutoEmbeddingCapability:
    def test_default_is_false(self):
        """A bare capability set claims no native auto-embedding."""
        assert GraphCapabilities().native_auto_embedding is False

    def test_helper_reads_driver_capability(self):
        on = SimpleNamespace(capabilities=GraphCapabilities(native_auto_embedding=True))
        off = SimpleNamespace(capabilities=GraphCapabilities(native_auto_embedding=False))
        assert uses_native_auto_embedding(on) is True
        assert uses_native_auto_embedding(off) is False

    def test_helper_tolerates_missing_capabilities(self):
        """A driver without a capabilities attribute is treated as non-native."""
        assert uses_native_auto_embedding(SimpleNamespace()) is False


# ---------------------------------------------------------------------------
# DREVO save queries must not emit Neo4j vector procedures and must store the
# embedding as a plain list (drevo supports neither setNodeVectorProperty nor
# vecf32; it stores/reads name_embedding as a native array).
# ---------------------------------------------------------------------------


class TestDrevoSaveQueries:
    def test_node_save_query_uses_plain_set(self):
        query = get_entity_node_save_query(GraphProvider.DREVO, 'Person')
        assert 'setNodeVectorProperty' not in query
        assert 'vecf32' not in query
        assert 'SET n = $entity_data' in query

    def test_edge_save_query_uses_plain_set(self):
        query = get_entity_edge_save_query(GraphProvider.DREVO)
        assert 'setRelationshipVectorProperty' not in query
        assert 'vecf32' not in query


# ---------------------------------------------------------------------------
# Negotiation + registration on the DrevoDriver
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_NEO4J, reason='neo4j driver package is not installed')
class TestDrevoNegotiation:
    def _make_driver(self) -> 'DrevoDriver':
        with patch('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase') as mock_gdb:
            mock_gdb.driver.return_value = MagicMock()
            return DrevoDriver(uri='bolt://localhost:7687', user='neo4j', password='password')

    def test_capability_is_instance_level_and_off_by_default(self):
        """Negotiation must flip a per-connection flag, never mutate the shared class."""
        driver = self._make_driver()
        assert driver.capabilities.native_auto_embedding is False
        # Mutating one instance must not leak into the class default or a sibling.
        driver.capabilities.native_auto_embedding = True
        assert GraphCapabilities().native_auto_embedding is False

    @pytest.mark.asyncio
    async def test_negotiation_enables_and_registers_when_embedder_present(self):
        driver = self._make_driver()
        info_record = {
            'embedder_present': True,
            'model': 'text-embedding-3-small',
            'dimension': 1536,
        }
        exec_mock = AsyncMock(return_value=([info_record], None, None))
        with patch.object(driver, 'execute_query', exec_mock):
            await driver.build_indices_and_constraints()

        assert driver.capabilities.native_auto_embedding is True
        emitted = ' '.join(str(call.args[0]) for call in exec_mock.await_args_list)
        assert 'drevo.semantic.info' in emitted
        assert 'drevo.semantic.register' in emitted
        assert 'drevo.semantic.registerRel' in emitted

    @pytest.mark.asyncio
    async def test_negotiation_stays_off_when_procedure_absent(self):
        """Old drevo (no semantic procedures) → capability stays off, no registration."""
        driver = self._make_driver()
        exec_mock = AsyncMock(side_effect=Exception('no such procedure `drevo.semantic.info`'))
        with patch.object(driver, 'execute_query', exec_mock):
            await driver.build_indices_and_constraints()

        assert driver.capabilities.native_auto_embedding is False
        emitted = ' '.join(str(call.args[0]) for call in exec_mock.await_args_list)
        assert 'register' not in emitted

    @pytest.mark.asyncio
    async def test_negotiation_stays_off_when_embedder_absent(self):
        """Procedures exist but no embedder configured → no auto-embedding, no registration."""
        driver = self._make_driver()
        info_record = {'embedder_present': False, 'model': None, 'dimension': None}
        exec_mock = AsyncMock(return_value=([info_record], None, None))
        with patch.object(driver, 'execute_query', exec_mock):
            await driver.build_indices_and_constraints()

        assert driver.capabilities.native_auto_embedding is False
        emitted = ' '.join(str(call.args[0]) for call in exec_mock.await_args_list)
        assert 'register' not in emitted


# ---------------------------------------------------------------------------
# Write-embedding no-op: with native auto-embedding on, the bulk ingest path
# must not call the embedder and must not write an embedding property (so drevo
# stays authoritative and an unchanged-text update never nulls the vector).
# ---------------------------------------------------------------------------


def _entity_node():
    from graphiti_core.nodes import EntityNode

    return EntityNode(name='Alice', group_id='g', labels=[], summary='a person')


def _entity_edge(source_uuid: str, target_uuid: str):
    from graphiti_core.edges import EntityEdge

    return EntityEdge(
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        name='RELATES_TO',
        fact='Alice mentors Bob',
        group_id='g',
        created_at=datetime.now(timezone.utc),
    )


def _drevo_driver_stub(native: bool):
    return SimpleNamespace(
        provider=GraphProvider.DREVO,
        graph_operations_interface=None,
        capabilities=GraphCapabilities(native_auto_embedding=native),
    )


class TestBulkWriteNoOp:
    @pytest.mark.asyncio
    async def test_native_skips_embedder_and_omits_embedding_property(self):
        from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx

        node = _entity_node()
        edge = _entity_edge(node.uuid, node.uuid)
        embedder = MagicMock()
        embedder.create = AsyncMock(side_effect=AssertionError('embedder must not be called'))
        embedder.create_batch = AsyncMock(side_effect=AssertionError('embedder must not be called'))
        tx = MagicMock()
        tx.run = AsyncMock(return_value=None)

        await add_nodes_and_edges_bulk_tx(
            tx, [], [], [node], [edge], embedder, _drevo_driver_stub(native=True)
        )

        assert node.name_embedding is None
        assert edge.fact_embedding is None
        # Inspect the node/edge dicts handed to tx.run — no embedding key survives.
        written = [c.kwargs for c in tx.run.await_args_list]
        node_rows = [row for kw in written for row in kw.get('nodes', [])]
        edge_rows = [row for kw in written for row in kw.get('entity_edges', [])]
        assert node_rows and all('name_embedding' not in row for row in node_rows)
        assert edge_rows and all('fact_embedding' not in row for row in edge_rows)

    @pytest.mark.asyncio
    async def test_non_native_generates_embeddings(self):
        from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx

        node = _entity_node()
        edge = _entity_edge(node.uuid, node.uuid)
        embedder = MagicMock()
        embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
        tx = MagicMock()
        tx.run = AsyncMock(return_value=None)

        await add_nodes_and_edges_bulk_tx(
            tx, [], [], [node], [edge], embedder, _drevo_driver_stub(native=False)
        )

        assert node.name_embedding == [0.1, 0.2, 0.3]
        assert edge.fact_embedding == [0.1, 0.2, 0.3]


class TestSingleSaveDropsEmbedding:
    """A single save must not carry an embedding property in native mode, so an
    unchanged-text update can never null drevo's server-maintained vector."""

    @pytest.mark.asyncio
    async def test_node_save_omits_name_embedding_when_native(self):
        node = _entity_node()
        node.name_embedding = [0.1, 0.2]  # stale client value must not be written
        exec_mock = AsyncMock(return_value=([{'uuid': node.uuid}], None, None))
        driver = SimpleNamespace(
            provider=GraphProvider.DREVO,
            graph_operations_interface=None,
            capabilities=GraphCapabilities(native_auto_embedding=True),
            execute_query=exec_mock,
        )
        await node.save(driver)
        entity_data = exec_mock.await_args.kwargs['entity_data']
        assert 'name_embedding' not in entity_data

    @pytest.mark.asyncio
    async def test_node_save_keeps_name_embedding_when_not_native(self):
        node = _entity_node()
        node.name_embedding = [0.1, 0.2]
        exec_mock = AsyncMock(return_value=([{'uuid': node.uuid}], None, None))
        driver = SimpleNamespace(
            provider=GraphProvider.DREVO,
            graph_operations_interface=None,
            capabilities=GraphCapabilities(native_auto_embedding=False),
            execute_query=exec_mock,
        )
        await node.save(driver)
        entity_data = exec_mock.await_args.kwargs['entity_data']
        assert entity_data['name_embedding'] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_edge_save_omits_fact_embedding_when_native(self):
        node = _entity_node()
        edge = _entity_edge(node.uuid, node.uuid)
        edge.fact_embedding = [0.3, 0.4]
        exec_mock = AsyncMock(return_value=([{'uuid': edge.uuid}], None, None))
        driver = SimpleNamespace(
            provider=GraphProvider.DREVO,
            graph_operations_interface=None,
            capabilities=GraphCapabilities(native_auto_embedding=True),
            execute_query=exec_mock,
        )
        await edge.save(driver)
        edge_data = exec_mock.await_args.kwargs['edge_data']
        assert 'fact_embedding' not in edge_data
