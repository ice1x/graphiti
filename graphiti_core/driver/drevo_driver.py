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

import logging
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager

from graphiti_core.driver.capabilities import GraphCapabilities
from graphiti_core.driver.drevo_search_interface import DrevoSearchInterface
from graphiti_core.driver.driver import GraphProvider, _SessionTransaction
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.driver.query_executor import Transaction

logger = logging.getLogger(__name__)

# drevo Phase 21 semantic control plane (ice1x/drevo#251/#266/#267). Registering a
# target makes drevo embed the text property into the vector property server-side
# on write, so graphiti can skip client-side embedding for storage. Registration is
# idempotent (re-registering the same target is a no-op/update). There is no
# deregister procedure, so these are registered once at index-build time.
_SEMANTIC_INFO_QUERY = (
    'CALL drevo.semantic.info() '
    'YIELD embedder_present, model, dimension '
    'RETURN embedder_present, model, dimension'
)
_SEMANTIC_REGISTER_QUERIES = (
    "CALL drevo.semantic.register('Entity', 'name', 'name_embedding', 'auto') "
    'YIELD label RETURN label',
    "CALL drevo.semantic.register('Community', 'name', 'name_embedding', 'auto') "
    'YIELD label RETURN label',
    "CALL drevo.semantic.registerRel('RELATES_TO', 'fact', 'fact_embedding', 'auto') "
    'YIELD label RETURN label',
)
# Standalone server-side query embedding (drevo#272). Lets graphiti embed the
# query text server-side and feed the vector to its existing *filtered* cosine
# Cypher — no client embedder, no lost filters.
_SEMANTIC_EMBED_QUERY = 'CALL drevo.semantic.embed($text) YIELD vector RETURN vector'

# Server-version compatibility gate. DrevoDriver has accumulated hard requirements
# on the drevo server: datetime Bolt params (graphiti#21), dedicated bulk save
# Cypher with dynamic labels/vector props (graphiti#25/#26), the managed-tx fix
# (drevo#298/#299 → v0.0.15), and the empty-UNWIND+write fix (drevo#300/#301 →
# v0.0.16). v0.0.16 is therefore the lowest drevo this connector can drive
# correctly. drevo exposes its own version via CALL drevo.info() (drevo#303/#304,
# available from v0.0.18); a drevo too old to have that procedure cannot be
# verified and is only warned about.
MINIMUM_DREVO_VERSION = '0.0.16'
_INFO_QUERY = (
    'CALL drevo.info() '
    'YIELD version, git_sha, build_date, protocol '
    'RETURN version, git_sha, build_date, protocol'
)


def _parse_semver(version: str) -> tuple[int, ...]:
    """Parse a dotted version string like ``0.0.18`` (or ``v0.0.18``) into a
    comparable integer tuple. Parsing stops at the first non-numeric component, so
    a build/pre-release suffix (``0.0.18-rc1`` → ``(0, 0, 18)``) degrades to a
    best-effort comparison instead of raising."""
    parts: list[int] = []
    for chunk in version.strip().lstrip('vV').split('.'):
        digits = ''
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


class DrevoDriver(Neo4jDriver):
    """Connector for `drevo <https://github.com/ice1x/drevo>`_.

    drevo speaks the Bolt wire protocol on port 7687 and accepts a Cypher subset,
    so the connector reuses the Neo4j Bolt machinery (connection, sessions, query
    execution, CRUD operations) and only diverges where drevo differs:

    * **No index DDL.** drevo's Cypher subset has no ``CREATE INDEX/FULLTEXT/VECTOR``
      statements — indexing is managed out-of-band — so index build/teardown are
      no-ops rather than emitting Neo4j index procedures.
    * **No explicit transactions.** The Bolt handshake advertises only
      ``RUN/PULL/DISCARD/RESET`` (no ``BEGIN/COMMIT``), so ``transaction()`` runs
      in immediate mode instead of Neo4j's begin/commit path.

    Search is served by :class:`DrevoSearchInterface` (set on the instance below
    and honored by ``graphiti_core.search.search_utils``): vector similarity uses
    drevo's native ``cosine_similarity`` scalar (drevo#202), with a library-side
    fallback for older builds; fulltext is an interim lexical match (no BM25 over
    Cypher yet — drevo#208); and BFS / rerankers / community embeddings fall back to
    the Neo4j-compatible inline Cypher that drevo's Cypher subset supports.
    """

    provider = GraphProvider.DREVO

    # Default capability template. Copied to an instance attribute in __init__ so
    # per-connection negotiation (native auto-embedding) never mutates the shared
    # class object.
    capabilities = GraphCapabilities(
        supports_transactions=False,
        # Native BM25 node full-text via fts.search (drevo#208/#227) and native
        # vector similarity via cosine_similarity (drevo#202); DrevoSearchInterface
        # falls back to lexical / library-side ranking on older builds.
        supports_native_fulltext_search=True,
        supports_native_vector_search=True,
        supports_vector_index=False,
        # Off until negotiated: build_indices_and_constraints probes
        # drevo.semantic.info and flips this on only when an embedder is configured.
        native_auto_embedding=False,
    )

    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
        database: str = 'neo4j',
    ) -> None:
        super().__init__(uri, user, password, database)
        self.search_interface = DrevoSearchInterface()
        # Per-instance copy so negotiation is scoped to this connection.
        self.capabilities = self.capabilities.model_copy()

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        """drevo manages indexes out-of-band (no index DDL), but this is the setup
        hook where we verify server compatibility and negotiate server-side
        auto-embedding / register the embedding targets."""
        await self._negotiate_drevo_version()
        await self._negotiate_native_auto_embedding()
        return None

    async def _negotiate_drevo_version(self) -> None:
        """Assert the connected drevo meets the connector's minimum version.

        Probes ``CALL drevo.info()`` (drevo#303/#304, v0.0.18+):

        * reports ``version < MINIMUM_DREVO_VERSION`` → raise ``RuntimeError`` with a
          clear upgrade message (fail fast rather than emit writes an older drevo
          mishandles);
        * procedure missing / unreadable (drevo older than v0.0.18, which introduced
          ``drevo.info``) or no ``version`` field → cannot verify, log a warning and
          continue. Such a drevo may still be below the minimum in a way this check
          can't detect, hence the warning."""
        try:
            records, _, _ = await self.execute_query(_INFO_QUERY, routing_='r')
        except Exception as e:  # noqa: BLE001 - any failure means "can't verify"
            logger.warning(
                'Could not verify drevo version (drevo.info unavailable: %s). '
                'Ensure the connected drevo is >= %s.',
                e,
                MINIMUM_DREVO_VERSION,
            )
            return

        version = records[0].get('version') if records else None
        if not version:
            logger.warning(
                'drevo.info returned no version; cannot verify compatibility. '
                'Ensure the connected drevo is >= %s.',
                MINIMUM_DREVO_VERSION,
            )
            return

        if _parse_semver(version) < _parse_semver(MINIMUM_DREVO_VERSION):
            raise RuntimeError(
                f'Connected drevo is version {version}, but this graphiti connector '
                f'requires drevo >= {MINIMUM_DREVO_VERSION}. Upgrade drevo to >= '
                f'{MINIMUM_DREVO_VERSION}.'
            )
        logger.info('drevo version %s satisfies minimum %s', version, MINIMUM_DREVO_VERSION)

    async def _negotiate_native_auto_embedding(self) -> None:
        """Detect drevo's Phase 21 server-side embedder and register embedding
        targets. Sets ``capabilities.native_auto_embedding`` only when drevo reports
        a configured embedder AND every target registers successfully. Any failure
        (old drevo without the semantic procedures, no embedder configured, a
        connection error) leaves the capability off — the connector then falls back
        to client-side embedding (today's behavior)."""
        try:
            records, _, _ = await self.execute_query(_SEMANTIC_INFO_QUERY, routing_='r')
        except Exception as e:  # noqa: BLE001 - any failure means "not available"
            logger.debug('drevo native auto-embedding unavailable (semantic.info failed): %s', e)
            return

        if not records or not records[0].get('embedder_present'):
            logger.debug('drevo has no server-side embedder configured; using client embedder')
            return

        for query in _SEMANTIC_REGISTER_QUERIES:
            try:
                await self.execute_query(query)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    'drevo semantic target registration failed, staying client-side: %s', e
                )
                return

        self.capabilities.native_auto_embedding = True
        logger.info('drevo native auto-embedding enabled; skipping client-side write embeddings')

        # Separately probe standalone query embedding (drevo#272). A build can
        # auto-embed writes without exposing drevo.semantic.embed, so the query
        # path is negotiated independently and falls back to the client embedder.
        try:
            records, _, _ = await self.execute_query(
                _SEMANTIC_EMBED_QUERY, text='drevo native query embedding probe', routing_='r'
            )
        except Exception as e:  # noqa: BLE001
            logger.debug('drevo standalone query embedding unavailable (semantic.embed): %s', e)
            return

        if records and records[0].get('vector'):
            self.capabilities.native_query_embedding = True
            logger.info('drevo native query embedding enabled; skipping client-side query embedder')

    async def embed_query(self, text: str) -> list[float]:
        """Embed query text server-side via ``drevo.semantic.embed`` (drevo#272)."""
        records, _, _ = await self.execute_query(_SEMANTIC_EMBED_QUERY, text=text, routing_='r')
        if not records or records[0].get('vector') is None:
            raise ValueError('drevo.semantic.embed returned no vector')
        return records[0]['vector']

    def delete_all_indexes(self) -> Coroutine:
        """No-op: nothing to drop, and drevo exposes no index-management procedures."""

        async def _noop() -> None:
            return None

        return _noop()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Transaction]:
        """Immediate-mode transaction — drevo's Bolt has no BEGIN/COMMIT, so
        queries execute as they run against a plain session."""
        session = self.session()
        try:
            yield _SessionTransaction(session)
        finally:
            await session.close()
