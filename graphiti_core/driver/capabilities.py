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

from pydantic import BaseModel


class GraphCapabilities(BaseModel):
    """What a graph backend supports natively.

    Connectors declare their capabilities so the search/index layer can branch on
    a *capability* rather than on backend identity (``provider == GraphProvider.X``).
    A backend that lacks a native capability can then be routed through a fallback
    path (external vector-store overlay, in-library BM25) instead of emitting
    backend-specific procedures it does not implement.

    Defaults are conservative — a bare capability set claims nothing native, which
    is the safe assumption for a minimal Bolt/Cypher backend.
    """

    supports_transactions: bool = False
    """Real commit/rollback semantics (as opposed to immediate-mode / autocommit)."""

    supports_native_fulltext_search: bool = False
    """Native fulltext search wired through the connector's search operations
    (e.g. Neo4j ``db.index.fulltext.*``)."""

    supports_native_vector_search: bool = False
    """Native vector similarity search wired through the connector's search
    operations."""

    supports_vector_index: bool = False
    """Can create/manage a persistent vector index via the query language."""

    native_auto_embedding: bool = False
    """Server embeds configured node/edge text properties on write (and keeps the
    vector index in sync), so the connector can skip generating those embeddings
    client-side for storage. Negotiated per-connection: a connector flips this on
    only after confirming the backend has a configured embedder (e.g. drevo's
    ``drevo.semantic.info`` reporting ``embedder_present``). When off, embeddings
    are generated client-side as before (the safe fallback)."""

    native_query_embedding: bool = False
    """Server can embed **query** text on demand (e.g. drevo's
    ``drevo.semantic.embed(text) YIELD vector``), so the connector can obtain the
    query vector server-side and feed it to its existing filtered similarity Cypher
    instead of calling a client-side embedder. Negotiated separately from
    :attr:`native_auto_embedding` because a backend can auto-embed writes without
    exposing a standalone query-embed procedure. When off, query embedding is done
    client-side (the safe fallback)."""


def uses_native_auto_embedding(driver: object) -> bool:
    """Whether ``driver`` embeds stored node/edge properties server-side.

    Duck-typed on ``driver.capabilities`` so it works for any connector (and for
    lightweight stubs) without importing the driver types. Returns ``False`` when
    the driver exposes no capabilities, keeping the client-side embedding path as
    the default.
    """
    capabilities = getattr(driver, 'capabilities', None)
    return bool(capabilities is not None and getattr(capabilities, 'native_auto_embedding', False))


def uses_native_query_embedding(driver: object) -> bool:
    """Whether ``driver`` can embed query text server-side (a standalone embed
    procedure), letting the connector skip the client-side query embedder.

    Duck-typed on ``driver.capabilities`` like :func:`uses_native_auto_embedding`;
    returns ``False`` when the driver exposes no capabilities.
    """
    capabilities = getattr(driver, 'capabilities', None)
    return bool(capabilities is not None and getattr(capabilities, 'native_query_embedding', False))
