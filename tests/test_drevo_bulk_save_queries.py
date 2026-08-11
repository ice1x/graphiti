"""The DREVO bulk-save queries must avoid Neo4j-only Cypher drevo can't parse.

drevo's Cypher does not implement Neo4j-5 dynamic labels (``SET n:$(node.labels)``)
or the ``db.create.set{Node,Relationship}VectorProperty`` procedures. Falling into
the default (Neo4j) branch therefore raises a ``CypherSyntaxError`` on the entity
node/edge bulk writes. These tests pin the dedicated DREVO branch.
"""

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_save_bulk_query
from graphiti_core.models.nodes.node_db_queries import get_entity_node_save_bulk_query

_NODES = [{'uuid': 'n1', 'name': 'Ada', 'group_id': 'g', 'labels': ['Entity', 'Person']}]


def test_drevo_entity_node_query_avoids_neo4j_only_constructs():
    query = get_entity_node_save_bulk_query(GraphProvider.DREVO, _NODES)

    assert isinstance(query, str)
    # No dynamic labels and no vector-property procedure.
    assert '$(' not in query
    assert 'setNodeVectorProperty' not in query
    # One map assignment stores every property (embedding as a plain list).
    assert 'SET n = node' in query
    assert 'MERGE (n:Entity {uuid: node.uuid})' in query


def test_drevo_entity_edge_query_avoids_neo4j_only_constructs():
    query = get_entity_edge_save_bulk_query(GraphProvider.DREVO)

    assert isinstance(query, str)
    assert '$(' not in query
    assert 'setRelationshipVectorProperty' not in query
    assert 'SET e = edge' in query
    assert 'MERGE (source)-[e:RELATES_TO {uuid: edge.uuid}]->(target)' in query
