"""Hermetic tests for Node.node_id formatting (no model/GPU/network required).

Regression coverage for issue #78: MLP reconstruction-error nodes used to be
assigned node_id = f"0_{layer}_{pos}", which (a) collided with a CLT/feature
node at layer 0 / feature {layer} / ctx {pos}, and (b) was misparsed by
decode_url_features (which does map(int, node_id.split("_"))) as layer=0,
feature={layer}. Error nodes now use f"{layer}_-1_{pos}", matching the CLT
format, staying collision-free (feature -1 is never a real feature index), and
parsing back to (layer, -1, pos).

These tests import only the pure-pydantic Node model and require no torch.
"""

from circuit_tracer.frontend.graph_models import Node


def test_error_node_id_uses_layer_negone_pos_format():
    for layer in range(5):
        pos = 1
        node = Node.error_node(layer=layer, pos=pos)
        assert node.node_id == f"{layer}_-1_{pos}"
        assert node.feature == -1


def test_error_node_does_not_collide_with_feature_node():
    # The historical collision: a layer-1 error node was "0_1_1", which is
    # identical to a CLT node at layer 0, feature 1, ctx 1.
    error_node = Node.error_node(layer=1, pos=1)
    feature_node = Node.feature_node(layer=0, pos=1, feat_idx=1)
    assert error_node.node_id != feature_node.node_id


def test_error_node_id_parses_like_decode_url_features():
    # decode_url_features.py does: layer, feature_idx, pos = map(int, id.split("_"))
    layer, pos = 3, 7
    node = Node.error_node(layer=layer, pos=pos)
    parsed_layer, parsed_feature, parsed_pos = map(int, node.node_id.split("_"))
    assert (parsed_layer, parsed_feature, parsed_pos) == (layer, -1, pos)


def test_mixed_node_ids_are_unique():
    nodes = [
        Node.error_node(layer=0, pos=1),
        Node.error_node(layer=1, pos=1),
        Node.error_node(layer=2, pos=1),
        Node.feature_node(layer=0, pos=1, feat_idx=1),
        Node.feature_node(layer=1, pos=1, feat_idx=928),
        Node.feature_node(layer=2, pos=1, feat_idx=0),
    ]
    node_ids = [n.node_id for n in nodes]
    assert len(node_ids) == len(set(node_ids))
