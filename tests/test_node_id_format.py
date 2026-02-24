"""Hermetic tests for Node.node_id formatting (no model/GPU/network required).

Regression coverage for issue #78: MLP reconstruction-error nodes used to be
assigned node_id = f"0_{layer}_{pos}", which (a) collided with a CLT/feature
node at layer 0 / feature {layer} / ctx {pos}, and (b) was misparsed by
decode_url_features (which does map(int, node_id.split("_"))) as layer=0,
feature={layer}. Error nodes now use f"{layer}_-1_{pos}", matching the CLT
format, staying collision-free (feature -1 is never a real feature index), and
parsing back to (layer, -1, pos).

The same layer/index overlap also affected jsNodeId (consumed by the
Neuronpedia frontend): error nodes used f"{layer}_{pos}-{reverse_ctx_idx}"
while feature/CLT nodes use f"{layer}_{feat_idx}-{reverse_ctx_idx}", so a
layer-1 error node and a layer-1 feature node with feat_idx 1 both rendered
"1_1-0" and aliased each other on hover. Error nodes now use feature slot -1
in jsNodeId too, with the position as the trailing discriminator
(f"{layer}_-1-{pos}"), so error nodes at the same layer but different
positions stay distinct.

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


def test_error_node_jsnodeid_does_not_collide_with_feature_node():
    # jsNodeId collision (issue #78 follow-up): error nodes used
    # f"{layer}_{pos}-{reverse_ctx_idx}" while feature/CLT nodes use
    # f"{layer}_{feat_idx}-{reverse_ctx_idx}". A layer-1 error node at pos 1 and
    # a layer-1 feature node with feat_idx 1 both rendered "1_1-0", which caused
    # the Neuronpedia UI hover-bug (hovering one node highlighted the other).
    error_node = Node.error_node(layer=1, pos=1)
    feature_node = Node.feature_node(layer=1, pos=1, feat_idx=1)
    assert error_node.jsNodeId != feature_node.jsNodeId
    # The specific historical collision string must no longer be shared.
    assert not (error_node.jsNodeId == feature_node.jsNodeId == "1_1-0")


def test_error_node_jsnodeid_never_aliases_any_feature_feat_idx():
    # For a fixed layer/pos, no feature index may reproduce the error node's
    # jsNodeId. -1 is never a valid feat_idx, so this holds by construction.
    layer, pos = 4, 2
    error_js = Node.error_node(layer=layer, pos=pos).jsNodeId
    for feat_idx in range(50):
        assert Node.feature_node(layer=layer, pos=pos, feat_idx=feat_idx).jsNodeId != error_js


def test_error_node_jsnodeids_unique_within_a_layer():
    # A real graph has many error nodes at the same layer, one per token
    # position. Their jsNodeIds must stay distinct: a layer's error nodes
    # differ only in pos, so the position has to appear in jsNodeId. (A constant
    # trailing slot collapsed every error node at a layer onto one jsNodeId,
    # which re-created the Neuronpedia hover-bug among the error nodes
    # themselves.)
    layer = 7
    js_ids = [Node.error_node(layer=layer, pos=pos).jsNodeId for pos in range(20)]
    assert len(js_ids) == len(set(js_ids))


def test_mixed_jsnodeids_are_unique():
    nodes = [
        Node.error_node(layer=0, pos=1),
        Node.error_node(layer=1, pos=1),
        Node.error_node(layer=2, pos=1),
        Node.feature_node(layer=0, pos=1, feat_idx=1),
        Node.feature_node(layer=1, pos=1, feat_idx=1),
        Node.feature_node(layer=2, pos=1, feat_idx=2),
        Node.token_node(pos=1, vocab_idx=5),
        Node.logit_node(pos=1, vocab_idx=5, token="Austin", num_layers=2),
    ]
    js_ids = [n.jsNodeId for n in nodes]
    assert len(js_ids) == len(set(js_ids))
