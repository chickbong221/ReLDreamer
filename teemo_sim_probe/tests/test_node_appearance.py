"""Per-camera bounding boxes and patch-coverage grids.

The packed box is what the model reads visibility back off, so it has to be
exact in normalised coordinates, and the grid has to carry partial cells rather
than a hard binary mask so a node that only partly fills a DINO patch
contributes proportionally.
"""

import unittest

import numpy as np

from teemo_sim_probe.core.node_builder import fill_appearance
from teemo_sim_probe.core.schema import Node


def _node(seg_ids=()):
    return Node(
        node_id="a", node_type="object", name="a",
        segmentation_ids=list(seg_ids),
    )


def _seg(pixels, size=8):
    """8x8 segmentation with ``pixels`` mapping seg id -> (rows, cols)."""
    seg = np.zeros((size, size), np.int32)
    for seg_id, (rows, cols) in pixels.items():
        seg[np.ix_(rows, cols)] = seg_id
    return seg


def _fill(nodes, *segs, grid=4):
    fill_appearance(nodes, list(segs), grid)


class BoundingBoxTests(unittest.TestCase):

    def test_box_is_normalised_with_exclusive_maxima(self):
        nodes = {"a": _node([7])}
        _fill(nodes, _seg({7: ([2, 3], [4, 5, 6])}))
        np.testing.assert_allclose(
            nodes["a"].bbox[0], [4 / 8, 7 / 8, 2 / 8, 4 / 8])

    def test_one_pixel_node_has_nonzero_extent(self):
        nodes = {"a": _node([7])}
        _fill(nodes, _seg({7: ([0], [0])}))
        x0, x1, y0, y1 = nodes["a"].bbox[0]
        self.assertGreater(x1 - x0, 0.0)
        self.assertGreater(y1 - y0, 0.0)

    def test_full_frame_node_spans_the_unit_square(self):
        nodes = {"a": _node([7])}
        _fill(nodes, _seg({7: (range(8), range(8))}))
        np.testing.assert_allclose(nodes["a"].bbox[0], [0.0, 1.0, 0.0, 1.0])

    def test_absent_node_keeps_a_zero_row(self):
        nodes = {"a": _node([9])}
        _fill(nodes, _seg({7: ([0], [0])}))
        self.assertTrue((nodes["a"].bbox == 0).all())
        self.assertTrue((nodes["a"].patch_weights == 0).all())

    def test_node_without_segmentation_ids_is_skipped(self):
        nodes = {"a": _node()}
        _fill(nodes, _seg({7: ([0], [0])}))
        self.assertTrue((nodes["a"].bbox == 0).all())

    def test_several_segmentation_ids_union_into_one_box(self):
        nodes = {"a": _node([7, 9])}
        _fill(nodes, _seg({7: ([0], [0]), 9: ([5], [6])}))
        np.testing.assert_allclose(
            nodes["a"].bbox[0], [0 / 8, 7 / 8, 0 / 8, 6 / 8])


class CameraIndependenceTests(unittest.TestCase):
    """Every camera carries its own box and coverage; nothing is fused."""

    def test_each_camera_gets_its_own_row(self):
        nodes = {"a": _node([7])}
        _fill(nodes,
              _seg({7: ([0], [0])}),
              _seg({7: ([4, 5], [4, 5])}))
        self.assertEqual(nodes["a"].bbox.shape, (2, 4))
        np.testing.assert_allclose(
            nodes["a"].bbox[0], [0.0, 1 / 8, 0.0, 1 / 8])
        np.testing.assert_allclose(
            nodes["a"].bbox[1], [4 / 8, 6 / 8, 4 / 8, 6 / 8])

    def test_seen_by_one_camera_only_zeroes_the_other(self):
        nodes = {"a": _node([7])}
        _fill(nodes, _seg({7: ([0], [0])}), _seg({}))
        self.assertGreater(nodes["a"].bbox[0, 1] - nodes["a"].bbox[0, 0], 0.0)
        self.assertTrue((nodes["a"].bbox[1] == 0).all())
        self.assertGreater(nodes["a"].patch_weights[0].sum(), 0.0)
        self.assertEqual(nodes["a"].patch_weights[1].sum(), 0.0)

    def test_seen_by_neither_camera_zeroes_both(self):
        nodes = {"a": _node([7])}
        _fill(nodes, _seg({}), _seg({}))
        self.assertTrue((nodes["a"].bbox == 0).all())


class CoverageGridTests(unittest.TestCase):

    def test_partial_cells_land_between_zero_and_full(self):
        nodes = {"a": _node([7])}
        _fill(nodes, _seg({7: ([2, 3], [4, 5, 6])}))
        w = nodes["a"].patch_weights[0].reshape(4, 4)
        self.assertEqual(nodes["a"].patch_weights.shape, (1, 16))
        self.assertAlmostEqual(float(w[1, 2]), 1.0)    # 2x2 cell fully inside
        self.assertAlmostEqual(float(w[1, 3]), 0.5)    # half the cell
        self.assertAlmostEqual(float(w.sum() - w[1, 2] - w[1, 3]), 0.0)

    def test_grid_resolution_follows_the_patch_grid(self):
        nodes = {"a": _node([7])}
        _fill(nodes, _seg({7: ([0], [0])}), grid=8)
        self.assertEqual(nodes["a"].patch_weights.shape, (1, 64))

    def test_indivisible_resolution_is_rejected(self):
        nodes = {"a": _node([7])}
        with self.assertRaises(ValueError):
            _fill(nodes, _seg({7: ([0], [0])}), grid=3)


if __name__ == "__main__":
    unittest.main()
