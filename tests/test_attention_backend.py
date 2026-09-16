import copy
import unittest

import torch

from models.components import Attention, Block, set_attention_backend


class AttentionBackendTest(unittest.TestCase):
    def test_fused_attention_matches_math_without_dropout(self):
        torch.manual_seed(7)
        math_attention = Attention(16, num_heads=4, qkv_bias=True).eval()
        fused_attention = copy.deepcopy(math_attention).eval()
        fused_attention.set_backend("fused")
        values = torch.randn(2, 9, 16)

        expected, expected_map = math_attention(values)
        actual, actual_map = fused_attention(values)

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertIsNone(actual_map)
        self.assertEqual(expected_map.shape, (2, 4, 9, 9))

    def test_attention_map_request_keeps_math_semantics(self):
        torch.manual_seed(11)
        block = Block(16, num_heads=4, qkv_bias=True).eval()
        set_attention_backend(block, "fused")
        attention_map = block(torch.randn(2, 7, 16), return_attention=True)
        self.assertEqual(attention_map.shape, (2, 4, 7, 7))
        self.assertTrue(torch.isfinite(attention_map).all())


if __name__ == "__main__":
    unittest.main()
