import unittest
from pathlib import Path
from unittest.mock import patch

from app import te_quant


class SourceCopyPolicyTests(unittest.TestCase):
    def test_windows_always_copies(self):
        with patch.object(te_quant.sys, "platform", "win32"):
            self.assertTrue(te_quant._requires_source_copy(Path("shard")))

    def test_other_platforms_do_not_copy(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform), patch.object(te_quant.sys, "platform", platform):
                self.assertFalse(te_quant._requires_source_copy(Path("shard")))


if __name__ == "__main__":
    unittest.main()
