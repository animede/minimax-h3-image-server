import unittest
from pathlib import Path
from unittest.mock import patch

from app import te_quant


class SourceCopyPolicyTests(unittest.TestCase):
    def test_windows_filesystems(self):
        for filesystem, expected in [("exFAT", True), ("EXFAT", True), ("NTFS", False), ("ReFS", False)]:
            with self.subTest(filesystem=filesystem), patch.object(te_quant.sys, "platform", "win32"), \
                    patch.object(te_quant, "_windows_filesystem", return_value=filesystem):
                self.assertEqual(te_quant._requires_source_copy(Path("shard")), expected)

    def test_detection_failure_copies(self):
        with patch.object(te_quant.sys, "platform", "win32"), \
                patch.object(te_quant, "_windows_filesystem", side_effect=OSError("Unavailable")), \
                self.assertLogs(te_quant.logger, level="WARNING"):
            self.assertTrue(te_quant._requires_source_copy(Path("shard")))

    def test_empty_filesystem_copies(self):
        with patch.object(te_quant.sys, "platform", "win32"), \
                patch.object(te_quant, "_windows_filesystem", return_value=""), \
                self.assertLogs(te_quant.logger, level="WARNING"):
            self.assertTrue(te_quant._requires_source_copy(Path("shard")))

    def test_other_platforms_do_not_inspect_filesystem(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform), patch.object(te_quant.sys, "platform", platform), \
                    patch.object(te_quant, "_windows_filesystem") as inspect:
                self.assertFalse(te_quant._requires_source_copy(Path("shard")))
                inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
