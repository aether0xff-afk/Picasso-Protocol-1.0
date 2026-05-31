import tempfile
import unittest
from pathlib import Path

from picasso_protocol.model_paths import resolve_model_path


class ModelPathTest(unittest.TestCase):
    def test_prefers_current_working_directory_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd = root / "toWebPage"
            cwd.mkdir()
            root_model = root / "artist_x_best_model.pth"
            cwd_model = cwd / "artist_x_best_model.pth"
            root_model.write_bytes(b"root")
            cwd_model.write_bytes(b"cwd")

            self.assertEqual(
                resolve_model_path("artist_x_best_model.pth", project_root=root, cwd=cwd),
                cwd_model,
            )

    def test_falls_back_to_project_root_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd = root / "toWebPage"
            cwd.mkdir()
            root_model = root / "artist_x_best_model.pth"
            root_model.write_bytes(b"root")

            self.assertEqual(
                resolve_model_path("artist_x_best_model.pth", project_root=root, cwd=cwd),
                root_model,
            )


if __name__ == "__main__":
    unittest.main()
