import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.explain import largest_component
from src.review import apply_region, draft_report, editor_value, mask_stats, native_map
from src.review_export import export_review


def fixture():
    mask = np.zeros((80, 120), bool)
    mask[20:40, 30:60] = True
    return dict(id="test", image_sha256="fixture", name="<test>.png", rgb=np.full((80, 120, 3), 100, np.uint8),
                mask=mask, auto_mask=mask.copy(), uncertain=np.zeros_like(mask), probs=[.8, .1, .05, .05],
                pred=0, confidence=.8, agreement=1., warnings=["Масштаб в мм неизвестен."],
                method="test", revision=0, history=[])


class ReviewTests(unittest.TestCase):
    def test_empty_cam_is_not_full_image(self):
        mask, bbox = largest_component(np.zeros((224, 224)), .5)
        self.assertFalse(mask.any())
        self.assertIsNone(bbox)

    def test_letterbox_coordinates(self):
        a = np.zeros((224, 224)); a[56:168, :] = 1
        restored = native_map(a, 400, 200, True)
        self.assertEqual(restored.shape, (200, 400))
        self.assertTrue(restored.all())

    def test_edit_rebuild_and_history(self):
        case = fixture(); value = editor_value(case)
        value["layers"][0][:] = 0
        value["layers"][0][5:15, 8:18] = (20, 200, 120, 160)
        changed = apply_region(case, value, "Текст врача до изменения")
        self.assertEqual(mask_stats(changed["mask"])["bbox"], [8, 5, 18, 15])
        self.assertEqual(changed["revision"], 1)
        self.assertIn("x=[8, 18)", draft_report(changed))
        self.assertEqual(changed["history"][0]["previous_text"], "Текст врача до изменения")
        self.assertEqual(case["revision"], 0)
        unchanged = apply_region(case, editor_value(case), "Сохранить даже без правки маски")
        self.assertEqual(unchanged["history"][-1]["previous_text"], "Сохранить даже без правки маски")

    def test_export_rejects_stale_mask(self):
        case = fixture(); value = editor_value(case); value["layers"][0][0,0,3] = 255
        with self.assertRaisesRegex(ValueError, "Сначала нажмите"):
            export_review(case, value, "Текст", "", False)

    def test_export_preserves_text_and_mask_and_escapes_html(self):
        case = fixture(); text = "Правка врача <script>alert(1)</script>\n\nРусский текст"
        with tempfile.TemporaryDirectory() as temp:
            result = export_review(case, editor_value(case), text, "И. И.", True, Path(temp))
            root = Path(result["directory"])
            record = json.loads((root / "review.json").read_text())
            self.assertEqual(record["final_text"], text)
            self.assertNotIn(chr(0x2014), draft_report(case))
            self.assertNotIn(chr(0x2013), draft_report(case))
            self.assertEqual(draft_report(case).count("Ограничения"), 1)
            self.assertEqual(record["final_region"]["area_px"], 600)
            self.assertTrue(record["approved"])
            self.assertNotIn("<script>", (root / "conclusion.html").read_text())
            self.assertTrue(Path(result["pdf"]).read_bytes().startswith(b"%PDF"))
            import zipfile
            with zipfile.ZipFile(result["zip"]) as z:
                self.assertIn("region_final.png", z.namelist())
                self.assertIn("review.json", z.namelist())

    def test_full_erase_is_preserved(self):
        case = fixture(); value = editor_value(case); value["layers"] = []
        changed = apply_region(case, value, "")
        self.assertFalse(changed["mask"].any())
        self.assertIn("Локализация недоступна", draft_report(changed))

    def test_no_region_export_has_no_finding_caption(self):
        case = fixture(); case["mask"][:] = False; case["auto_mask"][:] = False; case["pred"] = 2
        with tempfile.TemporaryDirectory() as temp:
            result = export_review(case, editor_value(case), draft_report(case), "", False, Path(temp))
            doc = (Path(result["directory"]) / "conclusion.html").read_text()
            self.assertIn("Выделенная область отсутствует", doc)
            self.assertNotIn("Находка F1 - зелёная область", doc)


if __name__ == "__main__":
    unittest.main()
