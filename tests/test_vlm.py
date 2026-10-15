"""Offline protocol tests; these never contact a model server."""

import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from manga_pipeline.prompts import TASK_PROMPTS, TASK_SCHEMAS
from manga_pipeline.vlm import VLMClient, VLMError, VLMResponseError, _parse_json, _request_schema


def response(value, finish_reason="stop"):
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return io.BytesIO(json.dumps({"choices": [{
        "finish_reason": finish_reason,
        "message": {"content": content},
    }]}).encode())


class VLMTests(unittest.TestCase):
    def client(self, **overrides):
        return VLMClient({"model": "local-qwen", "request_retries": 0, **overrides})

    def test_parses_plain_and_fenced_json(self):
        expected = {"text": "こんにちは", "uncertain": False, "reason": "visible"}
        self.assertEqual(_parse_json(json.dumps(expected)), expected)
        self.assertEqual(_parse_json("```json\n" + json.dumps(expected) + "\n```"), expected)
        for invalid in ("", "{}", "[]", "Here is JSON: {\"a\":1}", '{"a":1} trailing',
                        '{"text": "first", "text": "second"}', '{"value":NaN}'):
            with self.subTest(invalid=invalid), self.assertRaises(VLMResponseError):
                _parse_json(invalid)

    def test_http_contract_and_raw_response_logging(self):
        result = {"text": "待って！", "uncertain": False, "reason": "visible"}
        with tempfile.TemporaryDirectory() as directory:
            client = VLMClient({"model": "local-qwen", "request_retries": 0, "top_k": 20, "repetition_penalty": 1.0,
                                "api_key_env": "MANGA_TEST_KEY"}, Path(directory))
            with patch.dict(os.environ, {"MANGA_TEST_KEY": "private-test-key"}), \
                 patch("urllib.request.urlopen", return_value=response(result)) as mocked:
                self.assertEqual(client.ask("ocr", {"region_id": "r1", "language": "ja"}), result)
            request = mocked.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/chat/completions")
            body = json.loads(request.data)
            self.assertEqual(body["model"], "local-qwen")
            self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual((body["top_k"], body["repetition_penalty"], body["repeat_penalty"]), (20, 1.0, 1.0))
            self.assertNotIn("top_p", body)
            schema = body["response_format"]["json_schema"]["schema"]
            self.assertEqual(body["response_format"]["type"], "json_schema")
            self.assertEqual(schema["required"], ["text", "uncertain", "reason"])
            self.assertEqual(request.get_header("Authorization"), "Bearer private-test-key")
            logs = list(Path(directory).glob("*.response.txt"))
            self.assertEqual(len(logs), 1)
            self.assertNotIn("private-test-key", logs[0].read_text())
            self.assertIn("choices", json.loads(logs[0].read_text()))

    def test_every_task_has_a_response_schema(self):
        self.assertEqual(set(TASK_SCHEMAS), set(TASK_PROMPTS))

    def test_request_schema_pins_region_ids(self):
        payload = {"regions": [{"region_id": "a"}, {"region_id": "b"}]}
        order = _request_schema("reading_order", payload)["properties"]["region_ids"]
        self.assertEqual((order["items"], order["minItems"], order["maxItems"]), ({"enum": ["a", "b"]}, 2, 2))
        translations = _request_schema("translate", payload)["properties"]["translations"]
        self.assertEqual(translations["items"]["properties"]["region_id"], {"enum": ["a", "b"]})
        self.assertNotIn("enum", str(TASK_SCHEMAS["reading_order"]))

    def test_retries_malformed_output_then_succeeds(self):
        good = {"text": "猫", "uncertain": False, "reason": "visible"}
        with patch("urllib.request.urlopen", side_effect=[response("not JSON"), response(good)]) as mocked:
            self.assertEqual(self.client(request_retries=1).ask("ocr", {"region_id": "r1"}), good)
        self.assertEqual(mocked.call_count, 2)

    def test_network_retries_are_bounded(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")) as mocked:
            with self.assertRaisesRegex(VLMError, "after 3 attempt"):
                self.client(request_retries=2).ask("coverage", {"known_regions": []})
        self.assertEqual(mocked.call_count, 3)

    def test_http_auth_failure_is_not_retried(self):
        error = urllib.error.HTTPError("http://localhost", 401, "unauthorized", {}, io.BytesIO(b"unauthorized"))
        with patch("urllib.request.urlopen", side_effect=error) as mocked:
            with self.assertRaisesRegex(VLMError, "HTTP 401"):
                self.client(request_retries=2).ask("coverage", {})
        self.assertEqual(mocked.call_count, 1)

    def test_truncation_is_never_accepted_even_with_parseable_json(self):
        good = {"text": "猫", "uncertain": False, "reason": "visible"}
        with patch("urllib.request.urlopen", return_value=response(good, "length")):
            with self.assertRaisesRegex(VLMError, "hit max_tokens"):
                self.client().ask("ocr", {"region_id": "r1"})

    def test_empty_ocr_must_be_explicitly_uncertain(self):
        for value in ({"text": "", "uncertain": False, "reason": ""},
                      {"text": "", "uncertain": True, "reason": ""}):
            with patch("urllib.request.urlopen", return_value=response(value)), self.assertRaises(VLMError):
                self.client().ask("ocr", {"region_id": "r1"})
        uncertain = {"text": "", "uncertain": True, "reason": "unreadable crop"}
        with patch("urllib.request.urlopen", return_value=response(uncertain)):
            self.assertEqual(self.client().ask("ocr", {"region_id": "r1"}), uncertain)

    def test_independent_ocr_rejects_candidate_text(self):
        with patch("urllib.request.urlopen") as mocked, self.assertRaisesRegex(ValueError, "Independent OCR"):
            self.client().ask("ocr", {"region_id": "r1", "candidates": {"mangaocr": "猫"}})
        mocked.assert_not_called()

    def test_translation_cannot_drop_duplicate_or_invent_regions(self):
        item = {"region_id": "r1", "text": "貓", "uncertain": False, "reason": ""}
        payload = {"regions": [{"region_id": "r1", "original_text": "猫"}]}
        for items in ([], [item, item], [{**item, "region_id": "invented"}]):
            with self.subTest(items=items), \
                 patch("urllib.request.urlopen", return_value=response({"translations": items})), \
                 self.assertRaisesRegex(VLMError, "every input region ID"):
                self.client().ask("translate", payload)
        with patch("urllib.request.urlopen", return_value=response({"translations": [item]})):
            self.assertEqual(self.client().ask("translate", payload)["translations"], [item])

    def test_verification_checks_every_region_and_consistent_status(self):
        payload = {"regions": [{"region_id": "r1"}]}
        for checks in ([], [{"region_id": "r1", "passed": True, "issues": ["negation omitted"]}],
                       [{"region_id": "r1", "passed": False, "issues": []}]):
            with self.subTest(checks=checks), \
                 patch("urllib.request.urlopen", return_value=response({"checks": checks})), \
                 self.assertRaises(VLMError):
                self.client().ask("verify", payload)

    def test_normalized_coverage_coordinates_are_checked(self):
        for bbox in ([0, 0, 1001, 900], [20, 0, 10, 50], [False, 0, 10, 20]):
            value = {"regions": [{"bbox": bbox, "kind": "text", "reason": "visible"}], "uncertain": False}
            with self.subTest(bbox=bbox), patch("urllib.request.urlopen", return_value=response(value)), \
                 self.assertRaises(VLMError):
                self.client().ask("coverage", {"known_regions": []})
        value = {"regions": [], "uncertain": False}
        with patch("urllib.request.urlopen", return_value=response(value)):
            self.assertEqual(self.client().ask("coverage", {"known_regions": []}), value)

    def test_visual_qa_cannot_pass_with_issues(self):
        value = {"passed": True, "issues": [{"region_id": "r1", "category": "mask", "reason": "residual text"}]}
        with patch("urllib.request.urlopen", return_value=response(value)), self.assertRaises(VLMError):
            self.client().ask("visual_qa", {"regions": [{"region_id": "r1"}]})

    def test_context_cannot_invent_glossary_evidence(self):
        value = {"summary": "A character enters.", "characters": [], "glossary": [
            {"source": "猫", "target": "貓", "status": "confirmed", "evidence": ["invented"]}
        ]}
        payload = {"pages": [{"regions": [{"region_id": "r1", "original_text": "猫"}]}]}
        with patch("urllib.request.urlopen", return_value=response(value)), \
             self.assertRaisesRegex(VLMError, "evidence absent"):
            self.client().ask("context", payload)
        value["glossary"][0]["evidence"] = ["r1"]
        with patch("urllib.request.urlopen", return_value=response(value)):
            self.assertEqual(self.client().ask("context", payload), value)

    def test_image_transport_preserves_ratio_and_original(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")
        original = Image.new("RGBA", (800, 400), (255, 0, 0, 0))
        url = self.client(image_max_pixels=80000)._image_url(original)
        sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        self.assertEqual(sent.size, (400, 200))
        self.assertEqual(original.size, (800, 400))
        self.assertEqual(sent.getpixel((0, 0)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
