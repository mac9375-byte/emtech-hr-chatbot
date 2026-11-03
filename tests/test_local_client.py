"""환경변수로 로컬 문서의 모델 호출을 외부 주소에 돌릴 수 없어야 한다."""
import os
import runpy
import unittest
from unittest.mock import patch

from src.config import ROOT


class LocalClientTest(unittest.TestCase):
    def test_environment_cannot_redirect_local_ollama_client(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "https://remote.invalid", "HTTPS_PROXY": "https://proxy.invalid"}):
            values = runpy.run_path(str(ROOT / "src" / "config.py"))
        client = values["OLLAMA_CLIENT"]
        self.assertEqual(str(client._client.base_url), "http://127.0.0.1:11434")
        self.assertFalse(client._client._trust_env)
        client._client.close()


if __name__ == "__main__":
    unittest.main()
