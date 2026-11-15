"""Cloud AI providers: current default models, and Claude requests the current API accepts.

Model IDs were read from each provider's model page on 2026-11-10.
"""
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from whisperjav.translate.providers import PROVIDER_CONFIGS

APP_JS = Path(__file__).resolve().parents[1] / "whisperjav" / "webview_gui" / "assets" / "app.js"

# Retired or shut down by the provider. None may be a default or a dropdown choice.
RETIRED = {
    "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229",
    "anthropic/claude-3.5-sonnet", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "openai/gpt-4o",
    "glm-4-flash", "glm-4", "glm-4-plus",
    "llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768",
}


def _gui_model_lists() -> dict:
    """Parse the `providerModels` table out of app.js.

    Returns:
        A dict of provider key to its list of model IDs.
    """
    text = APP_JS.read_text(encoding="utf-8")
    block = re.search(r"providerModels: \{(.*?)\n    \}", text, re.S).group(1)
    lists = {}
    for key, body in re.findall(r"^\s*(\w+): \[(.*?)\]", block, re.M | re.S):
        lists[key] = re.findall(r"'([^']+)'", body)
    return lists


class TestDefaults:
    def test_no_retired_default(self):
        for key, config in PROVIDER_CONFIGS.items():
            assert config["model"] not in RETIRED, key

    def test_claude_output_room_for_thinking(self):
        # Current Claude models think before answering, and the thinking counts against max_tokens.
        assert PROVIDER_CONFIGS["claude"]["max_tokens"] >= 16000


class TestGuiModelLists:
    def test_one_table(self):
        # The settings dialog reuses the main table, so the two cannot drift apart.
        assert APP_JS.read_text(encoding="utf-8").count("providerModels: {") == 1

    def test_no_retired_choice(self):
        for key, models in _gui_model_lists().items():
            assert not RETIRED & set(models), key

    def test_first_choice_is_the_default(self):
        for key, models in _gui_model_lists().items():
            if models:
                assert models[0] == PROVIDER_CONFIGS[key]["model"], key


class TestConnectionCheck:
    """The GUI's Test Connection button must ask each provider, not OpenAI."""

    @pytest.fixture
    def api(self):
        from whisperjav.webview_gui.api import WhisperJAVAPI
        return WhisperJAVAPI()

    @pytest.fixture
    def sent(self, monkeypatch):
        """Record the request urlopen gets, and answer it with the status in `sent["status"]`."""
        import io
        import urllib.error
        import urllib.request

        sent = {"status": 200}

        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def urlopen(req, timeout=None):
            sent["url"] = req.full_url
            sent["headers"] = {k.lower(): v for k, v in req.header_items()}
            if sent["status"] != 200:
                raise urllib.error.HTTPError(req.full_url, sent["status"], "err", None, io.BytesIO())
            return _Response()

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        return sent

    @pytest.mark.parametrize("provider, url, auth", [
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/models", ("x-goog-api-key", "k")),
        ("groq", "https://api.groq.com/openai/v1/models", ("authorization", "Bearer k")),
        ("glm", "https://open.bigmodel.cn/api/paas/v4/models", ("authorization", "Bearer k")),
        ("gpt", "https://api.openai.com/v1/models", ("authorization", "Bearer k")),
        ("deepseek", "https://api.deepseek.com/models", ("authorization", "Bearer k")),
    ])
    def test_each_provider_asks_its_own_server(self, api, sent, provider, url, auth):
        assert api.test_provider_connection(provider, "k")["success"]
        assert sent["url"] == url
        assert sent["headers"][auth[0]] == auth[1]
