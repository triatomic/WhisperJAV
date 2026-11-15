"""Cloud AI providers: current default models, the Claude model name, and the Test Connection check.

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


class TestClaudeModelName:
    """A Claude model the key's list does not hold is sent as given, not refused before the request."""

    def _provider(self, listed):
        def strict(name):
            if name in listed:
                return name
            raise ValueError(f"Model {name} not found")
        return SimpleNamespace(_get_model_id=strict, claude_models=[SimpleNamespace(id=m) for m in listed])

    @pytest.mark.parametrize("listed", [[], ["claude-haiku-4-5-20251001"]])
    def test_unlisted_name_is_sent_as_given(self, listed, capsys):
        from whisperjav.translate.core import apply_claude_lenient_model_id
        provider = self._provider(listed)
        assert apply_claude_lenient_model_id(provider)
        assert provider._get_model_id("claude-haiku-4-5") == "claude-haiku-4-5"
        assert "sending" in capsys.readouterr().err

    def test_listed_name_is_unchanged_and_silent(self, capsys):
        from whisperjav.translate.core import apply_claude_lenient_model_id
        provider = self._provider(["claude-haiku-5-5"])
        apply_claude_lenient_model_id(provider)
        assert provider._get_model_id("claude-haiku-5-5") == "claude-haiku-5-5"
        assert capsys.readouterr().err == ""

    def test_pysubtrans_still_has_the_method_we_wrap(self):
        # A rename in PySubtrans would switch the wrapper off without an error.
        from PySubtrans.Providers.Provider_Claude import ClaudeProvider
        assert hasattr(ClaudeProvider, "_get_model_id")


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
