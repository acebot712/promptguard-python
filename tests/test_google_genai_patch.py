"""The Gemini SDK people actually install.

We hooked ``google-generativeai``. Google deprecated it when Gemini 2.0 shipped
— its own PyPI description opens with "[Deprecated]" — and replaced it with
``google-genai``, a different import path entirely. So anyone on the current SDK
called ``promptguard.init()``, got no error, and was unprotected.

Both are patched now. The old one is not dropped: customers are still on it, and
removing it to chase the new one would recreate the identical bug pointed the
other way.

Two things this patch can do that the legacy one could not, and both are tested
below because both are the reason the new signature matters:

* ``config.system_instruction`` is scanned — the highest-authority text in the
  request, and not part of ``contents`` at all.
* Redaction can rewrite the prompt, because ``contents`` is keyword-only here.
  The legacy patch has to escalate a redact decision to a block instead.
"""

from __future__ import annotations

import pytest

from promptguard.patches import google_genai_patch as patch


class TestItTargetsTheCurrentSdk:
    def test_it_detects_the_new_import_path(self):
        assert patch.DETECTS == ("google.genai",)

    def test_it_is_registered_for_patching(self):
        from promptguard import auto

        assert patch in auto._known_patches()

    def test_the_legacy_patch_is_still_registered(self):
        """Deprecated by Google is not the same as unused by customers."""
        from promptguard import auto
        from promptguard.patches import google_patch

        assert google_patch in auto._known_patches()
        assert google_patch.NAME != patch.NAME, (
            "the two Google SDKs must report separately, or patched_sdks() cannot "
            "tell you which one you are actually covered on"
        )


class TestEverythingScannableIsRead:
    def test_the_system_instruction_is_scanned(self):
        messages, _model, _ctx = patch._extract_messages(
            (None,),
            {
                "model": "gemini-2.5-flash",
                "contents": "hi",
                "config": {"system_instruction": "SYS"},
            },
        )
        assert {"role": "system", "content": "SYS"} in messages, (
            "the system instruction lives in `config`, not `contents` — missing it "
            "leaves the highest-authority text in the request unscanned"
        )

    def test_a_bare_string_is_a_user_turn(self):
        messages, model, _ = patch._extract_messages(
            (None,), {"model": "gemini-2.5-flash", "contents": "hello"}
        )
        assert messages == [{"role": "user", "content": "hello"}]
        assert model == "gemini-2.5-flash"

    @pytest.mark.parametrize(
        ("part", "expected"),
        [
            ({"text": "alpha"}, "alpha"),
            ({"function_call": {"name": "lookup", "args": {"q": "bravo"}}}, "bravo"),
            ({"functionCall": {"name": "lookup", "args": {"q": "charlie"}}}, "charlie"),
            ({"function_response": {"response": {"r": "delta"}}}, "delta"),
            ({"executable_code": {"code": "print('echo')"}}, "echo"),
            ({"code_execution_result": {"output": "foxtrot"}}, "foxtrot"),
        ],
    )
    def test_every_kind_of_part_reaches_the_scanner(self, part, expected):
        """Tool arguments especially: exfiltration is written there, not in prose."""
        messages, _, _ = patch._extract_messages(
            (None,), {"model": "m", "contents": [{"role": "user", "parts": [part]}]}
        )
        assert expected in messages[0]["content"]

    def test_all_candidates_are_read_from_the_response(self):
        """`response.text` is only the first candidate. An n>1 reply must not
        lose the rest of what the model said before it is scanned."""
        from types import SimpleNamespace

        def _candidate(text: str) -> SimpleNamespace:
            return SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))

        response = SimpleNamespace(
            candidates=[_candidate("first"), _candidate("second")],
            text="first",
        )

        extracted = patch._extract_response_text(response) or ""
        assert "first" in extracted and "second" in extracted


class TestRedaction:
    def test_it_rewrites_contents(self):
        """Possible here and not on the legacy SDK, because `contents` is a kwarg."""
        kwargs = {"model": "m", "contents": "my email is alice@example.com"}
        out = patch._apply_redaction((None,), kwargs, [{"role": "user", "content": "[REDACTED]"}])

        assert out is not None
        assert out["contents"] == [{"role": "user", "parts": [{"text": "[REDACTED]"}]}]
        assert out["model"] == "m", "unrelated kwargs must survive the rewrite"

    def test_a_system_only_redaction_fails_safe(self):
        """Returning None makes `_base` escalate to a block in enforce mode.

        The system instruction lives in `config` and this rewrite does not touch
        it, so claiming success would forward text we were told to redact.
        """
        assert patch._apply_redaction((None,), {}, [{"role": "system", "content": "x"}]) is None


class TestApplyAndRevert:
    def test_it_patches_and_unpatches_every_surface(self):
        genai_models = pytest.importorskip("google.genai.models")

        before = {
            (cls, meth): getattr(getattr(genai_models, cls), meth)
            for cls, meth, _ in patch._SURFACES
            if hasattr(genai_models, cls)
        }
        assert patch.apply() is True
        try:
            for (cls, meth), original in before.items():
                assert getattr(getattr(genai_models, cls), meth) is not original, (
                    f"{cls}.{meth} was not wrapped — streaming and async surfaces "
                    "matter as much as the plain one; a customer who passes "
                    "stream=True must not silently stop being scanned"
                )
        finally:
            patch.revert()

        for (cls, meth), original in before.items():
            assert getattr(getattr(genai_models, cls), meth) is original

    def test_apply_is_false_when_the_sdk_is_absent(self, monkeypatch):
        monkeypatch.setattr(patch, "_patched", False)
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

        assert patch.apply() is False
