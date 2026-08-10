"""An installed SDK we did not hook must never pass in silence.

`init()` used to warn only when it patched **nothing**. A customer with
``openai`` and ``google-genai`` installed got "patched SDKs: openai" at INFO and
not one word about Gemini — so the partial case, which is the common case, was
silent, and they had every reason to believe they were covered.

That matters more here than anywhere else in the product. Auto-instrumentation
works by rewriting a named function inside the customer's process, so it is
inherently a finite list that goes stale every time a provider renames a
package — which Google just did. We cannot win that race by being fast. We win
it by never being quietly wrong about it: an unhooked library is named, and the
advice points at the proxy, which is library-blind and always works.
"""

from __future__ import annotations

import logging

import pytest

from promptguard import auto


@pytest.fixture(autouse=True)
def _no_patches_applied():
    """Each test decides for itself what counts as patched."""
    original = list(auto._applied_patches)
    auto._applied_patches.clear()
    yield
    auto._applied_patches[:] = original


class _FakePatch:
    def __init__(self, name: str, detects: tuple[str, ...]):
        self.NAME = name
        self.DETECTS = detects


class TestDetection:
    def test_an_installed_but_unhooked_library_is_named(self, monkeypatch):
        """The regression. `pytest` stands in for a package that is definitely
        importable, so the test does not depend on which provider SDKs happen to
        be installed in the environment running it."""
        monkeypatch.setattr(
            auto, "_known_patches", lambda: [_FakePatch("fake-provider", ("pytest",))]
        )
        assert auto._detected_unpatched() == ["pytest"]

    def test_a_hooked_library_is_not_reported(self, monkeypatch):
        patch = _FakePatch("fake-provider", ("pytest",))
        monkeypatch.setattr(auto, "_known_patches", lambda: [patch])
        auto._applied_patches.append(patch)

        assert auto._detected_unpatched() == []

    def test_an_absent_library_is_not_reported(self, monkeypatch):
        """Silence about a provider the customer does not use is correct.

        A report that names everything is as useless as one that names nothing —
        people stop reading it, which is how the next real warning gets missed.
        """
        monkeypatch.setattr(
            auto,
            "_known_patches",
            lambda: [_FakePatch("fake", ("a_module_that_does_not_exist_xyz",))],
        )
        monkeypatch.setattr(auto, "_KNOWN_UNPATCHED", {})

        assert auto._detected_unpatched() == []

    def test_known_unpatched_providers_are_reported_by_distribution_name(self, monkeypatch):
        """The name in the warning must be the one you would `pip install`.

        Telling someone `google.cloud.aiplatform` is unhooked is less actionable
        than telling them `google-cloud-aiplatform` is.
        """
        monkeypatch.setattr(auto, "_known_patches", list)
        monkeypatch.setattr(auto, "_KNOWN_UNPATCHED", {"pytest": "pytest-the-distribution"})

        assert auto._detected_unpatched() == ["pytest-the-distribution"]

    def test_a_framework_is_not_reported(self, monkeypatch):
        """LangChain and friends must NOT appear.

        `langchain_openai.ChatOpenAI` calls the `openai` package underneath, and
        we patch `openai`, so LangChain traffic is scanned transitively. Warning
        about it would be a false alarm that teaches customers to ignore the
        real ones.
        """
        monkeypatch.setattr(auto, "_known_patches", list)

        for framework in ("langchain", "langchain_openai", "llama_index", "crewai"):
            assert framework not in auto._KNOWN_UNPATCHED


class TestTheWarningIsActuallyEmitted:
    def test_it_names_the_library_and_points_somewhere_useful(self, monkeypatch, caplog):
        monkeypatch.setattr(auto, "_detected_unpatched", lambda: ["google-genai"])

        with caplog.at_level(logging.WARNING, logger="promptguard"):
            auto._warn_about_unpatched_libraries()

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "google-genai" in message
        assert "not being scanned" in message
        assert auto._ADVICE_URL in message

    def test_nothing_is_logged_when_everything_is_hooked(self, monkeypatch, caplog):
        monkeypatch.setattr(auto, "_detected_unpatched", list)

        with caplog.at_level(logging.WARNING, logger="promptguard"):
            auto._warn_about_unpatched_libraries()

        assert caplog.records == []


class TestTheReportIsMachineReadable:
    """A startup warning is read by whoever is watching at startup.

    The same facts as data can be asserted in the customer's own CI, which turns
    "we told you" into "you cannot ship without knowing".
    """

    def test_the_report_has_the_three_things_a_caller_needs(self, monkeypatch):
        monkeypatch.setattr(auto, "_known_patches", list)
        monkeypatch.setattr(auto, "_KNOWN_UNPATCHED", {})

        report = auto.instrumentation_report()

        assert set(report) == {"patched", "detected_unpatched", "advice_url"}
        assert report["detected_unpatched"] == [], (
            "an empty list must be reachable, or `== []` is an assertion no "
            "customer could ever satisfy"
        )

    def test_it_is_exported_from_the_package_root(self):
        import promptguard

        assert promptguard.instrumentation_report is auto.instrumentation_report
