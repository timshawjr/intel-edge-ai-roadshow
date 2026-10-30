"""M6 Task 5: the signing verdict is written down exactly, and it says "unsigned".

There is no code-signing certificate on this machine and no Authenticode signature on the
package. That is a fact about the deliverable, not a documentation gap, so the document has
to do four things and this suite checks each one:

1. **Say the package is unsigned and no certificate is available.** A reader must not have
   to infer it from a manifest field.
2. **Print the exact commands a human with a certificate would run** - the `signtool sign`
   form with `/fd SHA256`, an RFC 3161 timestamp URL, `/td SHA256` and `/a`, and the
   `signtool verify /pa /all` form.
3. **State what the absence costs** - the SmartScreen "unknown publisher" warning, and the
   possibility that an enterprise policy blocks the package outright.
4. **Forbid working around it by weakening a security control** - no Defender exclusions,
   no ASR changes, no execution-policy change, no AppLocker bypass - and say that a step
   needing elevation is an install step for a human, not a runtime step.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCUMENT = REPO / "docs" / "install" / "SIGNING.md"

SIGN_COMMAND = "signtool sign /fd SHA256 /tr <timestamp-url> /td SHA256 /a <file>"
VERIFY_COMMAND = "signtool verify /pa /all <file>"


def _document() -> str:
    if not DOCUMENT.is_file():
        raise AssertionError(
            f"{DOCUMENT.relative_to(REPO)} does not exist; the signing verdict is the "
            "deliverable this test exists to check"
        )
    return DOCUMENT.read_text(encoding="utf-8")


class ThePackageIsUnsignedAndSaysSo(unittest.TestCase):
    def test_the_document_states_the_package_is_unsigned(self):
        text = _document()
        self.assertIn("unsigned", text.lower())
        self.assertIn("not signed", text.lower(),
                      "the document must say the artifact is not signed, not merely "
                      "mention the word 'unsigned' once")

    def test_the_document_states_no_certificate_is_available(self):
        text = _document().lower()
        self.assertIn("no certificate", text)
        self.assertIn("available", text)
        self.assertIn("authenticode", text,
                      "name the signature technology the package lacks")

    def test_the_document_names_the_artifact_and_the_manifest_field(self):
        text = _document()
        self.assertIn("dist/M6-bundle", text)
        self.assertIn("bundle-manifest.json", text)
        self.assertIn('"signed": false', text,
                      "the manifest field a reader can check must be quoted exactly")


class TheExactCommandsAreWrittenDown(unittest.TestCase):
    def test_the_exact_sign_command_is_present(self):
        self.assertIn(
            SIGN_COMMAND, _document(),
            "the document must print the complete command, including the timestamp and "
            "digest options; a partial command is not something a human can run",
        )

    def test_the_exact_verify_command_is_present(self):
        self.assertIn(VERIFY_COMMAND, _document())

    def test_the_timestamp_url_is_explained(self):
        text = _document()
        self.assertIn("<timestamp-url>", text)
        self.assertIn("RFC 3161", text,
                      "the placeholder must name the timestamp protocol it stands for")
        self.assertIn("/tr", text)

    def test_the_commands_require_a_certificate_a_human_holds(self):
        text = _document().lower()
        self.assertIn("code-signing certificate", text)
        self.assertIn("windows sdk", text,
                      "signtool ships with the Windows SDK; say where it comes from")
        self.assertIn("human", text,
                      "the step is performed by a human with the certificate, not by the "
                      "application")


class WhatTheAbsenceCosts(unittest.TestCase):
    def test_the_smartscreen_unknown_publisher_warning_is_stated(self):
        text = _document()
        self.assertIn("SmartScreen", text)
        self.assertIn("unknown publisher", text.lower())

    def test_enterprise_policy_may_block_the_package_outright(self):
        text = _document().lower()
        self.assertIn("enterprise policy", text)
        self.assertIn("may block", text)
        self.assertIn("outright", text)


class WeakeningASecurityControlIsForbidden(unittest.TestCase):
    def test_every_named_control_is_forbidden(self):
        text = _document().lower()
        for control in ("defender exclusion", "asr rule", "execution policy",
                        "applocker"):
            with self.subTest(control=control):
                self.assertIn(control, text,
                              f"the document must name the {control} workaround it "
                              "forbids")
        self.assertIn("forbidden", text)

    def test_elevation_is_an_install_step_not_a_runtime_step(self):
        text = _document().lower()
        self.assertIn("elevation", text)
        self.assertIn("install step", text)
        self.assertIn("standard user", text,
                      "the application must still run as a standard user")


if __name__ == "__main__":
    unittest.main()
