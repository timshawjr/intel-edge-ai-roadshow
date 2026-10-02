"""The preflight gate's report, and the guards that stop it flattering a machine.

`tools/preflight.py` is the one command a presenter runs before standing in front of a
crowd, and it prints a single word - Ready, Limited or Needs attention - that decides
whether they do. So the tests here are about the cases where that word could be wrong in
the flattering direction:

- **A failing check is `FAIL`, not `WARN`.** The temptation is to soften everything, because
  a `FAIL` on a booth machine is embarrassing. A degraded-but-working capability is `WARN`;
  something that does not work is `FAIL`. A check cannot choose its own level to suit the
  headline.
- **Unavailable is a named reason, never a `0`.** Pinned in
  `TelemetryAvailability`: `available=False` cannot be constructed without a reason, and
  `available=True` cannot carry one. There is no field in which "no reading" is a number.
- **A recorded number is labelled as recorded.** The latency table comes from
  `logs/qualification/m0.json`, written by `tools/qualify_m0.py`. Reusing it is the point -
  a preflight that re-measured would produce a *different* table from the one in the M0
  report - but a reused number must say it was reused, or the reader cannot tell a fresh
  measurement from a copy.
- **The readiness word is derived, never asserted.** A caller that hands `run_checks` a
  hand-written list cannot stamp "Ready" over a failing machine, and a missing machine is
  not Ready.
- **A `WARN` carries a reason.** `ok=False` and a non-`PASS` level must come with a detail,
  because a `WARN` a reader cannot check is a shrug.

The brief's own four tests are kept verbatim in `PreflightIsReadable`; everything after them
is this repository's own addition, covering the guards above.
"""

import unittest

from app.contracts.telemetry_availability import TelemetryAvailability
from tools.preflight import (
    LEVELS,
    Check,
    device_control_lines,
    offer_cpu_only,
    overall_status,
    render_row,
    render_summary,
    validate_rows,
)


class PreflightIsReadable(unittest.TestCase):
    def test_every_row_has_a_level_and_a_plain_detail(self):
        rows = [Check("NPU model compile", True, "PASS", "100 requests, EXECUTION_DEVICES=(NPU,)")]
        out = render_row(rows[0])
        self.assertIn("PASS", out)
        self.assertIn("EXECUTION_DEVICES", out)

    def test_failed_check_is_fail_not_warn(self):
        self.assertEqual(Check("x", False, "PASS", "").level, "PASS")  # sanity on the dataclass
        self.assertNotEqual(Check("x", False, "WARN", "").level, "PASS")

    def test_summary_counts_match_the_rows(self):
        rows = [Check("a", True, "PASS", ""), Check("b", True, "PASS", ""),
                Check("c", False, "WARN", "")]
        self.assertIn("2 PASS", render_summary(rows))
        self.assertIn("1 WARN", render_summary(rows))
        self.assertIn("0 FAIL", render_summary(rows))

    def test_unavailable_telemetry_is_a_named_reason_not_a_zero(self):
        a = TelemetryAvailability(provider="pdh_gpu_engine", available=False,
                                  metric_names=(), scope_note="",
                                  unavailable_reason="no engtype_Neural instance for the AI Boost LUID")
        self.assertFalse(a.available)
        self.assertIn("Neural", a.unavailable_reason)


class CheckIsAFaithfulRecord(unittest.TestCase):
    """A `Check` records what happened; it does not decide what it means."""

    def test_the_level_is_stored_verbatim_and_not_derived_from_ok(self):
        # The brief's own test constructs this. `ok` and `level` are two axes a caller
        # fills in, and the dataclass must not quietly reconcile them - a `Check` that
        # rewrote its own level would hide exactly the mislabelling the renderer needs to
        # be able to show.
        self.assertEqual(Check("x", False, "PASS", "detail").level, "PASS")
        self.assertEqual(Check("x", False, "PASS", "detail").ok, False)

    def test_an_unknown_level_is_refused(self):
        with self.assertRaises(ValueError):
            Check("x", True, "MAYBE", "")

    def test_a_check_with_no_name_is_refused(self):
        with self.assertRaises(ValueError):
            Check("   ", True, "PASS", "detail")

    def test_every_level_the_renderer_knows_is_constructible(self):
        for level in LEVELS:
            self.assertEqual(Check("x", level == "PASS", level, "d").level, level)

    def test_a_pass_needs_no_detail_but_the_rest_of_the_table_does(self):
        # `Check("a", True, "PASS", "")` is in the brief's own test, so an empty detail on
        # a PASS is constructible. The renderer is where that becomes a problem, so the
        # guard lives there and in `validate_rows`.
        self.assertEqual(render_row(Check("a", True, "PASS", "")).strip().count("PASS"), 1)
        self.assertNotEqual(render_row(Check("a", False, "FAIL", "")).strip(), "")


class ARowIsNotFlattered(unittest.TestCase):
    """The one failure mode this whole module exists to prevent.

    A device that failed must not be summarised as a device that was fast, and a machine
    that failed a check must not be summarised as Ready.
    """

    def test_a_row_never_renders_a_number_it_was_not_given(self):
        out = render_row(Check("NPU compile", False, "FAIL", "compile refused"))
        self.assertIn("FAIL", out)
        self.assertNotIn("0.00", out)
        self.assertNotIn("ms", out)

    def test_the_summary_reports_the_worst_level_present(self):
        rows = [Check("a", True, "PASS", "d"), Check("b", False, "WARN", "d"),
                Check("c", False, "FAIL", "d")]
        self.assertIn("1 FAIL", render_summary(rows))
        self.assertIn("Needs attention", render_summary(rows))

    def test_the_summary_does_not_say_ready_when_a_row_failed(self):
        rows = [Check("a", True, "PASS", "d"), Check("b", False, "FAIL", "d")]
        # Asserted on the `Status:` line specifically, not on the whole output. The footer
        # legitimately names all three words while explaining the exit codes, so a blanket
        # substring search for "Ready" would fail on a correct summary - and a test that has
        # to be weakened to pass is a test that has stopped checking anything. The line a
        # presenter reads is the one under test.
        status_lines = [line for line in render_summary(rows).splitlines()
                        if line.startswith("Status:")]
        self.assertEqual(len(status_lines), 1)
        self.assertNotIn("Ready", status_lines[0])

    def test_an_empty_table_is_not_ready(self):
        # Zero rows is zero evidence. Reporting Ready for a table that measured nothing is
        # the same class of error as reporting it for a machine that failed.
        self.assertIn("Needs attention", render_summary([]))


class WarningsCarryAReason(unittest.TestCase):
    """A `WARN` a reader cannot check is a shrug, not a warning."""

    def test_a_warn_without_a_detail_is_refused_by_validate_rows(self):
        with self.assertRaises(ValueError):
            validate_rows([Check("a", True, "PASS", "fine"),
                           Check("b", False, "WARN", "   ")])

    def test_a_fail_without_a_detail_is_refused_by_validate_rows(self):
        with self.assertRaises(ValueError):
            validate_rows([Check("b", False, "FAIL", "")])

    def test_a_pass_without_a_detail_is_allowed(self):
        # `Check("a", True, "PASS", "")` is in the brief's own test, so this must stay
        # constructible and must not be a validation error. The distinction is deliberate:
        # a PASS needs no explanation, a WARN and a FAIL are explanations.
        validate_rows([Check("a", True, "PASS", "")])

    def test_two_rows_with_the_same_name_are_refused(self):
        # Duplicate names make the summary ambiguous: a reader counting reasons for the
        # NPU row cannot tell which of the two NPU rows they are reading.
        with self.assertRaises(ValueError):
            validate_rows([Check("NPU compile", True, "PASS", "d"),
                           Check("NPU compile", True, "PASS", "d")])


class TheReadinessWordIsDerived(unittest.TestCase):
    """No caller can stamp a readiness verdict over a machine that failed."""

    def test_a_failed_check_is_needs_attention(self):
        self.assertEqual(overall_status([Check("a", False, "FAIL", "d")]),
                         "Needs attention")

    def test_a_warn_alone_is_limited(self):
        self.assertEqual(overall_status([Check("a", True, "PASS", "d"),
                                         Check("b", False, "WARN", "d")]),
                         "Limited")

    def test_all_pass_is_ready(self):
        self.assertEqual(overall_status([Check("a", True, "PASS", "d"),
                                         Check("b", True, "PASS", "d")]),
                         "Ready")

    def test_a_warn_does_not_escape_the_notice_when_a_fail_is_present(self):
        self.assertEqual(overall_status([Check("a", False, "FAIL", "d"),
                                         Check("b", False, "WARN", "d")]),
                         "Needs attention")

    def test_the_word_is_exactly_one_of_three(self):
        # Guards against a fourth word creeping in: a presenter-facing verdict has to be a
        # closed set, or "Ready-ish" becomes a thing people say.
        for rows in ([], [Check("a", True, "PASS", "d")],
                     [Check("a", False, "WARN", "d")], [Check("a", False, "FAIL", "d")]):
            self.assertIn(overall_status(rows),
                          ("Ready", "Limited", "Needs attention"))


class TelemetryAvailabilityIsAnUnavailabilityContract(unittest.TestCase):
    """`available=False` is a reason, never a number, and never a blank."""

    def test_unavailable_without_a_reason_is_refused(self):
        with self.assertRaises(ValueError):
            TelemetryAvailability(provider="p", available=False, metric_names=(),
                                  unavailable_reason="   ", scope_note="")

    def test_unavailable_with_a_none_reason_is_refused(self):
        with self.assertRaises(ValueError):
            TelemetryAvailability(provider="p", available=False, metric_names=(),
                                  unavailable_reason=None, scope_note="")

    def test_available_with_a_reason_is_refused(self):
        # The inverse contradiction: an available provider carrying an explanation is a
        # record that has not decided what it measured.
        with self.assertRaises(ValueError):
            TelemetryAvailability(provider="p", available=True, metric_names=("a",),
                                  unavailable_reason="because", scope_note="")

    def test_metric_names_are_normalised_to_a_tuple(self):
        a = TelemetryAvailability(provider="p", available=True,
                                  metric_names=["gpu", "npu"], unavailable_reason=None,
                                  scope_note="per-adapter")
        self.assertIsInstance(a.metric_names, tuple)
        self.assertEqual(a.metric_names, ("gpu", "npu"))

    def test_a_provider_with_no_name_is_refused(self):
        # An unnamed provider cannot be shown as disabled or explained to a presenter.
        with self.assertRaises(ValueError):
            TelemetryAvailability(provider="  ", available=False, metric_names=(),
                                  unavailable_reason="r", scope_note="")

    def test_a_non_bool_available_is_refused(self):
        # `available=0` would be a number where the whole point is that unavailability is
        # not a number. A caller passing one has reintroduced the confusion this contract
        # exists to prevent. `TypeError` rather than `ValueError`: the value is a
        # well-formed int of the wrong type, not a value outside a permitted range, and
        # `assertRaises(TypeError)` is what pins that distinction.
        with self.assertRaises(TypeError):
            TelemetryAvailability(provider="p", available=0, metric_names=(),
                                  unavailable_reason="r", scope_note="")

    def test_to_json_carries_every_field_and_lists_the_tuple(self):
        a = TelemetryAvailability(provider="pdh_gpu_engine", available=False,
                                  metric_names=("gpu_compute_percent",),
                                  unavailable_reason="the in-process value read is refused",
                                  scope_note="per-adapter, never summed")
        payload = a.to_json()
        self.assertEqual(payload["provider"], "pdh_gpu_engine")
        self.assertFalse(payload["available"])
        self.assertEqual(payload["metric_names"], ["gpu_compute_percent"])
        self.assertIn("refused", payload["unavailable_reason"])
        self.assertIn("never summed", payload["scope_note"])


class ControlsForAbsentDevicesStayDisabled(unittest.TestCase):
    """A device that is not there gets a disabled control, never a live-looking one.

    These take `DeviceStatus` records because that is what `DeviceRegistry.describe()`
    returns, and because `DeviceStatus` already refuses to be constructed as an absent
    device without a reason - so a disabled line here always has something to say.
    """

    def _available(self, device_id):
        from app.contracts.device_status import DeviceStatus
        return DeviceStatus(
            device_id=device_id, full_name=f"Intel(R) {device_id}", device_type=device_id,
            available=True, validated=False, ready=False, runtime_version="2026.4.0",
            driver_version=None, supported_model_ids=(), unavailable_reason=None)

    def _absent(self, device_id, reason):
        from app.contracts.device_status import DeviceStatus
        return DeviceStatus(
            device_id=device_id, full_name="", device_type=device_id, available=False,
            validated=False, ready=False, runtime_version="2026.4.0", driver_version=None,
            supported_model_ids=(), unavailable_reason=reason)

    def test_an_available_device_gets_an_enabled_control(self):
        lines = device_control_lines([self._available("CPU")])
        self.assertEqual(len(lines), 1)
        self.assertIn("CPU", lines[0])
        self.assertIn("enabled", lines[0])

    def test_an_absent_device_is_disabled_and_says_why(self):
        lines = device_control_lines([
            self._absent("NPU", "device 'NPU' is not available on this machine")])
        self.assertIn("NPU", lines[0])
        self.assertIn("disabled", lines[0])
        self.assertIn("not available on this machine", lines[0])

    def test_an_absent_device_never_claims_to_be_enabled(self):
        # The check is on the word this module writes, not on the substring "available",
        # which legitimately appears inside `describe()`'s own reason text - the real
        # message quotes `ov.Core().available_devices()`. What must not happen is the line
        # claiming the accelerator is there, so that is what is asserted: this module's own
        # marker is `disabled`, and `enabled` never appears.
        lines = device_control_lines([
            self._absent("GPU", "not in ov.Core().available_devices()")])
        self.assertIn("disabled", lines[0])
        self.assertNotIn("enabled", lines[0])

    def test_an_absent_device_line_never_carries_a_capability_claim(self):
        # An absent device has no full name, so the line must not present one. The control
        # is a state, not a specification.
        lines = device_control_lines([self._absent("NPU", "no NPU plugin")])
        self.assertNotIn("Intel(R) NPU", lines[0])

    def test_the_offered_cpu_only_continuation_names_the_cpu_and_the_absent_devices(self):
        text = offer_cpu_only([self._available("CPU"),
                               self._absent("NPU", "no NPU plugin")])
        self.assertIn("Continue with CPU only", text)
        self.assertIn("NPU", text)

    def test_no_cpu_only_offer_when_every_device_is_present(self):
        # Offering CPU-only on a machine that has all three accelerators would teach a
        # presenter to expect a degradation that is not there.
        text = offer_cpu_only([self._available("CPU"), self._available("GPU"),
                               self._available("NPU")])
        self.assertEqual(text, "")

    def test_a_cpu_only_offer_still_appears_when_only_the_gpu_is_absent(self):
        text = offer_cpu_only([self._available("CPU"),
                               self._absent("GPU", "no GPU plugin"),
                               self._available("NPU")])
        self.assertIn("Continue with CPU only", text)
        self.assertIn("GPU", text)
        self.assertNotIn(" NPU ", f" {text} ")


class TheGateDoesNotCrashWhileReporting(unittest.TestCase):
    """A reporting helper that raises while reporting is the one bug nothing else catches.

    `_recorded_provenance_problem` builds its message with the recorded file's path. The
    first version called `Path.relative_to(REPO_ROOT)` unguarded, which **raises** when the
    path is outside the repo - so the "the recorded run is missing" branch crashed with
    `ValueError` on exactly the case where the file is absent and the caller most needs an
    explanation. Every other row of this gate is a measurement; this one is an explanation,
    and an explanation that raises is invisible until the moment it is needed.
    """

    def test_a_missing_recorded_file_outside_the_repo_is_explained_not_raised(self):
        import tempfile
        from pathlib import Path
        from tools.preflight import _recorded_provenance_problem
        outside = Path(tempfile.mkdtemp()) / "m0.json"
        problem = _recorded_provenance_problem(outside)
        self.assertIsNotNone(problem)
        self.assertIn("no usable recorded qualification run", problem)

    def test_a_path_inside_the_repo_is_shown_relative(self):
        from pathlib import Path
        from tools.preflight import _display_path
        self.assertEqual(_display_path(Path("logs/qualification/m0.json")),
                         "logs\\qualification\\m0.json")

    def test_a_path_outside_the_repo_is_shown_absolute_and_does_not_raise(self):
        import tempfile
        from pathlib import Path
        from tools.preflight import _display_path
        outside = Path(tempfile.mkdtemp()) / "elsewhere.json"
        self.assertEqual(_display_path(outside), str(outside))


class RecordedNumbersAreLabelledAsRecorded(unittest.TestCase):
    """A reused number must say it was reused.

    The per-device table is quoted from `logs/qualification/m0.json` rather than re-measured,
    because a preflight that re-measured would print a *different* table from the one in the
    M0 report. Reuse is only safe while the reader can tell, so both rows that quote a
    recorded figure carry the word in capitals and name the file.
    """

    def _rows(self):
        from tools.preflight import _check_recorded_qualification
        return {row.name: row for row in _check_recorded_qualification()}

    def test_the_recorded_table_row_never_quotes_an_unlabelled_number(self):
        """The invariant, stated so it holds on a machine that has never been qualified.

        `logs/` is gitignored, so `logs/qualification/m0.json` does not exist on a fresh
        clone. Asserting the row says "RECORDED VALUES" would then fail on a perfectly
        correct report - one that says there is nothing to quote. The property that actually
        matters is the disjunction: the row either quotes labelled recorded figures, or says
        there are none. A third possibility, a latency number with no label, is the failure.
        """
        row = self._rows()["Recorded per-device compile, latency and placement"]
        if "RECORDED VALUES" in row.detail:
            self.assertIn("not re-measured", row.detail)
            self.assertIn("m0.json", row.detail)
        else:
            self.assertIn("no usable TargetResult rows", row.detail)
            self.assertNotIn(" p50 ", row.detail)
            self.assertNotIn("ms", row.detail)

    def test_the_recorded_fixture_row_never_quotes_an_unlabelled_verdict(self):
        from tools.preflight import _check_recorded_sanity_fixture
        row = _check_recorded_sanity_fixture()
        if "RECORDED VALUES" in row.detail:
            self.assertIn("not re-measured", row.detail)
        else:
            self.assertIn("no per-device outcomes", row.detail)

    def test_a_multi_line_detail_keeps_its_own_line_structure(self):
        """The regression that made the embedded table unreadable.

        `render_row` originally passed the whole detail to one `textwrap.wrap` call, which
        joins a multi-line detail into a single paragraph. The recorded qualification table
        came out with `device`, `compile s` and `EXECUTION_DEVICES` interleaved - the very
        numbers a reader checks this gate against, rendered illegible. Wrapping per source
        line is what fixes it, and this is the test that says so.
        """
        detail = "RECORDED VALUES:\n\ndevice  p50 ms\nCPU     24.07 ms\nNPU      5.27 ms"
        out = render_row(Check("Recorded table", True, "PASS", detail))
        for line in ("device  p50 ms", "CPU     24.07 ms", "NPU      5.27 ms"):
            self.assertIn(line, out, f"{line!r} did not survive rendering intact")

    def test_a_long_single_line_detail_is_wrapped_not_truncated(self):
        out = render_row(Check("long", True, "PASS", "word " * 200 + "ENDMARKER"))
        self.assertIn("ENDMARKER", out)
        self.assertGreater(len(out.splitlines()), 5)


class TheProvenanceRowActuallyVerifiesTheHash(unittest.TestCase):
    """A file's verdict about itself is not a verification of it (F3).

    `_recorded_provenance_problem` originally read the recorded run's own boolean
    `model_ir_sha256_matches_config_models_json` and never opened the `model_ir_sha256`
    that sits beside it in the same JSON. `model_ir_sha256` **is present in the file**, so
    the comparison was available and simply not made. The consequence is the specific
    failure this gate exists to prevent: a stale, hand-edited or foreign `m0.json` passes
    provenance as long as somebody keeps that one flag `true`, and the row above then
    tells a presenter "The IR hash and the OpenVINO build in that file were checked
    against config/models.json" - which was half true. The OpenVINO half *was* checked;
    the IR-hash half was not.

    These tests write real JSON to a temp file and call the real check, so they fail
    against the old implementation rather than merely describing the new one.
    """

    def _report(self, **overrides):
        import json
        import tempfile
        from pathlib import Path
        import openvino as ov
        from tools.build_sanity_fixture import pinned_input_shape, pinned_model_entry

        entry = pinned_model_entry()
        payload = {
            "generated_by": "tools/qualify_m0.py",
            "openvino_runtime": str(ov.__version__),
            "model_id": entry["id"],
            "model_ir_sha256": entry["sha256"],
            # The flag the old implementation trusted, kept true in every case below.
            "model_ir_sha256_matches_config_models_json": True,
            "input_shape": list(pinned_input_shape()),
            "rows": [],
        }
        payload.update(overrides)
        path = Path(tempfile.mkdtemp()) / "m0.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _problem(self, path):
        from tools.preflight import _recorded_provenance_problem
        return _recorded_provenance_problem(path)

    def test_a_matching_hash_and_shape_has_no_provenance_problem(self):
        self.assertIsNone(self._problem(self._report()))

    def test_a_wrong_ir_hash_is_caught_even_when_the_recorded_flag_says_it_matches(self):
        problem = self._problem(self._report(model_ir_sha256="0" * 64))
        self.assertIsNotNone(problem, "a stale IR hash passed provenance on the flag alone")
        self.assertIn("IR hash", problem)

    def test_a_missing_ir_hash_is_caught_rather_than_trusted(self):
        path = self._report()
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["model_ir_sha256"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertIsNotNone(self._problem(path))

    def test_a_mismatched_input_shape_is_caught(self):
        problem = self._problem(self._report(input_shape=[1, 3, 320, 320]))
        self.assertIsNotNone(problem)
        self.assertIn("input_shape", problem)

    def test_a_missing_input_shape_is_caught(self):
        path = self._report()
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["input_shape"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertIsNotNone(self._problem(path))

    def test_the_self_reported_flag_is_never_the_thing_checked(self):
        """A record that lies *optimistically* about itself is the dangerous one.

        With the flag flipped false and the real hash matching, the row must still pass -
        otherwise the gate is reading a claim rather than the bytes, in the other
        direction. With the flag flipped true and the hash wrong, it must still fail. Both
        halves together say the flag is not consulted.
        """
        path = self._report(model_ir_sha256_matches_config_models_json=False)
        self.assertIsNone(self._problem(path),
                          "the row trusted the record's own false verdict")


if __name__ == "__main__":
    unittest.main()
