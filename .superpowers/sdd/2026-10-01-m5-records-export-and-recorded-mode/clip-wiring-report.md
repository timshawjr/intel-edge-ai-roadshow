# M5 clip wiring — every provisioned clip selectable, with packs the engine can express

**Status:** complete, with the content shortfall stated rather than rounded up.

**Branch:** `main` (no branch created). **Commit:** the commit that contains this report;
its SHA is in the return message. No branch was created and no history was rewritten.

## 1. What was wrong, and what changed

The application named one clip in code (`RETAIL_CHECKOUT_CLIP`) at three call sites and
accepted only `--scenario retail`. Nine clips sat in `assets/media/`; eight were
unreachable. There was one pack manifest, and it was an asset inventory rather than the
configuration the event machine read.

The selection is now one resolved `Selection` — a catalogue record plus its pack — and it
flows to every `ClipSource` construction. `config/clips.json` is the measured catalogue;
`assets/scenarios/<pack>/manifest.json` is the pack configuration for all four packs.

## 2. Clip catalogue

`config/clips.json`, nine records, every field measured from the file with
`cv2.VideoCapture`; SHA-256 and byte count read from the bytes. `in_duration_band` is the
spec's 45–90 s band, inclusive, computed from `frames / fps`. Re-measured by
`tests/test_clip_catalogue.py` and hash-verified by `tools/verify_sources.py`.

| id | pack | resolution | fps | frames | duration (s) | 45–90 s | licence |
|---|---|---|---:|---:|---:|---|---|
| retail-checkout | retail_sanity | 1920x1080 | 29.97003 | 3387 | 113.01290 | out | unverified |
| store-aisle-detection | retail_sanity | 720x404 | 59.94006 | 3921 | 65.41535 | in | unverified |
| bottle-detection | retail_sanity | 640x360 | 29.83333 | 1189 | 39.85475 | out | CC BY 4.0 |
| one-by-one-person-detection | retail_sanity | 768x432 | 10.00000 | 1394 | 139.40000 | out | unverified |
| person-bicycle-car-detection | smart_city | 768x432 | 12.00000 | 647 | 53.91667 | in | CC BY 4.0 |
| car-detection | smart_city | 768x432 | 12.50000 | 377 | 30.16000 | out | CC BY 4.0 |
| people-detection | medical_ops | 768x432 | 12.00000 | 596 | 49.66667 | in | CC BY 4.0 |
| classroom | medical_ops | 1920x1080 | 30.00000 | 984 | 32.80000 | out | CC BY 4.0 |
| worker-zone-detection | depot_logistics | 1920x1080 | 59.94006 | 4548 | 75.87580 | in | CC BY 4.0 |

**Licence status.** The six sidecarred clips carry `CC BY 4.0` and
`https://github.com/intel-iot-devkit/sample-videos` from their clip-scoped
`*.LICENSE` files, with `distribution_status = "licence file present; clearance not
established"`. The three without a sidecar carry `unverified`; none is upgraded to
`cleared`. `retail-checkout`, `store-aisle-detection` and
`one-by-one-person-detection` keep their recorded provenance in `license_source` and
remain uncleared. Rights-cleared remains **0**.

**Band.** Four of nine clips are inside 45–90 s. The five outside are recorded as out, not
rounded in: `retail-checkout` 113.0129 s, `one-by-one-person-detection` 139.4 s,
`bottle-detection` 39.85475 s, `car-detection` 30.16 s, `classroom` 32.8 s.

## 3. The four packs

Each manifest carries `clip_ids`, measured zones (`zones`), an `event_rules` mapping, a
`business_line`, a `business_metric`, an `explanation`, a `delivered_event` sentence, and
an `unexpressible_rules` list. Assignment is **by content, not a proven scene fit**, and
each manifest says so in `content_assignment`.

| pack | clips | event rule the machine computes | spec rule(s) it cannot express | business metric |
|---|---:|---|---|---|
| retail_sanity | 4 | `zone_occupancy` — person in `counter_band` (x 0–1, y 0.20–0.60) for ≥ 2.0 s | product transfer / shelf-to-basket (`product_transfer`) | counted sustained presences at the checkout counter, one per person track |
| smart_city | 2 | `zone_occupancy` — person/car/bicycle in `road_area` (x 0.02–0.98, y 0.05–0.95) for ≥ 0.2 s | directional line crossing (`directional_line_crossing`); stopped-object dwell (`stopped_object_dwell`) | counted sustained presences in the road area, one per track and class |
| medical_ops | 2 | `zone_occupancy` — person in `waiting_area` (x 0.08–0.95, y 0.25–0.90) for ≥ 1.0 s | waiting-zone dwell (`waiting_zone_dwell`) | counted sustained presences in the waiting area, one per person track |
| depot_logistics | 1 | `zone_occupancy` — person in `loading_zone` (x 0.15–1.0, y 0.30–0.90) for ≥ 1.0 s | loading-zone entry (`loading_zone_entry`); loading-zone dwell (`loading_zone_dwell`) | counted sustained presences in the loading zone, one per worker track |

**The counts are 4 / 2 / 2 / 1, and the spec asks for 3 per pack.** No clip was invented
and no pack was emptied to move the number. Three packs are short; the content gate stays
**NOT MET** overall.

**The engine's kinds are exactly `zone_occupancy`, `candidate_transfer` and
`candidate_return`.** There is no line-crossing kind and no dwell-event kind. Every pack
therefore uses `zone_occupancy`. `candidate_transfer` is real and tested, but no
provisioned clip satisfies `Zone.contains` (full containment) at both ends of a route: the
measured boxes span half the frame, so a full-containment transfer never fires. No pack
claims one.

**Zones are measured, not guessed.** Each manifest records the detected-centroid range
and the source. Measured with `yolo11n-fp16`, conf 0.25, iou 0.45, sampled at 5 fps over
the assigned clips:

- smart_city: person cx 0.018–0.986, cy 0.034–0.932 (72 obs); car cx 0.019–0.276,
  cy 0.241–0.887 (41); bicycle cx 0.769–0.962, cy 0.134–0.203 (17).
- medical_ops: people-detection person cx 0.015–0.950, cy 0.356–0.885 (187);
  classroom person cx 0.167–0.739, cy 0.468–0.692 (659).
- depot_logistics: worker-zone person cx 0.204–0.977, cy 0.345–0.792 (295).
- retail_sanity: `COUNTER_BAND` and its 65.1% / 3512-of-5393 measurement are the existing
  M1 record in `app/engine/retail_config.py`; the manifest quotes the same values.

## 4. Selection

`--clip <id>` selects any of the nine; `--pack <id>` selects a pack and its first clip;
both together are accepted only when the clip belongs to that pack. The default is
unchanged: `retail-checkout` under `retail_sanity`.

The selection is resolved once in `main()` and passed to both `run_headless` and
`run_windowed`; each resolves it again through the same `selection_for` function. No
`Path(RETAIL_CHECKOUT_CLIP)` remains on the run path. `--headless --streams 3` is still
refused with its stated reason.

In the windowed screen, the presenter drawer's `status:` line names the running pack and
clip, `drawerScopeNote` is the running pack's `scope_note`, and `drawerScopeReduction` is
the running pack's `delivered_event` — all notified properties, not literals. The header
is deliberately unchanged: the committed capture has 291 px between the title and the
explanation button, and a clip id does not fit at 44 px without shrinking the type.

Help and the default proof:

```text
--clip {retail-checkout,store-aisle-detection,bottle-detection,one-by-one-person-detection,
        person-bicycle-car-detection,car-detection,people-detection,classroom,
        worker-zone-detection}
--pack {retail_sanity,smart_city,medical_ops,depot_logistics}

>>> build_parser().parse_args([])
scenario='retail' clip=None pack=None
>>> default_selection()
clip='retail-checkout' pack='retail_sanity'
```

`--clip retail-checkout --pack smart_city` exits 2 with
`--clip 'retail-checkout' belongs to pack 'retail_sanity', but --pack 'smart_city' was also given`.

The windowed path was started offscreen with `--pack medical_ops` and printed:

```text
[startup] pack medical_ops · clip people-detection (assets/media/people-detection.mp4)
[startup] 1 camera(s): A -> CPU
```


## 5. Real runs

Three different clips, headless, CPU, 8 seconds each. `--device CPU` requested and
confirmed on every run.

| clip | pack | submitted | analysed | drops | events | dedup key |
|---|---|---:|---:|---:|---|---|
| retail-checkout (no flags: the default) | retail_sanity | 121 | 121 | 0 | 1 `zone_occupancy` | `0:1:person:presence:counter_band` |
| people-detection (`--pack medical_ops`) | medical_ops | 121 | 121 | 0 | 1 `zone_occupancy` | `0:8:person:presence:waiting_area` |
| person-bicycle-car-detection (`--clip ...`) | smart_city | 121 | 121 | 0 | 1 `zone_occupancy` | `0:1:person:presence:road_area` |

A windowed run with `--pack medical_ops` was also started offscreen and printed
`[startup] pack medical_ops · clip people-detection` before it was stopped after 12 s; it
loaded the QML window and the selected clip. The full windowed demo was not left running
for this report.

The summary excerpts (retail / people / pbc):

```json
{"selection": {"pack": "retail_sanity", "clip": "retail-checkout", "path": "assets/media/retail-checkout.mp4"},
 "device": {"requested": "CPU", "confirmed": ["CPU"], "state": "active"},
 "frames_submitted": 121, "frames_analysed": 121, "drops": 0,
 "events": [{"kind": "zone_occupancy", "class_name": "person", "track_id": 1,
             "dedup_key": "0:1:person:presence:counter_band"}]}
{"selection": {"pack": "medical_ops", "clip": "people-detection", "path": "assets/media/people-detection.mp4"},
 "device": {"requested": "CPU", "confirmed": ["CPU"], "state": "active"},
 "frames_submitted": 121, "frames_analysed": 121, "drops": 0,
 "events": [{"kind": "zone_occupancy", "class_name": "person", "track_id": 8,
             "dedup_key": "0:8:person:presence:waiting_area"}]}
{"selection": {"pack": "smart_city", "clip": "person-bicycle-car-detection", "path": "assets/media/person-bicycle-car-detection.mp4"},
 "device": {"requested": "CPU", "confirmed": ["CPU"], "state": "active"},
 "frames_submitted": 121, "frames_analysed": 121, "drops": 0,
 "events": [{"kind": "zone_occupancy", "class_name": "person", "track_id": 1,
             "dedup_key": "0:1:person:presence:road_area"}]}
```


A clip may legitimately produce zero events. `bottle-detection` is assigned to
retail_sanity, whose tracked class is `person`; the bottles are visible but the pack's
declared rule does not track them, so zero is the honest count for that pack rather than a
detector failure. The 8 s runs are not full-clip counts and are not presented as such.

## 6. Verification

### New tests

- `tests/test_clip_catalogue.py` — 15 tests.
- `tests/test_packs.py` — 14 tests.
- `tests/test_clip_selection.py` — 21 tests.

### Full suite

`.venv\Scripts\python.exe -m unittest discover -s tests -v` — **Ran 1687 tests in
616.918 s; OK.** That includes the 50 new tests. No failures, no errors, no skips in this
run. The suite's own stderr carries deliberate fault-injection tracebacks from
`test_runner`; they are expected and the result line is `OK`.

### Slot-binding guard

`.venv\Scripts\python.exe -m unittest tests.test_ui_contract.SlotBindingAudit -v` — **7
tests, OK**. The two new QML bindings are `bridge.selection_label` and
`bridge.delivered_event`, both notified properties; no Slot call was added to a binding.

### Mutation checks

Every one of the 50 new tests failed under at least one constructed fault. The full
per-mutation table is the output of the temporary mutation harness; the decisive rows:

| fault introduced | tests that failed |
|---|---|
| catalogue SHA set to zeros | `test_every_measured_field_reproduces_from_the_file` |
| one clip's `pack` swapped | `test_the_operator_assigned_each_clip_to_the_named_pack`, `test_a_pack_selects_its_first_clip`, pack setUpClass |
| `default_clip` changed | `test_the_default_is_unchanged_at_retail_checkout_under_retail_sanity`, both dispatch tests |
| `in_duration_band` flipped and the check removed | `test_the_duration_band_is_computed_from_the_measurement` |
| licence upgraded to CC BY | `test_a_sidecar_is_recorded_as_cc_by_and_never_upgraded_to_cleared` |
| delivered-event sentence claims line crossing | `test_a_limitation_is_stated_in_the_same_words_on_the_screen`, `test_load_pack_refuses_a_manifest_that_claims_line_crossing` |
| claim guard call removed | `test_load_pack_refuses_a_manifest_that_claims_line_crossing` |
| `main` hardcodes the clip | `test_a_clip_is_passed_to_headless_with_its_own_pack` |
| `require_files` bypassed | `test_a_missing_selected_clip_is_refused_before_any_run` |
| QML selection replaced by a literal | `test_the_qml_reads_the_notified_selection_properties` |
| QML delivered event replaced by a literal | `test_the_qml_reads_the_notified_selection_properties` |
| bridge stops updating `scope_note` | `test_set_selection_changes_every_string_together` |
| bridge accepts a clip from another pack | `test_a_clip_its_pack_does_not_contain_is_refused` |
| summary drops the selection | `test_the_headless_summary_carries_pack_clip_and_licence` |
| parser clip choices emptied | `test_every_catalogue_id_is_a_valid_clip_choice` and the dispatch tests |
| catalogue drops a clip | `test_exactly_the_nine_provisioned_clips_are_catalogued` |
| catalogue read touches its own file | `test_the_catalogue_file_is_committed_and_read_only_in_this_test` |
| all four manifests but one dropped from the loader | `test_all_four_manifests_are_on_disk`, count and rule tests |
| business metric emptied in the returned Pack | `test_the_business_metric_and_explanation_are_present` |
| valid route injected into the event rule | `test_the_event_rule_is_one_the_machine_can_compute` |
| zones emptied from the event rule | `test_zones_validate_through_the_only_supported_path` |

The harness backed up and restored every file in a `finally` block; the final
`git status` after it showed only the intended changes, and the new test modules were
re-run green after restoration.

### Numeric claims audit

`git diff`/new-file grep for confident numeric claims in the changed production files:

| claim | where | verification |
|---|---|---|
| nine clips; every resolution, fps, frame count, duration, SHA-256 and byte count | `config/clips.json` | re-measured from each file by `tests/test_clip_catalogue.py`; `tools/verify_sources.py` hashes all nine, 15 PASS / 0 FAIL |
| four in the 45–90 s band, five out | `config/clips.json` | computed from `frames/fps` and asserted by `test_the_duration_band_is_computed_from_the_measurement` |
| 4 / 2 / 2 / 1 clips per pack | the four manifests | `test_the_clip_counts_are_four_two_two_and_one` |
| smart/medical/depot centroid ranges and observation counts | the three manifests | recorded from the 5 fps yolo11n-fp16 dump used to choose the zones; the values are the min/max of that dump, not a guess |
| retail 65.1%, 3512/5393, 470/2920, 17 labels, 12.7% | retail manifest, quoted from `app/engine/retail_config.py` | the M1 measurement; `tests/test_retail_measurements.py` pins the constants and docstring |
| `2 events per clip` | retail `delivered_event` | M1 qualification / footage guard; unchanged wording from the shipped pack |
| 291 px between title and explanation button | `PresenterDrawer.qml` comment | committed capture `docs/qualification/m3-capture/100pct.json`: title x=464 + w=429 = 893, button x=1184, gap 291 |
| 0.2 s / 1.0 s / 2.0 s minimum presence | manifests | these are **rule thresholds chosen for the pack**, not measurements; the measured duration is in the catalogue and the event carries no duration, which the manifests say |
| 121 frames / 8 s / 15.1 fps | the three run summaries | measured by the runs; `frames_analysed=121`, `feed_elapsed_s≈8.0`, `ai_fps≈15.11-15.12` |

### Repository gates

- `tools/verify_sources.py`: **15 PASS, 0 FAIL** (6 model/media rows + 9 catalogue rows).
- `python -m app.main --selftest`: **PASS**, four stages ran.
- `tools/preflight.py`: **16 PASS, 2 WARN, 0 FAIL — Limited** (the two WARNs are the
  recorded-run age and the stale per-boot LUID map; neither is a device fault).
- `tools/probe_telemetry.py`: **PROBE OK**. It rewrites `config/telemetry_map.json` by
  design; that file was restored to its committed bytes and is **not** part of this
  change.

## 7. Concerns

1. **The content gate is still NOT MET.** Four manifests exist, but the clip counts are
   4/2/2/1 against 3 per pack, five of nine clips are outside 45–90 s, and zero clips are
   rights-cleared. Nothing here changes those facts.
2. **Three clips remain unverified** (`retail-checkout`, `store-aisle-detection`,
   `one-by-one-person-detection`). They are selectable, and the headless selection record
   shows `unverified`; that is not a redistribution clearance.
3. **Smart city's line crossing and stopped-object dwell are not computed.** The pack
   measures presence in `road_area` and says so on the drawer. Medical's waiting-zone
   dwell and depot's loading-zone entry/dwell are likewise not computed.
4. **`candidate_transfer` is not used by any pack.** Full containment at both route ends
   is never satisfied by the measured boxes on these nine clips, so a transfer route
   would produce zero events and read as a broken detector.
5. **The bottle clip may show zero events under the retail rule**, because the retail pack
   tracks `person` only. That is the declared rule, not a missing clip.
6. **`config/models.json` was not modified.** The new catalogue is `config/clips.json`;
   `config/telemetry_map.json` was not modified either (the probe's rewrite was reverted).
7. **The windowed selection was proved by a 12 s offscreen startup, not a full booth
   session.** The dispatch is unit-tested and the window printed the selected pack/clip;
   a live presenter session is still the operator's rehearsal.
