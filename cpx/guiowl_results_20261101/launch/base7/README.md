# GUI-Owl BASE adapter repair: seven tasks only

The parser now accepts `open_app` as well as `open`, matching the official
`mobile_agent_utils_new.py` OPEN_APP mapping. Both require a nonempty app name
in `text`. Original model responses and native history remain unchanged.

No coordinate, sampling, prompt, seed, budget, answer, or execution policy changes.
The baseline rerun uses one fresh Android container, direct task seed30,
temperature0, five-image history, and uncapped `int(10*complexity)` budgets.
FULL45 is not rerun. The original paired run is never modified.

The seven tasks were selected from the original BASE execution errors, before
the retry: MarkorCreateNote, MarkorChangeNoteContent, MarkorMergeNotes,
NotesMeetingAttendeeCount, SimpleCalendarDeleteEventsOnRelativeDay,
SportsTrackerLongestDistanceActivity, TasksDueNextWeek, in original relative order.

Validation: 14 targeted unit tests passed. On-server replay checked all42 stored
raw responses from the seven failed tasks through the patched parser and the
actual runtime GELab action conversion; all mapped to OPEN_APP with the original
app name. Formal task execution supplies the end-to-end device check.

Remote root: `/home/pxchen/workspace/android_world_mai8b_20261024/cpx_guiowl_base7_20261101`

Tag: `guiowl8b_base7_openapp_fix_20261101`

`merge_results.py` preserves the other109 original rows byte-equivalent as JSON
objects, replacing only these seven rows. Its output explicitly identifies the
result as an original109 + repaired7 combination, not a fresh full116 run.

Official reference:
https://github.com/X-PLUG/MobileAgent/blob/main/Mobile-Agent-v3.5/android_world_v3.5/android_world/agents/mobile_agent_utils_new.py
