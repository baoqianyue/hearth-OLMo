# OLMo-core Patches

This directory is for narrow compatibility patches against OLMo-core releases.

Prefer upstream OLMo-core or a pinned fork for long-lived changes. Keep patches here only when the change is small, easy to audit, and needed to make hearth-OLMo reproducible on common small-GPU workstations.

Current local note:

- PyTorch `2.6.0` does not support `DefaultSavePlanner(enable_plan_caching=...)`. The local OLMo-core checkout used during validation has a guard that only passes `enable_plan_caching` when the installed torch version supports it.
