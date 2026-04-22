# OLMo-core Dependency Strategy

hearth-OLMo should remain a thin project on top of OLMo-core for the first public release.

## Recommendation

Do not copy OLMo-core source into this repository now. Treat OLMo-core as the training engine and keep hearth-OLMo focused on:

- workstation-sized OLMo3 configs;
- small-GPU launch, batch, checkpoint, and evaluation workflows;
- reproducible data preparation and reporting scripts;
- optional compatibility patches for known version mismatches.

## Why Not Vendor OLMo-core Now

Vendoring would make the repository look self-contained, but it would also create immediate maintenance costs:

- every upstream OLMo-core fix has to be manually merged;
- local changes become harder to compare with official OLMo behavior;
- GitHub users cannot easily tell which code is hearth-OLMo and which code is upstream;
- publication and attribution requirements become heavier because OLMo-core is Apache-2.0 and copied files must preserve notices and modification markers.

## Preferred Upgrade Path

Use this order as hearth-OLMo grows:

1. Depend on released `ai2-olmo-core` for normal use.
2. Support `OLMO_CORE_SRC=/path/to/OLMo-core/src` or a sibling `../OLMo-core/src` checkout for development.
3. Keep small compatibility patches under `patches/olmo-core/` when a released OLMo-core version needs a narrow fix.
4. If small-GPU work requires sustained OLMo-core internals changes, create a dedicated OLMo-core fork and pin hearth-OLMo to that fork.
5. Only vendor a minimal subset if the fork/submodule path becomes impossible.

## When Vendoring Becomes Reasonable

Vendoring is reasonable only if all of these are true:

- the optimization changes touch many private OLMo-core internals;
- upstream cannot accept or review the changes in time;
- the fork diverges enough that pinning a dependency becomes confusing;
- hearth-OLMo maintainers are ready to track OLMo-core security, correctness, and API updates.

Until then, a dependency plus optional patch files is the cleaner engineering boundary.
