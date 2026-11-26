# victory-mot — a VIctory branch tuned for one 6 GB laptop

A branch of [VIctory](https://github.com/Marvisak/VIctory) that runs the Xbox Series
build of one open-world title natively on Windows, from files the user supplies.
It is a compatibility layer, not an emulator: the title's own x86-64 code runs on
the CPU, and the layer provides the console libraries it imports, translates its
shaders and presents the console's memory model to it.

**This repository contains no game files, code or assets.** You need your own
copy of the game. It is not affiliated with or endorsed by the game's developer
or publisher.

## What the layer does

- Reimplements the console libraries a Series title imports: `xgameruntime`
  (the Xbox game runtime), the `d3d12_xs` variant of Direct3D 12,
  `xboxbvhbuilder`, DirectStorage, the file-system and the kernel calls.
- Translates the console's GPU programs (RDNA2 machine code) to HLSL and then to
  DXIL for the host's shader model, including console waves of 64 lanes on a
  host whose waves are 32.
- Presents the console's memory model unchanged: virtual allocation,
  reservations, views, and the remapping a title does while it streams.
- Keeps the title's view of its own memory consistent while the GPU writes into
  it: a write watch, per-page guards, and a copy back of what shaders wrote.

## Status on the machine this branch is tuned for (2026-11-26)

One laptop: Windows 11, NVIDIA RTX 3050 with 6 GB of local video memory (a
4,407 MB budget as DXGI reports it), the small console profile throughout.
**Not playable yet.**

| Stage | State here |
|-------|-------------|
| Boot, start screen, menus | drawn at the title's own 30 fps |
| Opening cutscene | plays in order |
| The world | renders: characters, terrain, sky, HUD, pad input |
| Frame rate in the world | **1 to 2 frames per second**; the fit `frame = intercept + slope * draws` over 142 logged frames gives a **267 ms** intercept, at 2,600 to 4,400 draws |
| Light in shadow | the title's ray-traced bounce light does not reach the screen yet, so shadowed surfaces are flat and dark. A constant can stand in for it (`VICTORY_AMBIENT=0.8`), and it is a stand-in, not the game's light |
| Textures | most arrive, late; where one has not arrived the slot shows a 1×1 grey, and the count of those is in the log |
| Start screen image | the front-end draws, but its picture comes from the same missing-texture path: the logo shows as the grey fallback with its alpha cut out |
| Stability | no crash and no device loss in runs of 15 minutes |

Everything above is measured, not estimated: runs of a fixed length, captures
scored against a fixed rule, and frame time compared only at equal draw counts.

## What this branch adds

**A crash reporter that names the function.** Every crash line carries the
`.pdata` function that holds the faulting instruction — `0x4faf7cb in
0x4faf710-0x4faf844 (+0xbb) of the image` — instead of an offset inside a 120 MB
image. It reads through `ReadProcessMemory` with checked arithmetic, so a
half-corrupt image yields no function rather than a second fault.

**A guard that cannot leave a block read-only and forgotten.** A remap drops a
block from the guard set and puts the page back to writable; the copy that writes
title memory announces itself before it writes; an unmap is notified after it
happened and its result checked. A test failed with `PAGE_READONLY` before this
and passes with `PAGE_READWRITE` after. Without it, a run died 280 seconds into
the world, once in twelve runs.

**A texture view the title asks for and the layer did not have.** The title names
its ray-tracing acceleration structure by heap index, with the view dimension the
layer had no branch for; the slot fell to a null texture view, the shader's node
reads found zeros, and the shadow mask came out with no occluders. It is now a
raw `R32_TYPELESS` view over the host copy from the address the title gives.

**Two diagnostics that cost nothing when off.** `VICTORY_ABLATE=heap,copy,sync,…`
skips one stage at a time so its maximum worth can be measured; `VICTORY_TRACE`
writes a timeline of every stage and thread that opens in Perfetto.

**A colour pass, kept as a diagnostic only.** A full-screen pass just before the
title's 8-bit encode can move the black point, the highlight shoulder and the
chroma. It is **not** part of a faithful picture: measured on captured frames it
lifts the floor by 13 codes, compresses the peak to 247 and costs 0.24 of
saturation in shadow. This branch runs it only when asked, and compares images
without it.

### Known issue

In some scenes a run dies after four to ten minutes. A bisect, one build at a
time with the same configuration, put it on a guard added in this branch that
held guarded blocks open across the copy-back: builds without it ran fifteen
minutes clean three times, builds with it died in eight runs out of eight. The
guard is reverted here.

## What it is built on

This is a branch, and the work under it is other people's:

- **[VIctory](https://github.com/Marvisak/VIctory)** — the loader, the runtime
  services, the Direct3D 12 back end and the shader recompiler. GPL-3.0-or-later,
  as this branch is.
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)** — the
  fork this branch grows from; its status notes and switch list are the base of
  the numbers here.
- **`cool_fox` (the CGBJ group)** — the block pool for host textures, the
  copy-back by `rep movsb`, the parts mask for a page's 4 KiB writes, and the
  alias bookkeeping, all carried with GPL-3.0 attribution in each file's header.
- **Belobug** — the stand-in for indirect light this branch's `ambient` module
  descends from.
- **[XCL](https://github.com/ivall/xcl)** — read as a reference for how the same
  title behaves under another layer; fixes inspired by it were written anew here.

## Building and running

```
cargo build --release -p victory --features diagnostics
```

Copy `victory.example.toml` to `victory.toml` outside this folder and set your
own paths there. Then start the layer with the path of your copy's executable.
Every switch is described in a comment where it is read; the ones this branch
measures with are in `crates/victory-gpu/src` and are named `VICTORY_*`.

Nothing here downloads, decrypts or ships game data, and no switch makes it do
so.

## Credits and license

The full credit list — whose commit carried which fix, and the GPL-2.0-only
project read as a reference — is in the sections above. Upstream's history is
kept intact below the single commit that carries this branch's changes.

**GPL-3.0-or-later**, as VIctory. The one file that is not covered is named in
`docs/asset-policy.md`. This branch's changes were developed with
[Claude Code](https://claude.com/claude-code).
