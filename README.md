# Evidence for the VC3D save/export fix (skip unchanged surface channels)

Replay of one correction drag in VC3D on two public scans. It compares the editor on Villa `main` (`ad0f864`, after ScrollPrize/villa#2563) with the
same build plus the one-file fix (`SurfaceStoreEdit.cpp`). Both editors come from the same tree and load the same
tracer library.

| File | What it shows |
| --- | --- |
| `0139_stock_status.png`, `0139_this_pr_status.png` | Status bar right after Save on PHerc 0139 (crop of the full window) |
| `0139_stock_window.png`, `0139_this_pr_window.png` | The full VC3D windows those crops come from |
| `1447_*` | The same for the PHerc 1447 crop |
| `results/main_only/` | Driver output with the tracer from `main` |
| `results/with_pr2666/` | Driver output with ScrollPrize/villa#2666 merged into the tracer |
| `fixture1447/` | 24 × 64 crop of a public PHerc 1447 mesh, its correction and run parameters |
| `verify_gui_anchor_save.py` | The replay driver (needs `verify_anchor_fix.py` next to it) |

The PHerc 0139 fixture is the one used for ScrollPrize/villa#2666:
<https://github.com/DeepakEz/villa/tree/b6ec47be3d857950bd861cbf2dec9a31340daea6/fixture>.

## What the driver does

It drives VC3D through its agent bridge (`--agent-bridge-name`):

1. Create a project and open the public scan.
2. Import the surface and drag one correction point in the XY view.
3. Wait for the solver, then Save and export.
4. Save again, then reopen the project in a fresh process and export.

It runs this once with each editor. It then checks the following:

- Without the fix, Save fails and export returns `-32005`.
- With the fix, Save succeeds and a repeat Save is idle.
- The immediate export equals the export after reopening, and the stock editor's reopened mesh, at every vertex.
- Geometry history and the correction link are unchanged by the repeat Save and the reopen.

```bash
python verify_gui_anchor_save.py <VC3D without the fix> <vc_grow_seg_from_seed> <dir containing patched/libvc_tracer.so> \
    <extra library dir> <fixture dir> <new output dir> --save-fix-gui <VC3D with the fix> --capture-save \
    [--volume <public zarr URL for PHerc 1447>] [--window-geometry 2300x1050+0+0]
```

We ran it headless under Xvfb with Mesa software rendering (`QT_QPA_PLATFORM=xcb`).

Scans are from the Vesuvius Challenge open data (CC BY-NC 4.0):

- PHerc 0139: `20250728140407-9.362um-1.2m-113keV-masked.zarr`
- PHerc 1447: `20250521151220-8.640um-1.2m-116keV-masked.zarr`

The PHerc 1447 crop comes from segment `20250502182142`. Its full source hashes are in `fixture1447/fixture.json`.
