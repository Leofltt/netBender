# NetBender Csound 7 Opcodes (`src/frontend/csound`)

Csound 7 plugin opcode frontend built on NetBender's self-contained `NetBender::backend` (`NetBenderSession`, `StreamingEngine`, and `ResamplingRingBufferBank`).

Supports `.pte` (ExecuTorch), `.onnx` (ONNX Runtime), and `.ts` (TorchScript) models with automatic host ↔ model sample-rate resampling, independence from Csound's `ksmps` (control period) and `-b`/`-B` I/O buffer sizes, and auto selection between asynchronous worker inference (`-odac`) and synchronous inline inference (`-o file.wav` / `-n`).

---

## Implemented Opcodes (Phase 1)

| Opcode | Signature | Description |
| :--- | :--- | :--- |
| `netbender_load` | `ihandle netbender_load Spath [, ibufsize, igpu, ithread, isentry, idrydelay]`<br>`ihandle netbender_load Spath, Smethod [, ibufsize, igpu, ithread, isentry, idrydelay]` | Loads a `.pte`/`.onnx`/`.ts` model once (typically in `instr 0`) and returns an integer handle. Defaults: `Smethod="forward"`, `ibufsize=-1` (backend preferred; `0` forces synchronous inline), `igpu=0`, `ithread=-1` (auto async for `-odac`, sync for file/null output), `isentry=-1` (auto on except `encode`/`decode`), `idrydelay=-1` (auto on for `forward`). |
| `netbender_free` | `netbender_free ihandle` | Explicitly closes and frees a loaded handle (all handles also auto-clean on Csound reset/exit). |
| `netbender_reset` | `netbender_reset ihandle`<br>`netbender_reset ihandle, ktrig` | Clears streaming ring buffers, `decode gen` FIFO, and resets the `SafetySentry` fuse (at i-time or on rising edge of `ktrig`). |
| `netbender_process` | `aout netbender_process ihandle, ain [, kdrywet=1, kenable=1]`<br>`aL, aR netbender_process ihandle, ainL, ainR [, kdrywet=1, kenable=1]`<br>`aOuts[] netbender_process ihandle, aIns[] [, kdrywet=1, kenable=1]` | Real-time neural audio or latent streaming (`forward`, `encode`, `decode`, etc.). Automatically resamples between Csound `sr` and model SR and normalizes by `0dbfs`. `aOuts[]` is auto-sized in `init()` to `model_out_channels`. |
| `netbender_gen` | `aout netbender_gen ihandle [, kenable=1]`<br>`aL, aR netbender_gen ihandle [, kenable=1]`<br>`aOuts[] netbender_gen ihandle [, kenable=1]` | 0-input generative synthesis (`decode gen`, `prior`, `generate`). |
| `netbender_info` | `iInCh, iOutCh, iModelSR, iLatency, iNumLayers netbender_info ihandle` | Returns model input/output channels, native model SR, host-side latency in samples, and number of bendable layers. |
| `netbender_dump` | `netbender_dump ihandle` | Prints model diagnostics, resampling state, methods, attributes, and bendable layers/shapes to the Csound console. |
| `netbender_layers` | `SLayers[] netbender_layers ihandle` | Returns an `S[]` array containing all bendable layer names. |
| `netbender_methods` | `SMethods[] netbender_methods ihandle` | Returns an `S[]` array containing available model methods/modes. |

## Implemented Opcodes (Phase 2 — Live Circuit Bending & Weights)

| Opcode | Signature | Description |
| :--- | :--- | :--- |
| `netbender_bend` | `netbender_bend ihandle, Slayer, kscale, koffset [, kheat=0, kmemory=-1, kdriftmode=-1, kfrozen=0]`<br>`netbender_bend ihandle, ilayer, kscale, koffset [, kheat=0, kmemory=-1, kdriftmode=-1, kfrozen=0]` | Wait-free k-rate parametric weight bending per layer (supports multiple layers simultaneously). Defaults: `kheat=0` (`>= 0`), `kmemory=-1` (`0..1`, `-1` -> `0.8`), `kdriftmode=-1` (`0`=Noise, `1`=ThermalOU, `2`=RandomWalk, `-1` -> `1`), `kfrozen=0` (`0\|1` locks current drift). |
| `netbender_short` | `netbender_short ihandle, kactive` | Wait-free k-rate momentary/short-circuit bend gate (`0\|1`). Drives `BendingManager::trigger_short_circuit` in `Momentary` trigger mode. |
| `netbender_trigger` | `netbender_trigger ihandle, imode [, iatt=-1, irel=-1, ithresh=-1]`<br>`netbender_trigger ihandle, kmode [, katt=-1, krel=-1, kthresh=-1]` | Configures bend trigger mode (`0`=Continuous, `1`=Momentary, `2`=Transient) and optional envelope parameters (`attack` `0.01..1`, `release` `0.5..0.999`, `threshold` `0.01..1`; negative keeps current value). |
| `netbender_bridge` | `netbender_bridge ihandle, Starget, Ssource, kdepth`<br>`netbender_bridge ihandle, itarget, isource, kdepth` | Wait-free cross-talk bridge blending a target layer's base weights with a source layer (`kdepth` in `0..1`). Pass empty `Ssource=""` or `isource=-1` to disconnect. |
| `netbender_latent` | `netbender_latent ihandle, ienable, ix, iy, idepth [, imode=0, islew=0]`<br>`netbender_latent ihandle, kenable, kx, ky, kdepth [, kmode=0, kslew=0]` | Controls `LatentTerrainEngine` (`enable` `0\|1`, `x/y` in `-1..1`, `depth` in `0..2`, `mode`: `0`=FourierOrbit, `1`=LatentSlew, `2`=Quantize, `slew` in `0..0.999`). Runs during `forward` when enabled and always during `decode gen`. |
| `netbender_sentry` | `kblown netbender_sentry ihandle [, kautoreset=-1, kresettrig=0]` | Monitors `SafetySentry` fuse state (`1` if tripped/bypassed, `0` if healthy). Optional `kautoreset` (`0\|1`, `<0` leaves unchanged) toggles auto-recovery; rising edge on `kresettrig` performs a lock-free fuse + full weight reset. |
| `netbender_getweights` | `iWeights[] netbender_getweights ihandle, Slayer [, ioriginal=0]`<br>`iWeights[] netbender_getweights ihandle, ilayer [, ioriginal=0]` | Reads a flattened `i[]` tensor of a layer's weights (`0`=live/current weights, `1`=original unbent weights). |
| `netbender_setweights` | `netbender_setweights ihandle, Slayer, iWeights[]`<br>`netbender_setweights ihandle, ilayer, iWeights[]`<br>`netbender_setweights ihandle, Slayer, kWeights[], ktrig`<br>`netbender_setweights ihandle, ilayer, kWeights[], ktrig` | Injects raw flattened weights into a layer at i-time or via wait-free SPSC triple buffering on a `ktrig` rising edge. Array length must match the layer's element count. |
| `netbender_resetweights` | `netbender_resetweights ihandle`<br>`netbender_resetweights ihandle, Slayer`<br>`netbender_resetweights ihandle, ktrig`<br>`netbender_resetweights ihandle, Slayer, ktrig`<br>`netbender_resetweights ihandle, ilayer, ktrig` | Resets one layer or all layers back to baseline weights (`i` / `iS` at init time; `ik` / `iSk` / `iik` on rising edge of `ktrig`, where `ilayer < 0` resets all layers). |

### Model Path Resolution Order

`netbender_load` searches in the following order (probing `.pte`, `.onnx`, `.ts` automatically if `Spath` omits an extension):
1. Exact / relative-to-current-working-directory / absolute path.
2. Csound's `FindInputFile` (`SSDIR`, `SFDIR`, `INCDIR`).
3. `~/.netbender/models` (`%USERPROFILE%\.netbender\models` on Windows).

### Recommended Real-Time Flags (`-b` / `-B` / `ksmps`)

Csound separates control timing from audio I/O buffering:
- **`ksmps`** (`sr / kr`): control-rate period in frames (e.g. `64` or `128`), i.e. how many samples each `aperf()` call processes.
- **`-b <frames>`**: Csound's software I/O buffer size (read from `OPARMS::outbufsamps`). `NetBender` sizes its `StreamingEngine` rings and resampler scratch to accommodate `-b` bursts even when `ksmps` is small. Keep `-b <= ibufsize` (for example `-b 256` with `ibufsize = 2048`) so worker-thread bursts never overflow a model cycle.
- **`-B <frames>`**: Csound's hardware audio I/O buffer size (read from `OPARMS::oMaxLag`, e.g. `-B 1024`).

Example real-time `<CsOptions>`:
```text
-odac -b 256 -B 1024
```

---

## Building & Installing

The Csound frontend is isolated under `src/frontend/csound` and is only built when you build the `csound` (or `all`) target with Csound 7 installed (`-DBUILD_CSOUND=ON`, enabled by default and skipped cleanly if Csound 7 is absent):

```bash
# Configure (pass -DCsound_DIR=/path/to/csound/cmake if Csound 7 is in a custom location)
cmake -S src -B build \
    -DCMAKE_PREFIX_PATH="$(pwd)/libtorch" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CSOUND=ON

# Build only the Csound 7 opcode library (plus NetBender::backend)
cmake --build build --target csound -j

# Install into Csound 7's user plugin directory:
#   macOS:   ~/Library/csound/7.0/plugins64
#   Linux:   ~/.local/lib/csound/7.0/plugins64
#   Windows: %LOCALAPPDATA%/csound/7.0/plugins64
cmake --install build --component csound

# Optional: build & run the Csound adapter unit test
cmake --build build --target csound_tests -j
ctest --test-dir build -R test_csound_adapter --output-on-failure
```

For redistributable / self-contained builds (`-DBUNDLE_DEPENDENCIES=ON`), dependent shared libraries are placed in a `netbender_libs/` subdirectory next to the plugin (loaded via `@loader_path/netbender_libs` / `$ORIGIN/netbender_libs` / `AddDllDirectory`) so Csound does not attempt to `dlopen` Torch/ONNX/ExecuTorch libraries directly as opcode plugins.

### macOS Hardened Runtime Note

If `csound` fails to load `libnetbender.dylib` with `WARNING: could not open library '.../7.0/plugins64/libnetbender.dylib' (-1)` (visible with `csound -v -z`), the Csound 7 macOS installer's `/Applications/Csound/csound` binary is signed with Hardened Runtime without the `com.apple.security.cs.disable-library-validation` entitlement, which blocks locally built `.dylib` plugins.

Re-sign the CLI binary locally:

```sh
sudo codesign --force --sign - /Applications/Csound/csound
```

(Reinstalling or updating Csound 7 overwrites `/Applications/Csound/csound`, so re-run this command after updates.)

---

## Verifying With the Example `.csd` Files

Run from the repository root so relative paths to `models/*` resolve:

```bash
# 1. Confirm Csound 7 loads the plugin and registers netbender_* opcodes
/Applications/Csound/csound -v -z 2>&1 | grep -E "netbender"

# 2. Introspection & metadata (runs with -n, no audio hardware needed)
/Applications/Csound/csound src/frontend/csound/examples/01_info_introspection.csd

# 3. Forward mono & stereo streaming with 44.1 kHz <-> 48 kHz resampling (realtime or disk render)
/Applications/Csound/csound src/frontend/csound/examples/02_process_forward_mono_stereo.csd
/Applications/Csound/csound -o forward_demo.wav src/frontend/csound/examples/02_process_forward_mono_stereo.csd

# 4. Split encode -> audio-rate latent modulation -> decode via a[] arrays
/Applications/Csound/csound src/frontend/csound/examples/03_process_encode_decode_array.csd

# 5. Generative synthesis (decode gen / prior)
/Applications/Csound/csound src/frontend/csound/examples/04_gen_generative.csd

# 6. Multi-layer parametric bending, cross-layer bridge & weight reset
/Applications/Csound/csound src/frontend/csound/examples/05_bend_parametric_and_bridge.csd

# 7. Momentary/transient triggers, short-circuit gate & SafetySentry fuse recovery
/Applications/Csound/csound src/frontend/csound/examples/06_trigger_short_and_sentry.csd

# 8. Latent terrain exploration (FourierOrbit / LatentSlew / Quantize)
/Applications/Csound/csound src/frontend/csound/examples/07_latent_terrain.csd

# 9. Raw weight inspection & i-rate / k-rate injection (getweights / setweights / resetweights)
/Applications/Csound/csound src/frontend/csound/examples/08_raw_weights_read_write.csd
```
