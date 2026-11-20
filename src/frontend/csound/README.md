# NetBender Csound 7 Opcodes (`src/frontend/csound`)

Csound 7 plugin opcode frontend built on NetBender's self-contained `NetBender::backend` (`NetBenderSession`, `StreamingEngine`, and `ResamplingRingBufferBank`).

Supports `.pte` (ExecuTorch), `.onnx` (ONNX Runtime), and `.ts` (TorchScript) models with automatic host ↔ model sample-rate resampling, arbitrary `ksmps`, and auto selection between asynchronous worker inference (`-odac`) and synchronous inline inference (`-o file.wav` / `-n`).

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

### Model Path Resolution Order

`netbender_load` searches in the following order (probing `.pte`, `.onnx`, `.ts` automatically if `Spath` omits an extension):
1. Exact / relative-to-current-working-directory / absolute path.
2. Csound's `FindInputFile` (`SSDIR`, `SFDIR`, `INCDIR`).
3. `~/.netbender/models` (`%USERPROFILE%\.netbender\models` on Windows).

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
```
