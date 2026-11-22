<CsoundSynthesizer>
<CsOptions>
; Verification 09: JSON preset save & load (interoperable with JUCE NetBenderSession presets)
; Swap -odac with -o preset_demo.wav for deterministic offline rendering.
-odac -b 256 -B 1024
;--opcode-dir=~/Library/csound/7.0/plugins64
</CsOptions>
<CsInstruments>

sr     = 48000
ksmps  = 128
nchnls = 2
0dbfs  = 1

giModel netbender_load "models/darbouka_xnnpack.pte", "forward", 2048, 0, -1
gSJson  init ""

; Prepare a reference raw weight vector for layer 3 (_rave.decoder.synth.branches.0.bias)
giOrigW[] netbender_getweights giModel, 3, 1
giLen     lenarray giOrigW
giBendW[] init giLen
giIdx = 0
while (giIdx < giLen) do
    giBendW[giIdx] = giOrigW[giIdx] * 2.25 + 0.12 * sin(giIdx * 0.5)
    giIdx += 1
od

prints "Layer 3 original[0..1] = %.4f, %.4f\n", giOrigW[0], giOrigW[1]

instr 1
    ; Phase A (0.0 .. 2.0s): Apply multi-layer bend + bridge + latent terrain + trigger + raw weights
    netbender_setweights giModel, 3, giBendW
    netbender_trigger    giModel, 0, 0.25, 0.85, 0.15
    netbender_latent     giModel, 1, 0.45, -0.35, 0.75, 0, 0.15

    kPulse metro 4
    aSrc = (pinkish(0.4) + oscili(0.5, 110)) * port(kPulse, 0.004) * 0.85
    netbender_bend   giModel, 2, 1.65, -0.04, 0.05, 0.8, 1, 0
    netbender_bridge giModel, 2, 8, 0.35
    aOut netbender_process giModel, aSrc, 0.85, 1
    out aOut, aOut
endin

instr 2
    ; Phase B (2.1s): Snapshot preset to JSON file AND in-memory string, then reset all weights
    netbender_savestate giModel, "/tmp/netbender_demo_preset.json", 1
    gSJson netbender_savestate giModel, 1
    prints "Saved preset to /tmp/netbender_demo_preset.json (JSON chars=%d)\n", strlen(gSJson)

    netbender_resetweights giModel
    iRstW[] netbender_getweights giModel, 3, 0
    prints "After netbender_resetweights: layer 3[0..1] = %.4f, %.4f\n", iRstW[0], iRstW[1]
    turnoff
endin

instr 3
    ; Phase C (2.3 .. 4.3s): Load preset back from disk at i-time and audition restored state
    netbender_loadstate giModel, "/tmp/netbender_demo_preset.json", 0
    iFileW[] netbender_getweights giModel, 3, 0
    prints "After file netbender_loadstate: layer 3[0..1] = %.4f, %.4f (expected %.4f, %.4f)\n", \
           iFileW[0], iFileW[1], giBendW[0], giBendW[1]

    kPulse metro 4
    aSrc = (pinkish(0.4) + oscili(0.5, 110)) * port(kPulse, 0.004) * 0.85
    aOut netbender_process giModel, aSrc, 0.85, 1
    out aOut, aOut
endin

instr 4
    ; Phase D (4.5s): Reset again, then restore directly from in-memory JSON string gSJson
    netbender_resetweights giModel
    netbender_loadstate  giModel, gSJson, 0
    iStrW[] netbender_getweights giModel, 3, 0
    prints "After string netbender_loadstate: layer 3[0..1] = %.4f, %.4f (expected %.4f, %.4f)\n", \
           iStrW[0], iStrW[1], giBendW[0], giBendW[1]
    turnoff
endin

</CsInstruments>
<CsScore>
i 1 0.0 2.0
i 2 2.1 0.1
i 3 2.3 2.0
i 4 4.5 0.1
</CsScore>
</CsoundSynthesizer>
