<CsoundSynthesizer>
<CsOptions>
; Verification 05: Multi-layer live circuit bending (by name & index), cross-layer bridge & resets
; Swap -odac with -o bend_bridge_demo.wav for deterministic offline rendering.
-odac -b 256 -B 1024
;--opcode-dir=~/Library/csound/7.0/plugins64
</CsOptions>
<CsInstruments>

sr     = 48000
ksmps  = 128
nchnls = 2
0dbfs  = 1

giModel netbender_load "models/darbouka_xnnpack.pte", "forward", 2048, 0, -1

instr 1
    ; Synthesized percussive source
    kPulse metro 4
    aSrc = (pinkish(0.4) + oscili(0.5, 115 + 180 * port(kPulse, 0.005))) * port(kPulse, 0.003) * 0.85

    ; Continuous stochastic thermal drift on decoder layer 2 (by name)
    kTime   timeinsts
    kScale0 linseg 1.0, 2.0, 2.2, 2.0, 1.0, p3 - 4.0, 1.0
    kOff0   = 0.05 * lfo(1.0, 0.4, 0)
    kHeat0  = (kTime > 1.0 && kTime < 4.5 ? 0.12 : 0.0)
    kFreeze = (kTime > 2.5 && kTime < 3.5 ? 1 : 0)
    netbender_bend giModel, "_rave.decoder.net.0.bias", kScale0, kOff0, kHeat0, 0.85, 1, kFreeze

    ; Simultaneous bend on layer index 3 (_rave.decoder.synth.branches.0.bias) with Noise drift (mode=0)
    kScale1 = 1.0 + 0.6 * lfo(1.0, 0.75, 1)
    netbender_bend giModel, 3, kScale1, 0.0, 0.05, 0.7, 0, 0

    ; Cross-talk bridge blending "_rave.decoder.net.0.bias" with "onnx::Conv_918"
    kBridgeDepth linseg 0.0, 1.5, 0.65, 2.5, 0.65, 1.0, 0.0, p3 - 5.0, 0.0
    netbender_bridge giModel, "_rave.decoder.net.0.bias", "onnx::Conv_918", kBridgeDepth

    ; k-rate trigger resets all layers at t = 4.5s back to baseline weights
    kResetTrig trigger kTime, 4.5, 0
    printf "  -> k-rate netbender_resetweights all at t=%.2fs\n", kResetTrig, kTime
    netbender_resetweights giModel, kResetTrig

    aOut netbender_process giModel, aSrc, 1.0, 1
    out aOut, aOut
endin

instr 2
    ; Second note: bridge by indices (target=3, source=8) and per-layer reset by index
    netbender_resetweights giModel, "_rave.decoder.net.0.bias"

    kPulse metro 3
    aSrc = (pinkish(0.35) + oscili(0.45, 95)) * port(kPulse, 0.004) * 0.8

    kTime timeinsts
    kBridgeDepth2 = (kTime < 2.5 ? 0.5 : 0.0)
    netbender_bend giModel, 3, 1.8, 0.1, 0.04, 0.8, 1, 0
    netbender_bridge giModel, 3, 8, kBridgeDepth2

    kRstLayer trigger kTime, 3.0, 0
    printf "  -> k-rate netbender_resetweights layer 3 at t=%.2fs\n", kRstLayer, kTime
    netbender_resetweights giModel, 3, kRstLayer

    aOut netbender_process giModel, aSrc, 1.0, 1
    out aOut, aOut
endin

</CsInstruments>
<CsScore>
i 1 0   5.5
i 2 6.0 4.5
</CsScore>
</CsoundSynthesizer>
