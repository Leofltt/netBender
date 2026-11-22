<CsoundSynthesizer>
<CsOptions>
; Verification 08: Raw weight inspection & injection (netbender_getweights / setweights / resetweights)
; Swap -odac with -o raw_weights_demo.wav for deterministic offline rendering.
-odac -b 256 -B 1024
;--opcode-dir=~/Library/csound/7.0/plugins64
</CsOptions>
<CsInstruments>

sr     = 48000
ksmps  = 128
nchnls = 2
0dbfs  = 1

giModel netbender_load "models/darbouka_xnnpack.pte", "forward", 2048, 0, -1

; Read layer 3 (_rave.decoder.synth.branches.0.bias, 16 weights) at i-time
giOrigW[] netbender_getweights giModel, 3, 1
giLen     lenarray giOrigW
prints "Layer 3 original weight count = %d (first 4: %.4f, %.4f, %.4f, %.4f)\n", \
       giLen, giOrigW[0], giOrigW[1], giOrigW[2], giOrigW[3]

; Modify array in Csound and upload at i-time
giModW[] init giLen
giIdx = 0
while (giIdx < giLen) do
    giModW[giIdx] = giOrigW[giIdx] * 2.5 + 0.15 * cos(giIdx * 0.5)
    giIdx += 1
od
netbender_setweights giModel, 3, giModW

; Read back live weights by layer name (ioriginal=0) to confirm injection
giLiveW[] netbender_getweights giModel, "_rave.decoder.synth.branches.0.bias", 0
prints "After i-rate setweights (first 4: %.4f, %.4f, %.4f, %.4f)\n", \
       giLiveW[0], giLiveW[1], giLiveW[2], giLiveW[3]

instr 1
    ; Periodic k-rate raw weight mutation triggered every 0.5s on layer 3
    kPulse metro 4
    aSrc = (pinkish(0.4) + oscili(0.5, 110)) * port(kPulse, 0.004) * 0.85

    kMutW[] init giLen
    kTrig   metro 2
    kStep   init 0
    kStep   += kTrig

    if (kTrig == 1) then
        kIdx = 0
        while (kIdx < giLen) do
            kMul = 1.0 + 1.5 * sin(kStep * 0.7 + kIdx * 0.4)
            kMutW[kIdx] = giOrigW[kIdx] * kMul + 0.1 * cos(kStep + kIdx)
            kIdx += 1
        od
    endif
    printf "  -> k-rate netbender_setweights step=%d\n", kTrig, kStep
    netbender_setweights giModel, 3, kMutW, kTrig

    ; Reset layer 3 back to original weights at t = 3.5s
    kTime timeinsts
    kResetTrig trigger kTime, 3.5, 0
    printf "  -> k-rate netbender_resetweights layer 3 at t=%.2fs\n", kResetTrig, kTime
    netbender_resetweights giModel, 3, kResetTrig

    aOut netbender_process giModel, aSrc, 1.0, 1
    out aOut, aOut
endin

instr 99
    ; Explicit i-time reset all & readback verification
    netbender_resetweights giModel
    iRestored[] netbender_getweights giModel, "_rave.decoder.synth.branches.0.bias", 0
    prints "After i-rate resetweights (first 4: %.4f, %.4f, %.4f, %.4f)\n", \
           iRestored[0], iRestored[1], iRestored[2], iRestored[3]
    turnoff
endin

</CsInstruments>
<CsScore>
i 1  0   4.8
i 99 5.0 0.1
</CsScore>
</CsoundSynthesizer>
