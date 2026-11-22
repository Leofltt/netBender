<CsoundSynthesizer>
<CsOptions>
; Verification 02: Forward streaming inference (mono + stereo overloads), resampling, dry/wet, enable & reset
; Swap -odac with -o forward_demo.wav for deterministic offline rendering.
; -b: Csound software I/O buffer (frames), -B: hardware buffer (frames) — distinct from ksmps (control rate).
-odac -b 256 -B 1024
;--opcode-dir=~/Library/csound/7.0/plugins64
</CsOptions>
<CsInstruments>

; Deliberately 44.1 kHz to verify automatic host <-> model resampling against 48 kHz RAVE models
sr     = 44100
ksmps  = 128
nchnls = 2
0dbfs  = 1

; Default method ("forward"), auto buffer size, CPU, auto worker thread (async for -odac / sync for -o file)
giModelMono   netbender_load "models/darbouka_xnnpack.pte", "forward", 2048, 0, -1, 1, 1
giModelStereo netbender_load "models/darbouka_xnnpack.pte"

instr 1
    ; Synthesized percussive exciter so this .csd runs standalone with no external WAV needed
    kTrig metro 4
    kEnv  expon 1, 0.14, 0.001
    kEnv  = (kTrig == 1 ? 1 : kEnv)
    aSrc  = (pinkish(0.45) + oscili(0.55, 110 + 240 * kEnv)) * port(kTrig, 0.003) * 0.85

    ; Crossfade dry -> wet over the first 1.5s, stay wet, fade back to dry at the end
    kDryWet linseg 0, 1.5, 1, p3 - 3.0, 1, 1.5, 0
    kEnable = 1

    ; Trigger a stream reset halfway through
    kResetTrig = metro(0.25)
    netbender_reset giModelMono, kResetTrig

    ; Mono overload: aout netbender_process ihandle, ain [, kdrywet=1, kenable=1]
    aOut netbender_process giModelMono, aSrc, kDryWet, kEnable

    out aOut, aOut
endin

instr 2
    ; Stereo overload with a mono-to-stereo model (automatically duplicates mono model output to L/R)
    kTrigL metro 3
    kTrigR metro 5
    aInL = oscili(0.4, 165) * port(kTrigL, 0.005) + pinkish(0.15)
    aInR = oscili(0.4, 220) * port(kTrigR, 0.005) + pinkish(0.15)

    ; Reset at i-time before starting note
    netbender_reset giModelStereo

    ; Bypass neural computation for the middle 1s to verify kenable gating
    kTime timeinsts
    kEnable = (kTime > 2.0 && kTime < 3.0 ? 0 : 1)

    ; Stereo overload: aL, aR netbender_process ihandle, ainL, ainR [, kdrywet=1, kenable=1]
    aOutL, aOutR netbender_process giModelStereo, aInL, aInR, 1.0, kEnable

    out aOutL, aOutR
endin

</CsInstruments>
<CsScore>
i 1 0  6
i 2 7  5
</CsScore>
</CsoundSynthesizer>
