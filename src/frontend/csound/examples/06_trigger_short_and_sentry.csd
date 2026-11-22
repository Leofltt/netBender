<CsoundSynthesizer>
<CsOptions>
; Verification 06: Trigger modes (Momentary/Transient), short-circuit gate, and SafetySentry recovery
; Swap -odac with -o trigger_sentry_demo.wav for deterministic offline rendering.
-odac -b 256 -B 1024
;--opcode-dir=~/Library/csound/7.0/plugins64
</CsOptions>
<CsInstruments>

sr     = 48000
ksmps  = 128
nchnls = 2
0dbfs  = 1

giModel netbender_load "models/darbouka_xnnpack.pte", "forward", 2048, 0, -1, 1, 1

instr 1
    ; Mode 1 = Momentary, controlled by netbender_short gate pulses
    netbender_resetweights giModel
    netbender_trigger giModel, 1, 0.35, 0.82, 0.15

    kPulse metro 4
    aSrc = (pinkish(0.4) + oscili(0.55, 130)) * port(kPulse, 0.004) * 0.85

    ; Bend target weights strongly so short-circuit flashes are clearly audible
    netbender_bend giModel, "_rave.decoder.net.0.bias", 3.0, 0.25, 0.15, 0.8, 1, 0

    kPh       phasor 1.5
    kShort    = (kPh < 0.25 ? 1 : 0)
    netbender_short giModel, kShort

    aOut netbender_process giModel, aSrc, 1.0, 1
    out aOut, aOut
endin

instr 2
    ; Mode 2 = Transient sidechain trigger (only loud bursts cross threshold = 0.25)
    netbender_resetweights giModel
    netbender_trigger giModel, 2, 0.5, 0.88, 0.25

    kBeat  metro 2
    kCount init 0
    kCount += kBeat
    kAmp   = ((kCount % 2) == 0 ? 0.85 : 0.08)
    aSrc   = (pinkish(0.35) + oscili(0.5, 110)) * port(kBeat, 0.004) * kAmp

    netbender_bend giModel, "_rave.decoder.net.0.bias", 2.8, -0.2, 0.1, 0.8, 1, 0

    aOut netbender_process giModel, aSrc, 1.0, 1
    out aOut, aOut
endin

instr 3
    ; SafetySentry verification: extreme bend trips fuse -> dry bypass -> manual lock-free reset
    netbender_resetweights giModel
    netbender_trigger giModel, 0

    kPulse metro 4
    aSrc = (pinkish(0.35) + oscili(0.45, 165)) * port(kPulse, 0.004) * 0.7

    kTime  timeinsts
    kExtremeScale = (kTime > 0.6 && kTime < 2.2 ? 1.0e6 : 1.0)
    netbender_bend giModel, "_rave.decoder.net.0.bias", kExtremeScale, 0.0, 0.0, 0.8, 1, 0

    kManualReset trigger kTime, 2.4, 0
    ; Disable auto-reset (0) so blown state stays visible until manual reset pulse at 2.4s
    kBlown netbender_sentry giModel, 0, kManualReset
    kChg   changed kBlown
    printf "  -> SafetySentry kBlown=%d at t=%.2fs\n", kChg, kBlown, kTime

    aOut netbender_process giModel, aSrc, 1.0, 1
    out aOut, aOut
endin

</CsInstruments>
<CsScore>
i 1 0   4.0
i 2 4.5 4.0
i 3 9.0 4.0
</CsScore>
</CsoundSynthesizer>
