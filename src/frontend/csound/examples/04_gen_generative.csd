<CsoundSynthesizer>
<CsOptions>
; Verification 04: 0-input generative synthesis via netbender_gen (mono, stereo, and a[] overloads)
; Swap -odac with -o gen_demo.wav for deterministic offline rendering.
-odac
;--opcode-dir=~/Library/csound/7.0/plugins64
</CsOptions>
<CsInstruments>

sr     = 48000
ksmps  = 128
nchnls = 2
0dbfs  = 1

; Use "decode gen" for RAVE/VAE models with a decoder, or "prior" for models exporting a prior
; (e.g. giGen netbender_load "models/darbouka.ts", "prior", 2048)
giGen netbender_load "models/darbouka_xnnpack.pte", "decode gen", 2048, 0, -1

netbender_dump giGen

instr 1
    ; Mono overload
    aOut netbender_gen giGen, 1
    out aOut * 0.6, aOut * 0.6
endin

instr 2
    ; Stereo overload with kenable gating + periodic reset
    kTrig metro 0.5
    netbender_reset giGen, kTrig
    aL, aR netbender_gen giGen, 1
    out aL * 0.6, aR * 0.6
endin

instr 3
    ; Multichannel a[] array overload
    aOut[] netbender_gen giGen, 1
    out aOut[0] * 0.6, aOut[0] * 0.6
endin

</CsInstruments>
<CsScore>
i 1 0  3
i 2 3.5 3
i 3 7  3
</CsScore>
</CsoundSynthesizer>
