<CsoundSynthesizer>
<CsOptions>
; Verification 03: Split encode -> Csound audio-rate latent manipulation -> decode via a[] array overload
; Swap -odac with -o encode_decode_demo.wav for deterministic offline rendering.
; -b: Csound software I/O buffer (frames), -B: hardware buffer (frames) — distinct from ksmps (control rate).
-odac -b 256 -B 1024
;--opcode-dir=~/Library/csound/7.0/plugins64
</CsOptions>
<CsInstruments>

sr     = 48000
ksmps  = 128
nchnls = 2
0dbfs  = 1

; Load separate encode and decode instances of the same model
giEnc netbender_load "models/darbouka_xnnpack.pte", "encode", 2048, 0, -1
giDec netbender_load "models/darbouka_xnnpack.pte", "decode", 2048, 0, -1

netbender_dump giEnc
netbender_dump giDec

instr 1
    kTrig metro 3.5
    aIn[] init 1
    aIn[0] = (oscili(0.5, 110) + pinkish(0.3)) * port(kTrig, 0.004) * 0.8

    ; Encode 1-channel audio into aLatents[] (automatically sized to latent_dim = 4)
    aLatents[] netbender_process giEnc, aIn

    ; Modulate latent dimensions directly with standard Csound audio-rate opcodes
    kDepth linseg 0.0, 2.0, 1.2, p3 - 4.0, 1.2, 2.0, 0.0
    aLatents[0] = aLatents[0] + lfo(kDepth, 0.35, 0)
    aLatents[1] = aLatents[1] * (1.0 + 0.5 * lfo(kDepth, 0.5, 1))

    ; Decode modified latent streams back into audio
    aOut[] netbender_process giDec, aLatents

    out aOut[0], aOut[0]
endin

</CsInstruments>
<CsScore>
i 1 0 8
</CsScore>
</CsoundSynthesizer>
