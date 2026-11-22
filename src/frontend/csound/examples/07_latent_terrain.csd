<CsoundSynthesizer>
<CsOptions>
; Verification 07: Latent Terrain Engine across FourierOrbit (0), LatentSlew (1) & Quantize (2)
; Swap -odac with -o latent_terrain_demo.wav for deterministic offline rendering.
-odac -b 256 -B 1024
;--opcode-dir=~/Library/csound/7.0/plugins64
</CsOptions>
<CsInstruments>

sr     = 48000
ksmps  = 128
nchnls = 2
0dbfs  = 1

giGen netbender_load "models/darbouka_xnnpack.pte", "decode gen", 2048, 0, -1
giFwd netbender_load "models/darbouka_xnnpack.pte", "forward",    2048, 0, -1

instr 1
    ; 0-input generative synthesis steered along a 2D circular FourierOrbit in latent space
    kPh    phasor 0.25
    kAngle = kPh * 6.2831853
    kX     = 0.85 * cos(kAngle)
    kY     = 0.85 * sin(kAngle)
    kDepth linseg 0.2, p3 * 0.5, 1.4, p3 * 0.5, 0.2

    netbender_latent giGen, 1, kX, kY, kDepth, 0, 0.0

    aGen netbender_gen giGen, 1
    out aGen * 0.6, aGen * 0.6
endin

instr 2
    ; Audio forward mode stepping through all 3 latent modes every 2 seconds
    kPulse metro 4
    aSrc = (pinkish(0.4) + oscili(0.5, 110 + 150 * port(kPulse, 0.004))) * port(kPulse, 0.003) * 0.8

    kTime  timeinsts
    kMode  = (kTime < 2.0 ? 0 : (kTime < 4.0 ? 1 : 2))
    kX     = 0.7 * lfo(1.0, 0.5, 0)
    kY     = 0.7 * lfo(1.0, 0.35, 1)
    kDepth = (kMode == 2 ? 1.25 : 0.9)
    kSlew  = (kMode == 1 ? 0.92 : 0.0)

    kChg changed kMode
    printf "  -> Latent mode=%d (0=FourierOrbit, 1=LatentSlew, 2=Quantize) at t=%.2fs\n", \
           kChg, kMode, kTime

    netbender_latent giFwd, 1, kX, kY, kDepth, kMode, kSlew

    aOut netbender_process giFwd, aSrc, 1.0, 1
    out aOut, aOut
endin

</CsInstruments>
<CsScore>
i 1 0   5.0
i 2 5.5 6.0
</CsScore>
</CsoundSynthesizer>
