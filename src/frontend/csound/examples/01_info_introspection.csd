<CsoundSynthesizer>
<CsOptions>
; Verification 01: Model loading, introspection, and handle lifecycle (no audio output needed)
-n
;--opcode-dir=~/Library/csound/7.0/plugins64
</CsOptions>
<CsInstruments>

sr     = 48000
ksmps  = 128
nchnls = 2
0dbfs  = 1

; Load a model once at orchestra init (instr 0)
; Signature: ihandle netbender_load Spath [, ibufsize, igpu, ithread, isentry, idrydelay]
;        or: ihandle netbender_load Spath, Smethod [, ibufsize, igpu, ithread, isentry, idrydelay]
giModel netbender_load "models/darbouka_xnnpack.pte"

; 1. Query scalar summary values
giInCh, giOutCh, giModelSR, giLatency, giNumLayers netbender_info giModel
prints "netbender_info -> in_ch=%d out_ch=%d model_sr=%.0f host_latency=%d bendable_layers=%d\n", \
       giInCh, giOutCh, giModelSR, giLatency, giNumLayers

; 2. Dump full model diagnostics to Csound console
netbender_dump giModel

; 3. Retrieve available methods as S[]
gSMethods[] netbender_methods giModel
giNumMethods lenarray gSMethods
prints "Available methods (%d):\n", giNumMethods
giIdx = 0
while (giIdx < giNumMethods) do
    prints "  method[%d] = %s\n", giIdx, gSMethods[giIdx]
    giIdx += 1
od

; 4. Retrieve bendable layer names as S[]
gSLayers[] netbender_layers giModel
giNumLayArr lenarray gSLayers
prints "Bendable layers (%d):\n", giNumLayArr
giIdx = 0
while (giIdx < giNumLayArr) do
    prints "  layer[%d] = %s\n", giIdx, gSLayers[giIdx]
    giIdx += 1
od

instr 1
    ; Nothing to synthesize in this introspection-only verification instrument
    turnoff
endin

instr 99
    ; Verify explicit handle cleanup (handles are also auto-cleaned on Csound reset/exit)
    netbender_free giModel
    prints "Freed giModel cleanly.\n"
    turnoff
endin

</CsInstruments>
<CsScore>
i 1  0    0.1
i 99 0.15 0.05
</CsScore>
</CsoundSynthesizer>
