<CsoundSynthesizer>
<CsOptions>
; Verification 10: Model attributes introspection & access (TorchScript .ts + .pte check)
-n
;--opcode-dir=~/Library/csound/7.0/plugins64
</CsOptions>
<CsInstruments>

sr     = 48000
ksmps  = 128
nchnls = 2
0dbfs  = 1

; 1. TorchScript model with registered nn_tilde attributes (learn_target, reset_target, etc.)
giTsModel netbender_load "models/voice_from_ckpt.ts", "forward", 2048, 0, 0
prints "\n--- TorchScript (.ts) Attribute Check ---\n"
netbender_dump giTsModel

gSTsAttrs[] netbender_attrs giTsModel, 1
giNumTsAttrs lenarray gSTsAttrs
prints "Settable attributes (%d):\n", giNumTsAttrs
giIdx = 0
while (giIdx < giNumTsAttrs) do
    SVal netbender_attr_get giTsModel, gSTsAttrs[giIdx]
    prints "  attr[%d] %s = '%s'\n", giIdx, gSTsAttrs[giIdx], SVal
    giIdx += 1
od

if (giNumTsAttrs > 0) then
    SBefore netbender_attr_get giTsModel, "learn_target"
    netbender_attr_set giTsModel, "learn_target", 1
    SAfterNum netbender_attr_get giTsModel, "learn_target"
    netbender_attr_set giTsModel, "learn_target", "false"
    SAfterStr netbender_attr_get giTsModel, "learn_target"
    prints "Toggled learn_target: initial='%s' -> after set(1)='%s' -> after set(\"false\")='%s'\n", \
           SBefore, SAfterNum, SAfterStr
endif
netbender_free giTsModel

; 2. ExecuTorch (.pte) model gracefully reports 0 attributes
giPteModel netbender_load "models/darbouka_xnnpack.pte", "forward", 2048, 0, 0
prints "\n--- ExecuTorch (.pte) Attribute Check ---\n"
gSPteAttrs[] netbender_attrs giPteModel, 1
giNumPteAttrs lenarray gSPteAttrs
prints "ExecuTorch settable attributes = %d (expected 0)\n", giNumPteAttrs
netbender_free giPteModel

instr 1
    turnoff
endin

</CsInstruments>
<CsScore>
i 1 0 0.05
</CsScore>
</CsoundSynthesizer>
