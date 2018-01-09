% diagrams.mp -- do not edit, generated automatically by learning_higgs_eft_notes.tex
input feynmp
require_RCS_revision "1.30";
beginchar(1, 150*1.0pt#, 70*1.0pt#, 0);
"feynmf: 1";
LaTeX_unitlength:=1.0pt;
subgraph (0, 0, w, h);
vinit;
pickup pencircle scaled thin;
pickup pencircle scaled 0.8pt;
save arrow_len; arrow_len:=2mm;
vleft (__i2, __i1);
vright (__o6, __o5, __o4, __o3, __o2, __o1);
vlabel ("\small $q$", __i1);
vlabel ("\small $q$", __i2);
vlabel ("\small $q$", __o1);
vlabel ("\small $\ell ^+$", __o2);
vlabel ("\small $\ell ^-$", __o3);
vlabel ("\small $\ell ^+$", __o4);
vlabel ("\small $\ell ^-$", __o5);
vlabel ("\small $q$", __o6);
vconnect ("fermion,tension=4", __i1, __v3);
vconnect ("fermion,tension=4", __i2, __v4);
vconnect ("fermion,tension=2.5", __v3, __o1);
vconnect ("fermion,tension=2.5", __v4, __o6);
vconnect ("wiggly,label=\small $W$,, $Z$,label.side=right", __v3, __v5);
vconnect ("wiggly,label=\small $W$,, $Z$,label.side=left", __v4, __v5);
vconnect ("dashes,label=\small $h$,tension=0.5", __v5, __v6);
vconnect ("wiggly,tension=0.3,label=\small $Z$,tension=0.3,label.side=right", __v7, __v6);
vconnect ("wiggly,tension=0.3,label=\small $Z$,tension=0.3,label.side=right", __v6, __v8);
vconnect ("fermion,tension=0.2", __o2, __v7, __o3);
vconnect ("fermion,tension=0.2", __o4, __v8, __o5);
vvertex ("decoration.shape=circle,foreground=(0.8,,0.,,0.),decoration.size=5", __v5, __v6);
vfreeze;
vdraw;
endsubgraph;
endchar;
% the end.
end.
endinput;
