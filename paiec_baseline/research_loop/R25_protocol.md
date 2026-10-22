# R25a: audit a coupled regularization confound before claiming metadata failure

Registered after R24. Combined0.207392 loses oldsemantic0.207180, butmetadatapermutation0.208174 shows real correspondence signal. R24joint ridge selected16/16/4; frozensemantic selected4/4/1. Therefore semanticcoordinates shrink fourtimes more even where metadata contributeszero centeredinformation. The blanket statement 'metadata has no value' is not justified.

First recompute semantic-only predictions at R24's ridge, sameouterdata andseed0. Decompose combined−oldsemantic into combined−same-ridge-semantic (addedfield component) plus same-ridge-semantic−oldsemantic (regularization component). Record exact budgets,benchmarks and inertmetadata cases. No new hyperparameter chosen from these outerresults. This is a posthoc attributiondiagnostic on reusedpublicdata.

If the addedfield component is beneficial but canceled by changedsemanticregularization, justify one source-selected separatepenalty test preservingR7semanticlambda exactly and allowingmetadataoff. If noaddedfieldbenefit, stopmetadataexpansion. Alloriginalgates and evidence boundaries stayfixed.
