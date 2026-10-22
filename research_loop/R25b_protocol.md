# R25b: separate penalties while preserving the established semantic predictor

Attribution completed: addedmetadata component−0.000367508, changedsemanticregularization+0.000579367, net+0.000211859. Metadata contribution itself is heterogeneous (thirdfoldworse,CIcrosses0), so do not claim it uniformlyhelps. One controlled separate-penalty test is justified to avoid changing semantic predictions on inertmetadata.

Freeze eachfold's R7semanticlambda4/4/1. Represent support/query as [semantic/sqrt(lambda_semantic), metadata/sqrt(lambda_metadata)] and solve ridge1. Select ONLY metadata lambda from off/16/4/1/.25 using sourceepisodes73/max25, allselectionsbeforeouter0. Off and constant/empty metadata must exactly reproduce R7 at allbudgets. This differs from the rejected fixedprior-mix sweeps: it tests an additional visiblefield block and leaves the oldsemanticpenalty fixed.

Compare R7, R24uniformpenalty, smoothed2 and training-selectedsmoothed, with metadataassociationpermutation. If no stableincrement, stop this metadata-kernel expansion; do not choose fieldweights/keys using outerlabels. Original0.005gate unchanged. If an increment appears, fixed3seedverification is required before any best-candidate claim.
