# R74 来源 pilot 完整数值表

所有数字均为 source pool 的 Benchmark-grouped OOF；三个 pool 重叠，不能当作三份独立样本。准确率是 response outcome 在 p≥0.5 下的预测正确率，不是 subject 原始答题正确率。

## 教师信息诊断：B0、四维完整注释 query 子集

| 方法 | B0 Brier ↓ | Benchmark-equal ↓ | B0准确率 |
|---|---:|---:|---:|
| base | 0.249250842 | 0.242643844 | 56.04% |
| teacher | 0.337674293 | 0.306078671 | 53.54% |
| teacher_permuted | 0.310736254 | 0.277456752 | 53.76% |
| teacher_only | 0.319300763 | 0.300677241 | 53.51% |
| student_subset | 0.248595761 | 0.242875140 | 56.12% |
| constant_0.5 | 0.250000000 | 0.250000000 | 54.97% |
| source_constant | 0.252452534 | 0.241100000 | 54.97% |
| R56_B0 | 0.251434052 | 0.246620828 | 56.16% |
| R66_item_B0 | 0.244381525 | 0.244756628 | 59.09% |

## CPU 学生部署：原完整来源 episode 六预算

| 方法 | 六预算 ALC ↓ | Benchmark-equal ↓ | B0 / B1 / B31准确率 |
|---|---:|---:|---:|
| R25 | 0.211347421 | 0.212611719 | 50.67% / 62.99% / 71.40% |
| R56_segment | 0.209950904 | 0.211802329 | 56.98% / 62.96% / 71.40% |
| R66_item_segment | 0.209609263 | 0.211603337 | 56.97% / 63.15% / 71.40% |
| constant_0.5 | 0.250000000 | 0.250000000 | 50.67% / 50.67% / 50.67% |
| smoothed_mean_2 | 0.212715179 | 0.215235638 | 50.67% / 62.99% / 70.58% |
| training_selected_smoothed | 0.212621518 | 0.215074626 | 50.67% / 62.99% / 70.58% |
| source_constant_segment | 0.211747749 | 0.213142314 | 50.28% / 62.99% / 71.40% |
| prior_base | 0.210674592 | 0.212265723 | 54.76% / 63.09% / 71.40% |
| prior_student | 0.210595327 | 0.212177911 | 54.44% / 63.28% / 71.40% |
| prior_permuted_student | 0.210732825 | 0.212336537 | 54.48% / 63.10% / 71.40% |
| prior_demand_only | 0.211492894 | 0.213139834 | 53.64% / 62.90% / 71.40% |
| local_student | 0.211345789 | 0.212608280 | 50.67% / 62.99% / 71.41% |
| local_permuted_student | 0.211346335 | 0.212608553 | 50.67% / 62.99% / 71.40% |
| local_demand_only | 0.212701057 | 0.215217266 | 50.67% / 62.99% / 70.58% |

## 六预算 Brier 与准确率

| 方法 | B0 | B1 | B3 | B7 | B15 | B31 |
|---|---:|---:|---:|---:|---:|---:|
| R25 Brier | 0.250000000 | 0.234472615 | 0.214964823 | 0.199373207 | 0.190696447 | 0.184460024 |
| R56_segment Brier | 0.243241055 | 0.230869501 | 0.214964823 | 0.199373207 | 0.190696447 | 0.184460024 |
| R66_item_segment Brier | 0.241544094 | 0.230009780 | 0.214964823 | 0.199373207 | 0.190696447 | 0.184460024 |
| constant_0.5 Brier | 0.250000000 | 0.250000000 | 0.250000000 | 0.250000000 | 0.250000000 | 0.250000000 |
| smoothed_mean_2 Brier | 0.250000000 | 0.234472615 | 0.215513087 | 0.201136403 | 0.193154411 | 0.188598754 |
| training_selected_smoothed Brier | 0.250000000 | 0.234060080 | 0.215240067 | 0.201268069 | 0.193249304 | 0.188580138 |
| source_constant_segment Brier | 0.252376906 | 0.235285804 | 0.214964823 | 0.199373207 | 0.190696447 | 0.184460024 |
| prior_base Brier | 0.247246903 | 0.232485019 | 0.214964823 | 0.199373207 | 0.190696447 | 0.184460024 |
| prior_student Brier | 0.246877095 | 0.232273600 | 0.214964823 | 0.199373207 | 0.190696447 | 0.184460024 |
| prior_permuted_student Brier | 0.247532598 | 0.232633338 | 0.214964823 | 0.199373207 | 0.190696447 | 0.184460024 |
| prior_demand_only Brier | 0.251772378 | 0.234313792 | 0.214964823 | 0.199373207 | 0.190696447 | 0.184460024 |
| local_student Brier | 0.250000000 | 0.234472615 | 0.214964987 | 0.199370382 | 0.190694374 | 0.184453171 |
| local_permuted_student Brier | 0.250000000 | 0.234472615 | 0.214964556 | 0.199370965 | 0.190695868 | 0.184455345 |
| local_demand_only Brier | 0.250000000 | 0.234472615 | 0.215509311 | 0.201120295 | 0.193130233 | 0.188545659 |
| R25 accuracy | 50.666% | 62.992% | 66.417% | 69.012% | 70.243% | 71.404% |
| R56_segment accuracy | 56.981% | 62.959% | 66.417% | 69.012% | 70.243% | 71.404% |
| R66_item_segment accuracy | 56.969% | 63.146% | 66.417% | 69.012% | 70.243% | 71.404% |
| constant_0.5 accuracy | 50.666% | 50.666% | 50.666% | 50.666% | 50.666% | 50.666% |
| smoothed_mean_2 accuracy | 50.666% | 62.992% | 66.307% | 68.612% | 69.702% | 70.581% |
| training_selected_smoothed accuracy | 50.666% | 62.992% | 66.307% | 68.612% | 69.702% | 70.581% |
| source_constant_segment accuracy | 50.277% | 62.992% | 66.417% | 69.012% | 70.243% | 71.404% |
| prior_base accuracy | 54.759% | 63.087% | 66.417% | 69.012% | 70.243% | 71.404% |
| prior_student accuracy | 54.442% | 63.281% | 66.417% | 69.012% | 70.243% | 71.404% |
| prior_permuted_student accuracy | 54.481% | 63.096% | 66.417% | 69.012% | 70.243% | 71.404% |
| prior_demand_only accuracy | 53.643% | 62.902% | 66.417% | 69.012% | 70.243% | 71.404% |
| local_student accuracy | 50.666% | 62.992% | 66.417% | 69.000% | 70.236% | 71.411% |
| local_permuted_student accuracy | 50.666% | 62.992% | 66.415% | 68.999% | 70.231% | 71.397% |
| local_demand_only accuracy | 50.666% | 62.992% | 66.307% | 68.612% | 69.702% | 70.581% |

## 关键配对差值（负数为改善）

| 范围 / 方法 − 对照 | Δ | BM-equal Δ | 三个pool Δ | 95% BM-cluster CI | 非alignment Δ |
|---|---:|---:|---|---|---:|
| teacher: teacher − base | +0.088423451 | +0.063434827 | +0.08259084, +0.05641151, +0.14405152 | [+0.03778892, +0.12847943] | +0.093929945 |
| teacher: teacher − teacher_permuted | +0.026938039 | +0.028621919 | +0.01262589, +0.05177650, +0.02739977 | [-0.01402332, +0.05931074] | +0.027155802 |
| teacher: student_subset − base | -0.000655081 | +0.000231295 | -0.00150139, -0.00058665, +0.00121475 | [-0.00547458, +0.00440906] | -0.000095023 |
| deployable: prior_student − R25 | -0.000752093 | -0.000433808 | -0.00129000, -0.00001069, -0.00084397 | [-0.00284794, +0.00108709] | -0.000840526 |
| deployable: prior_student − prior_base | -0.000079265 | -0.000087812 | -0.00005560, +0.00020454, -0.00040695 | [-0.00029524, +0.00013088] | -0.000098666 |
| deployable: prior_student − prior_permuted_student | -0.000137498 | -0.000158626 | -0.00023602, +0.00020250, -0.00036841 | [-0.00028394, +0.00001849] | -0.000134414 |
| deployable: prior_student − R56_segment | +0.000644424 | +0.000375582 | -0.00141795, +0.00253945, +0.00128556 | [-0.00171920, +0.00352022] | +0.000656039 |
| deployable: prior_student − R66_item_segment | +0.000986064 | +0.000574575 | -0.00150897, +0.00306800, +0.00198254 | [-0.00188027, +0.00488794] | +0.001125217 |
| deployable: local_student − R25 | -0.000001632 | -0.000003439 | -0.00000118, -0.00000308, -0.00000069 | [-0.00000525, +0.00000121] | -0.000001848 |
| deployable: local_student − local_permuted_student | -0.000000547 | -0.000000272 | -0.00000065, +0.00000027, -0.00000127 | [-0.00000141, +0.00000033] | -0.000000651 |

## Grade 质量

| Rubric | 模型 | MAE ↓ | RMSE ↓ | rounded exact | rounded error>1 |
|---|---|---:|---:|---:|---:|
| AS | student | 0.993672 | 1.165266 | 0.2202 | 0.1743 |
| AS | permuted_student | 0.992666 | 1.161474 | 0.2202 | 0.1697 |
| AS | fit_mean | 1.001694 | 1.167300 | 0.2156 | 0.2018 |
| MCr | student | 1.053596 | 1.225423 | 0.2593 | 0.2593 |
| MCr | permuted_student | 1.047683 | 1.221830 | 0.2500 | 0.2454 |
| MCr | fit_mean | 1.053330 | 1.242683 | 0.2870 | 0.2870 |
| QLl | student | 0.998156 | 1.167838 | 0.3119 | 0.2156 |
| QLl | permuted_student | 1.009685 | 1.178316 | 0.3119 | 0.2477 |
| QLl | fit_mean | 1.008848 | 1.176109 | 0.2661 | 0.2248 |
| KNf | student | 0.922483 | 1.242776 | 0.3136 | 0.1273 |
| KNf | permuted_student | 0.933348 | 1.250404 | 0.3091 | 0.1273 |
| KNf | fit_mean | 0.937082 | 1.246350 | 0.2955 | 0.1318 |
| ALL | student | 0.991676 | 1.200858 | 0.2764 | 0.1938 |
| ALL | permuted_student | 0.995583 | 1.203586 | 0.2729 | 0.1972 |
| ALL | fit_mean | 0.999972 | 1.208672 | 0.2661 | 0.2110 |

## 全部 Benchmark 分解

| 范围 | Benchmark | exposures | 方法 | score |
|---|---|---:|---|---:|
| teacher | abstract_reason/formatted_tables | 2 | base | 0.235132643 |
| teacher | abstract_reason/formatted_tables | 2 | teacher | 0.273703152 |
| teacher | abstract_reason/formatted_tables | 2 | teacher_permuted | 0.189619425 |
| teacher | abstract_reason/formatted_tables | 2 | teacher_only | 0.281309882 |
| teacher | abstract_reason/formatted_tables | 2 | student_subset | 0.235461972 |
| teacher | abstract_reason/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| teacher | abstract_reason/formatted_tables | 2 | source_constant | 0.244553042 |
| teacher | abstract_reason/formatted_tables | 2 | R56_B0 | 0.239243441 |
| teacher | abstract_reason/formatted_tables | 2 | R66_item_B0 | 0.238546610 |
| teacher | aegis/formatted_tables | 2 | base | 0.147182495 |
| teacher | aegis/formatted_tables | 2 | teacher | 0.138189439 |
| teacher | aegis/formatted_tables | 2 | teacher_permuted | 0.239634596 |
| teacher | aegis/formatted_tables | 2 | teacher_only | 0.122600231 |
| teacher | aegis/formatted_tables | 2 | student_subset | 0.134945211 |
| teacher | aegis/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| teacher | aegis/formatted_tables | 2 | source_constant | 0.200102360 |
| teacher | aegis/formatted_tables | 2 | R56_B0 | 0.265796769 |
| teacher | aegis/formatted_tables | 2 | R66_item_B0 | 0.260682255 |
| teacher | agentic_review_perturbation/formatted_tables | 6 | base | 0.353452500 |
| teacher | agentic_review_perturbation/formatted_tables | 6 | teacher | 0.426880300 |
| teacher | agentic_review_perturbation/formatted_tables | 6 | teacher_permuted | 0.438369382 |
| teacher | agentic_review_perturbation/formatted_tables | 6 | teacher_only | 0.347647723 |
| teacher | agentic_review_perturbation/formatted_tables | 6 | student_subset | 0.375946823 |
| teacher | agentic_review_perturbation/formatted_tables | 6 | constant_0.5 | 0.250000000 |
| teacher | agentic_review_perturbation/formatted_tables | 6 | source_constant | 0.276506944 |
| teacher | agentic_review_perturbation/formatted_tables | 6 | R56_B0 | 0.220605588 |
| teacher | agentic_review_perturbation/formatted_tables | 6 | R66_item_B0 | 0.223245063 |
| teacher | agi_elo/formatted_tables | 2 | base | 0.298812427 |
| teacher | agi_elo/formatted_tables | 2 | teacher | 0.253242508 |
| teacher | agi_elo/formatted_tables | 2 | teacher_permuted | 0.282137618 |
| teacher | agi_elo/formatted_tables | 2 | teacher_only | 0.243590982 |
| teacher | agi_elo/formatted_tables | 2 | student_subset | 0.302100294 |
| teacher | agi_elo/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| teacher | agi_elo/formatted_tables | 2 | source_constant | 0.269467439 |
| teacher | agi_elo/formatted_tables | 2 | R56_B0 | 0.261845544 |
| teacher | agi_elo/formatted_tables | 2 | R66_item_B0 | 0.267944666 |
| teacher | algotune/formatted_tables | 34 | base | 0.290754173 |
| teacher | algotune/formatted_tables | 34 | teacher | 0.468409478 |
| teacher | algotune/formatted_tables | 34 | teacher_permuted | 0.396141217 |
| teacher | algotune/formatted_tables | 34 | teacher_only | 0.355418940 |
| teacher | algotune/formatted_tables | 34 | student_subset | 0.288312236 |
| teacher | algotune/formatted_tables | 34 | constant_0.5 | 0.250000000 |
| teacher | algotune/formatted_tables | 34 | source_constant | 0.244958335 |
| teacher | algotune/formatted_tables | 34 | R56_B0 | 0.245783117 |
| teacher | algotune/formatted_tables | 34 | R66_item_B0 | 0.258520193 |
| teacher | alignment_faking/formatted_tables | 10 | base | 0.162305777 |
| teacher | alignment_faking/formatted_tables | 10 | teacher | 0.159321440 |
| teacher | alignment_faking/formatted_tables | 10 | teacher_permuted | 0.135998260 |
| teacher | alignment_faking/formatted_tables | 10 | teacher_only | 0.201585233 |
| teacher | alignment_faking/formatted_tables | 10 | student_subset | 0.152353726 |
| teacher | alignment_faking/formatted_tables | 10 | constant_0.5 | 0.250000000 |
| teacher | alignment_faking/formatted_tables | 10 | source_constant | 0.231040278 |
| teacher | alignment_faking/formatted_tables | 10 | R56_B0 | 0.247649814 |
| teacher | alignment_faking/formatted_tables | 10 | R66_item_B0 | 0.232794624 |
| teacher | appworld/formatted_tables | 16 | base | 0.241354745 |
| teacher | appworld/formatted_tables | 16 | teacher | 0.309229413 |
| teacher | appworld/formatted_tables | 16 | teacher_permuted | 0.415843249 |
| teacher | appworld/formatted_tables | 16 | teacher_only | 0.346391930 |
| teacher | appworld/formatted_tables | 16 | student_subset | 0.241461970 |
| teacher | appworld/formatted_tables | 16 | constant_0.5 | 0.250000000 |
| teacher | appworld/formatted_tables | 16 | source_constant | 0.258414860 |
| teacher | appworld/formatted_tables | 16 | R56_B0 | 0.250041599 |
| teacher | appworld/formatted_tables | 16 | R66_item_B0 | 0.250199388 |
| teacher | bbq/formatted_tables | 2 | base | 0.170865834 |
| teacher | bbq/formatted_tables | 2 | teacher | 0.188781992 |
| teacher | bbq/formatted_tables | 2 | teacher_permuted | 0.187969204 |
| teacher | bbq/formatted_tables | 2 | teacher_only | 0.209623677 |
| teacher | bbq/formatted_tables | 2 | student_subset | 0.165231352 |
| teacher | bbq/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| teacher | bbq/formatted_tables | 2 | source_constant | 0.214074691 |
| teacher | bbq/formatted_tables | 2 | R56_B0 | 0.211485632 |
| teacher | bbq/formatted_tables | 2 | R66_item_B0 | 0.218414263 |
| teacher | beavertails/formatted_tables | 4 | base | 0.249816616 |
| teacher | beavertails/formatted_tables | 4 | teacher | 0.248311891 |
| teacher | beavertails/formatted_tables | 4 | teacher_permuted | 0.277310729 |
| teacher | beavertails/formatted_tables | 4 | teacher_only | 0.260148286 |
| teacher | beavertails/formatted_tables | 4 | student_subset | 0.235065124 |
| teacher | beavertails/formatted_tables | 4 | constant_0.5 | 0.250000000 |
| teacher | beavertails/formatted_tables | 4 | source_constant | 0.244618056 |
| teacher | beavertails/formatted_tables | 4 | R56_B0 | 0.232610672 |
| teacher | beavertails/formatted_tables | 4 | R66_item_B0 | 0.219728998 |
| teacher | benger/formatted_tables | 8 | base | 0.233098090 |
| teacher | benger/formatted_tables | 8 | teacher | 0.244509641 |
| teacher | benger/formatted_tables | 8 | teacher_permuted | 0.257156527 |
| teacher | benger/formatted_tables | 8 | teacher_only | 0.310288744 |
| teacher | benger/formatted_tables | 8 | student_subset | 0.226385797 |
| teacher | benger/formatted_tables | 8 | constant_0.5 | 0.250000000 |
| teacher | benger/formatted_tables | 8 | source_constant | 0.236465278 |
| teacher | benger/formatted_tables | 8 | R56_B0 | 0.224172022 |
| teacher | benger/formatted_tables | 8 | R66_item_B0 | 0.219527846 |
| teacher | cellverse/formatted_tables | 2 | base | 0.263776631 |
| teacher | cellverse/formatted_tables | 2 | teacher | 0.338228631 |
| teacher | cellverse/formatted_tables | 2 | teacher_permuted | 0.253332540 |
| teacher | cellverse/formatted_tables | 2 | teacher_only | 0.313421045 |
| teacher | cellverse/formatted_tables | 2 | student_subset | 0.266749047 |
| teacher | cellverse/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| teacher | cellverse/formatted_tables | 2 | source_constant | 0.255840278 |
| teacher | cellverse/formatted_tables | 2 | R56_B0 | 0.262850272 |
| teacher | cellverse/formatted_tables | 2 | R66_item_B0 | 0.275455274 |
| teacher | cmmlu/formatted_tables | 2 | base | 0.229474352 |
| teacher | cmmlu/formatted_tables | 2 | teacher | 0.250670278 |
| teacher | cmmlu/formatted_tables | 2 | teacher_permuted | 0.283944020 |
| teacher | cmmlu/formatted_tables | 2 | teacher_only | 0.169197678 |
| teacher | cmmlu/formatted_tables | 2 | student_subset | 0.233903833 |
| teacher | cmmlu/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| teacher | cmmlu/formatted_tables | 2 | source_constant | 0.148947884 |
| teacher | cmmlu/formatted_tables | 2 | R56_B0 | 0.229373623 |
| teacher | cmmlu/formatted_tables | 2 | R66_item_B0 | 0.214204918 |
| teacher | cruxeval/formatted_tables | 2 | base | 0.274264096 |
| teacher | cruxeval/formatted_tables | 2 | teacher | 0.294283329 |
| teacher | cruxeval/formatted_tables | 2 | teacher_permuted | 0.264557711 |
| teacher | cruxeval/formatted_tables | 2 | teacher_only | 0.297400206 |
| teacher | cruxeval/formatted_tables | 2 | student_subset | 0.273880304 |
| teacher | cruxeval/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| teacher | cruxeval/formatted_tables | 2 | source_constant | 0.259851527 |
| teacher | cruxeval/formatted_tables | 2 | R56_B0 | 0.254056258 |
| teacher | cruxeval/formatted_tables | 2 | R66_item_B0 | 0.288041573 |
| teacher | dave/formatted_tables | 4 | base | 0.258092755 |
| teacher | dave/formatted_tables | 4 | teacher | 0.292363109 |
| teacher | dave/formatted_tables | 4 | teacher_permuted | 0.311479325 |
| teacher | dave/formatted_tables | 4 | teacher_only | 0.303634664 |
| teacher | dave/formatted_tables | 4 | student_subset | 0.258390542 |
| teacher | dave/formatted_tables | 4 | constant_0.5 | 0.250000000 |
| teacher | dave/formatted_tables | 4 | source_constant | 0.241632584 |
| teacher | dave/formatted_tables | 4 | R56_B0 | 0.246110034 |
| teacher | dave/formatted_tables | 4 | R66_item_B0 | 0.245927587 |
| teacher | editbench/formatted_tables | 24 | base | 0.243146408 |
| teacher | editbench/formatted_tables | 24 | teacher | 0.401290573 |
| teacher | editbench/formatted_tables | 24 | teacher_permuted | 0.362850508 |
| teacher | editbench/formatted_tables | 24 | teacher_only | 0.410110407 |
| teacher | editbench/formatted_tables | 24 | student_subset | 0.245755946 |
| teacher | editbench/formatted_tables | 24 | constant_0.5 | 0.250000000 |
| teacher | editbench/formatted_tables | 24 | source_constant | 0.289728969 |
| teacher | editbench/formatted_tables | 24 | R56_B0 | 0.246417493 |
| teacher | editbench/formatted_tables | 24 | R66_item_B0 | 0.247212354 |
| teacher | egoschema/formatted_tables | 2 | base | 0.222843570 |
| teacher | egoschema/formatted_tables | 2 | teacher | 0.330833568 |
| teacher | egoschema/formatted_tables | 2 | teacher_permuted | 0.284088118 |
| teacher | egoschema/formatted_tables | 2 | teacher_only | 0.415921328 |
| teacher | egoschema/formatted_tables | 2 | student_subset | 0.221220053 |
| teacher | egoschema/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| teacher | egoschema/formatted_tables | 2 | source_constant | 0.271483233 |
| teacher | egoschema/formatted_tables | 2 | R56_B0 | 0.265424091 |
| teacher | egoschema/formatted_tables | 2 | R66_item_B0 | 0.319210522 |
| teacher | emoji_attack/formatted_tables | 4 | base | 0.409831563 |
| teacher | emoji_attack/formatted_tables | 4 | teacher | 0.733986989 |
| teacher | emoji_attack/formatted_tables | 4 | teacher_permuted | 0.377210290 |
| teacher | emoji_attack/formatted_tables | 4 | teacher_only | 0.718706326 |
| teacher | emoji_attack/formatted_tables | 4 | student_subset | 0.468984931 |
| teacher | emoji_attack/formatted_tables | 4 | constant_0.5 | 0.250000000 |
| teacher | emoji_attack/formatted_tables | 4 | source_constant | 0.264451389 |
| teacher | emoji_attack/formatted_tables | 4 | R56_B0 | 0.287572335 |
| teacher | emoji_attack/formatted_tables | 4 | R66_item_B0 | 0.228976346 |
| teacher | enginemt_qa/formatted_tables | 2 | base | 0.169820140 |
| teacher | enginemt_qa/formatted_tables | 2 | teacher | 0.481525949 |
| teacher | enginemt_qa/formatted_tables | 2 | teacher_permuted | 0.221725925 |
| teacher | enginemt_qa/formatted_tables | 2 | teacher_only | 0.477701904 |
| teacher | enginemt_qa/formatted_tables | 2 | student_subset | 0.162341462 |
| teacher | enginemt_qa/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| teacher | enginemt_qa/formatted_tables | 2 | source_constant | 0.200102360 |
| teacher | enginemt_qa/formatted_tables | 2 | R56_B0 | 0.259080130 |
| teacher | enginemt_qa/formatted_tables | 2 | R66_item_B0 | 0.271718733 |
| teacher | evouna/formatted_tables | 2 | base | 0.262569449 |
| teacher | evouna/formatted_tables | 2 | teacher | 0.286859837 |
| teacher | evouna/formatted_tables | 2 | teacher_permuted | 0.264534235 |
| teacher | evouna/formatted_tables | 2 | teacher_only | 0.252858673 |
| teacher | evouna/formatted_tables | 2 | student_subset | 0.264299737 |
| teacher | evouna/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| teacher | evouna/formatted_tables | 2 | source_constant | 0.241632584 |
| teacher | evouna/formatted_tables | 2 | R56_B0 | 0.237788945 |
| teacher | evouna/formatted_tables | 2 | R66_item_B0 | 0.243812248 |
| teacher | faithcot/formatted_tables | 4 | base | 0.168119709 |
| teacher | faithcot/formatted_tables | 4 | teacher | 0.156681871 |
| teacher | faithcot/formatted_tables | 4 | teacher_permuted | 0.201926283 |
| teacher | faithcot/formatted_tables | 4 | teacher_only | 0.170757024 |
| teacher | faithcot/formatted_tables | 4 | student_subset | 0.170707753 |
| teacher | faithcot/formatted_tables | 4 | constant_0.5 | 0.250000000 |
| teacher | faithcot/formatted_tables | 4 | source_constant | 0.171777249 |
| teacher | faithcot/formatted_tables | 4 | R56_B0 | 0.208499069 |
| teacher | faithcot/formatted_tables | 4 | R66_item_B0 | 0.241615329 |
| teacher | fewshot_ttt_bbh/formatted_tables | 6 | base | 0.294935717 |
| teacher | fewshot_ttt_bbh/formatted_tables | 6 | teacher | 0.329593268 |
| teacher | fewshot_ttt_bbh/formatted_tables | 6 | teacher_permuted | 0.374525296 |
| teacher | fewshot_ttt_bbh/formatted_tables | 6 | teacher_only | 0.282345838 |
| teacher | fewshot_ttt_bbh/formatted_tables | 6 | student_subset | 0.299381474 |
| teacher | fewshot_ttt_bbh/formatted_tables | 6 | constant_0.5 | 0.250000000 |
| teacher | fewshot_ttt_bbh/formatted_tables | 6 | source_constant | 0.262593210 |
| teacher | fewshot_ttt_bbh/formatted_tables | 6 | R56_B0 | 0.236303628 |
| teacher | fewshot_ttt_bbh/formatted_tables | 6 | R66_item_B0 | 0.223048493 |
| teacher | financebench/formatted_tables | 26 | base | 0.246460350 |
| teacher | financebench/formatted_tables | 26 | teacher | 0.284244196 |
| teacher | financebench/formatted_tables | 26 | teacher_permuted | 0.251006945 |
| teacher | financebench/formatted_tables | 26 | teacher_only | 0.269045644 |
| teacher | financebench/formatted_tables | 26 | student_subset | 0.245734720 |
| teacher | financebench/formatted_tables | 26 | constant_0.5 | 0.250000000 |
| teacher | financebench/formatted_tables | 26 | source_constant | 0.264298025 |
| teacher | financebench/formatted_tables | 26 | R56_B0 | 0.298167874 |
| teacher | financebench/formatted_tables | 26 | R66_item_B0 | 0.258208476 |
| teacher | finegrain_t2i/formatted_tables | 2 | base | 0.262081335 |
| teacher | finegrain_t2i/formatted_tables | 2 | teacher | 0.264727321 |
| teacher | finegrain_t2i/formatted_tables | 2 | teacher_permuted | 0.279584477 |
| teacher | finegrain_t2i/formatted_tables | 2 | teacher_only | 0.263026072 |
| teacher | finegrain_t2i/formatted_tables | 2 | student_subset | 0.253827473 |
| teacher | finegrain_t2i/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| teacher | finegrain_t2i/formatted_tables | 2 | source_constant | 0.257741358 |
| teacher | finegrain_t2i/formatted_tables | 2 | R56_B0 | 0.259974312 |
| teacher | finegrain_t2i/formatted_tables | 2 | R66_item_B0 | 0.258036228 |
| teacher | hallusionbench/formatted_tables | 8 | base | 0.135260888 |
| teacher | hallusionbench/formatted_tables | 8 | teacher | 0.190019933 |
| teacher | hallusionbench/formatted_tables | 8 | teacher_permuted | 0.108016173 |
| teacher | hallusionbench/formatted_tables | 8 | teacher_only | 0.193521341 |
| teacher | hallusionbench/formatted_tables | 8 | student_subset | 0.106561576 |
| teacher | hallusionbench/formatted_tables | 8 | constant_0.5 | 0.250000000 |
| teacher | hallusionbench/formatted_tables | 8 | source_constant | 0.236118056 |
| teacher | hallusionbench/formatted_tables | 8 | R56_B0 | 0.228047619 |
| teacher | hallusionbench/formatted_tables | 8 | R66_item_B0 | 0.169087090 |
| deployable | abstract_reason/formatted_tables | 2 | R25 | 0.264062782 |
| deployable | abstract_reason/formatted_tables | 2 | R56_segment | 0.275679769 |
| deployable | abstract_reason/formatted_tables | 2 | R66_item_segment | 0.277276264 |
| deployable | abstract_reason/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| deployable | abstract_reason/formatted_tables | 2 | smoothed_mean_2 | 0.265242126 |
| deployable | abstract_reason/formatted_tables | 2 | training_selected_smoothed | 0.265242126 |
| deployable | abstract_reason/formatted_tables | 2 | source_constant_segment | 0.265724581 |
| deployable | abstract_reason/formatted_tables | 2 | prior_base | 0.266513532 |
| deployable | abstract_reason/formatted_tables | 2 | prior_student | 0.267317575 |
| deployable | abstract_reason/formatted_tables | 2 | prior_permuted_student | 0.267803267 |
| deployable | abstract_reason/formatted_tables | 2 | prior_demand_only | 0.266007278 |
| deployable | abstract_reason/formatted_tables | 2 | local_student | 0.264057413 |
| deployable | abstract_reason/formatted_tables | 2 | local_permuted_student | 0.264061439 |
| deployable | abstract_reason/formatted_tables | 2 | local_demand_only | 0.265236613 |
| deployable | adaptivestep/formatted_tables | 4 | R25 | 0.229305814 |
| deployable | adaptivestep/formatted_tables | 4 | R56_segment | 0.228821527 |
| deployable | adaptivestep/formatted_tables | 4 | R66_item_segment | 0.229580787 |
| deployable | adaptivestep/formatted_tables | 4 | constant_0.5 | 0.250000000 |
| deployable | adaptivestep/formatted_tables | 4 | smoothed_mean_2 | 0.241670526 |
| deployable | adaptivestep/formatted_tables | 4 | training_selected_smoothed | 0.241415635 |
| deployable | adaptivestep/formatted_tables | 4 | source_constant_segment | 0.231667471 |
| deployable | adaptivestep/formatted_tables | 4 | prior_base | 0.231552145 |
| deployable | adaptivestep/formatted_tables | 4 | prior_student | 0.231669093 |
| deployable | adaptivestep/formatted_tables | 4 | prior_permuted_student | 0.231739558 |
| deployable | adaptivestep/formatted_tables | 4 | prior_demand_only | 0.235843906 |
| deployable | adaptivestep/formatted_tables | 4 | local_student | 0.229285096 |
| deployable | adaptivestep/formatted_tables | 4 | local_permuted_student | 0.229250147 |
| deployable | adaptivestep/formatted_tables | 4 | local_demand_only | 0.241608715 |
| deployable | aegis/formatted_tables | 2 | R25 | 0.259244726 |
| deployable | aegis/formatted_tables | 2 | R56_segment | 0.259235666 |
| deployable | aegis/formatted_tables | 2 | R66_item_segment | 0.259295884 |
| deployable | aegis/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| deployable | aegis/formatted_tables | 2 | smoothed_mean_2 | 0.255075590 |
| deployable | aegis/formatted_tables | 2 | training_selected_smoothed | 0.251090647 |
| deployable | aegis/formatted_tables | 2 | source_constant_segment | 0.259390524 |
| deployable | aegis/formatted_tables | 2 | prior_base | 0.257768696 |
| deployable | aegis/formatted_tables | 2 | prior_student | 0.257427328 |
| deployable | aegis/formatted_tables | 2 | prior_permuted_student | 0.257955617 |
| deployable | aegis/formatted_tables | 2 | prior_demand_only | 0.257685960 |
| deployable | aegis/formatted_tables | 2 | local_student | 0.259227735 |
| deployable | aegis/formatted_tables | 2 | local_permuted_student | 0.259243468 |
| deployable | aegis/formatted_tables | 2 | local_demand_only | 0.255032190 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | R25 | 0.136871096 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | R56_segment | 0.137697562 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | R66_item_segment | 0.141611069 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | constant_0.5 | 0.250000000 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | smoothed_mean_2 | 0.141226430 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | training_selected_smoothed | 0.141226430 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | source_constant_segment | 0.142999958 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | prior_base | 0.141157566 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | prior_student | 0.140913807 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | prior_permuted_student | 0.140863847 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | prior_demand_only | 0.142676263 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | local_student | 0.136872188 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | local_permuted_student | 0.136874658 |
| deployable | agentic_review_perturbation/formatted_tables | 6 | local_demand_only | 0.141219870 |
| deployable | agi_elo/formatted_tables | 40 | R25 | 0.198882406 |
| deployable | agi_elo/formatted_tables | 40 | R56_segment | 0.195085648 |
| deployable | agi_elo/formatted_tables | 40 | R66_item_segment | 0.195867691 |
| deployable | agi_elo/formatted_tables | 40 | constant_0.5 | 0.250000000 |
| deployable | agi_elo/formatted_tables | 40 | smoothed_mean_2 | 0.199236559 |
| deployable | agi_elo/formatted_tables | 40 | training_selected_smoothed | 0.199236559 |
| deployable | agi_elo/formatted_tables | 40 | source_constant_segment | 0.197723098 |
| deployable | agi_elo/formatted_tables | 40 | prior_base | 0.194298683 |
| deployable | agi_elo/formatted_tables | 40 | prior_student | 0.193637126 |
| deployable | agi_elo/formatted_tables | 40 | prior_permuted_student | 0.193562964 |
| deployable | agi_elo/formatted_tables | 40 | prior_demand_only | 0.195190685 |
| deployable | agi_elo/formatted_tables | 40 | local_student | 0.198883642 |
| deployable | agi_elo/formatted_tables | 40 | local_permuted_student | 0.198882926 |
| deployable | agi_elo/formatted_tables | 40 | local_demand_only | 0.199232054 |
| deployable | ai2d_test/formatted_tables | 50 | R25 | 0.221216354 |
| deployable | ai2d_test/formatted_tables | 50 | R56_segment | 0.223802340 |
| deployable | ai2d_test/formatted_tables | 50 | R66_item_segment | 0.221452650 |
| deployable | ai2d_test/formatted_tables | 50 | constant_0.5 | 0.250000000 |
| deployable | ai2d_test/formatted_tables | 50 | smoothed_mean_2 | 0.222781541 |
| deployable | ai2d_test/formatted_tables | 50 | training_selected_smoothed | 0.221826079 |
| deployable | ai2d_test/formatted_tables | 50 | source_constant_segment | 0.220761095 |
| deployable | ai2d_test/formatted_tables | 50 | prior_base | 0.222213754 |
| deployable | ai2d_test/formatted_tables | 50 | prior_student | 0.222769888 |
| deployable | ai2d_test/formatted_tables | 50 | prior_permuted_student | 0.222650271 |
| deployable | ai2d_test/formatted_tables | 50 | prior_demand_only | 0.224923340 |
| deployable | ai2d_test/formatted_tables | 50 | local_student | 0.221217802 |
| deployable | ai2d_test/formatted_tables | 50 | local_permuted_student | 0.221218117 |
| deployable | ai2d_test/formatted_tables | 50 | local_demand_only | 0.222771691 |
| deployable | algotune/formatted_tables | 36 | R25 | 0.211792127 |
| deployable | algotune/formatted_tables | 36 | R56_segment | 0.215303559 |
| deployable | algotune/formatted_tables | 36 | R66_item_segment | 0.217351963 |
| deployable | algotune/formatted_tables | 36 | constant_0.5 | 0.250000000 |
| deployable | algotune/formatted_tables | 36 | smoothed_mean_2 | 0.212120430 |
| deployable | algotune/formatted_tables | 36 | training_selected_smoothed | 0.212120430 |
| deployable | algotune/formatted_tables | 36 | source_constant_segment | 0.210709911 |
| deployable | algotune/formatted_tables | 36 | prior_base | 0.213165249 |
| deployable | algotune/formatted_tables | 36 | prior_student | 0.212133379 |
| deployable | algotune/formatted_tables | 36 | prior_permuted_student | 0.212110817 |
| deployable | algotune/formatted_tables | 36 | prior_demand_only | 0.209836407 |
| deployable | algotune/formatted_tables | 36 | local_student | 0.211795523 |
| deployable | algotune/formatted_tables | 36 | local_permuted_student | 0.211797506 |
| deployable | algotune/formatted_tables | 36 | local_demand_only | 0.212121458 |
| deployable | alignment_faking/formatted_tables | 50 | R25 | 0.206604698 |
| deployable | alignment_faking/formatted_tables | 50 | R56_segment | 0.206756911 |
| deployable | alignment_faking/formatted_tables | 50 | R66_item_segment | 0.208389557 |
| deployable | alignment_faking/formatted_tables | 50 | constant_0.5 | 0.250000000 |
| deployable | alignment_faking/formatted_tables | 50 | smoothed_mean_2 | 0.207966070 |
| deployable | alignment_faking/formatted_tables | 50 | training_selected_smoothed | 0.207966070 |
| deployable | alignment_faking/formatted_tables | 50 | source_constant_segment | 0.208565317 |
| deployable | alignment_faking/formatted_tables | 50 | prior_base | 0.207000467 |
| deployable | alignment_faking/formatted_tables | 50 | prior_student | 0.207221536 |
| deployable | alignment_faking/formatted_tables | 50 | prior_permuted_student | 0.207406770 |
| deployable | alignment_faking/formatted_tables | 50 | prior_demand_only | 0.208262973 |
| deployable | alignment_faking/formatted_tables | 50 | local_student | 0.206606413 |
| deployable | alignment_faking/formatted_tables | 50 | local_permuted_student | 0.206605351 |
| deployable | alignment_faking/formatted_tables | 50 | local_demand_only | 0.207966615 |
| deployable | appworld/formatted_tables | 36 | R25 | 0.181537902 |
| deployable | appworld/formatted_tables | 36 | R56_segment | 0.183606426 |
| deployable | appworld/formatted_tables | 36 | R66_item_segment | 0.188875837 |
| deployable | appworld/formatted_tables | 36 | constant_0.5 | 0.250000000 |
| deployable | appworld/formatted_tables | 36 | smoothed_mean_2 | 0.183737310 |
| deployable | appworld/formatted_tables | 36 | training_selected_smoothed | 0.186093926 |
| deployable | appworld/formatted_tables | 36 | source_constant_segment | 0.184240124 |
| deployable | appworld/formatted_tables | 36 | prior_base | 0.179314022 |
| deployable | appworld/formatted_tables | 36 | prior_student | 0.178852693 |
| deployable | appworld/formatted_tables | 36 | prior_permuted_student | 0.179314597 |
| deployable | appworld/formatted_tables | 36 | prior_demand_only | 0.181522717 |
| deployable | appworld/formatted_tables | 36 | local_student | 0.181544375 |
| deployable | appworld/formatted_tables | 36 | local_permuted_student | 0.181542965 |
| deployable | appworld/formatted_tables | 36 | local_demand_only | 0.183738323 |
| deployable | bbq/formatted_tables | 14 | R25 | 0.241532675 |
| deployable | bbq/formatted_tables | 14 | R56_segment | 0.237149320 |
| deployable | bbq/formatted_tables | 14 | R66_item_segment | 0.237898865 |
| deployable | bbq/formatted_tables | 14 | constant_0.5 | 0.250000000 |
| deployable | bbq/formatted_tables | 14 | smoothed_mean_2 | 0.243686977 |
| deployable | bbq/formatted_tables | 14 | training_selected_smoothed | 0.240704605 |
| deployable | bbq/formatted_tables | 14 | source_constant_segment | 0.244068317 |
| deployable | bbq/formatted_tables | 14 | prior_base | 0.246026065 |
| deployable | bbq/formatted_tables | 14 | prior_student | 0.246695241 |
| deployable | bbq/formatted_tables | 14 | prior_permuted_student | 0.246416698 |
| deployable | bbq/formatted_tables | 14 | prior_demand_only | 0.252608003 |
| deployable | bbq/formatted_tables | 14 | local_student | 0.241529364 |
| deployable | bbq/formatted_tables | 14 | local_permuted_student | 0.241534190 |
| deployable | bbq/formatted_tables | 14 | local_demand_only | 0.243675307 |
| deployable | beavertails/formatted_tables | 8 | R25 | 0.186294920 |
| deployable | beavertails/formatted_tables | 8 | R56_segment | 0.182102629 |
| deployable | beavertails/formatted_tables | 8 | R66_item_segment | 0.181968657 |
| deployable | beavertails/formatted_tables | 8 | constant_0.5 | 0.250000000 |
| deployable | beavertails/formatted_tables | 8 | smoothed_mean_2 | 0.194575888 |
| deployable | beavertails/formatted_tables | 8 | training_selected_smoothed | 0.195112764 |
| deployable | beavertails/formatted_tables | 8 | source_constant_segment | 0.183642767 |
| deployable | beavertails/formatted_tables | 8 | prior_base | 0.184136396 |
| deployable | beavertails/formatted_tables | 8 | prior_student | 0.184657041 |
| deployable | beavertails/formatted_tables | 8 | prior_permuted_student | 0.184397088 |
| deployable | beavertails/formatted_tables | 8 | prior_demand_only | 0.184627572 |
| deployable | beavertails/formatted_tables | 8 | local_student | 0.186295314 |
| deployable | beavertails/formatted_tables | 8 | local_permuted_student | 0.186297764 |
| deployable | beavertails/formatted_tables | 8 | local_demand_only | 0.194568902 |
| deployable | benger/formatted_tables | 50 | R25 | 0.166625917 |
| deployable | benger/formatted_tables | 50 | R56_segment | 0.159481355 |
| deployable | benger/formatted_tables | 50 | R66_item_segment | 0.158649217 |
| deployable | benger/formatted_tables | 50 | constant_0.5 | 0.250000000 |
| deployable | benger/formatted_tables | 50 | smoothed_mean_2 | 0.166625917 |
| deployable | benger/formatted_tables | 50 | training_selected_smoothed | 0.166625917 |
| deployable | benger/formatted_tables | 50 | source_constant_segment | 0.162775459 |
| deployable | benger/formatted_tables | 50 | prior_base | 0.153995082 |
| deployable | benger/formatted_tables | 50 | prior_student | 0.152747337 |
| deployable | benger/formatted_tables | 50 | prior_permuted_student | 0.152953774 |
| deployable | benger/formatted_tables | 50 | prior_demand_only | 0.157037497 |
| deployable | benger/formatted_tables | 50 | local_student | 0.166625917 |
| deployable | benger/formatted_tables | 50 | local_permuted_student | 0.166625917 |
| deployable | benger/formatted_tables | 50 | local_demand_only | 0.166625917 |
| deployable | bertaqa/formatted_tables | 12 | R25 | 0.197930744 |
| deployable | bertaqa/formatted_tables | 12 | R56_segment | 0.203211347 |
| deployable | bertaqa/formatted_tables | 12 | R66_item_segment | 0.197388101 |
| deployable | bertaqa/formatted_tables | 12 | constant_0.5 | 0.250000000 |
| deployable | bertaqa/formatted_tables | 12 | smoothed_mean_2 | 0.198710431 |
| deployable | bertaqa/formatted_tables | 12 | training_selected_smoothed | 0.198710431 |
| deployable | bertaqa/formatted_tables | 12 | source_constant_segment | 0.196404849 |
| deployable | bertaqa/formatted_tables | 12 | prior_base | 0.194612911 |
| deployable | bertaqa/formatted_tables | 12 | prior_student | 0.194593555 |
| deployable | bertaqa/formatted_tables | 12 | prior_permuted_student | 0.194945867 |
| deployable | bertaqa/formatted_tables | 12 | prior_demand_only | 0.196080916 |
| deployable | bertaqa/formatted_tables | 12 | local_student | 0.197934597 |
| deployable | bertaqa/formatted_tables | 12 | local_permuted_student | 0.197934790 |
| deployable | bertaqa/formatted_tables | 12 | local_demand_only | 0.198710348 |
| deployable | bird_sql/formatted_tables | 50 | R25 | 0.112117727 |
| deployable | bird_sql/formatted_tables | 50 | R56_segment | 0.095741293 |
| deployable | bird_sql/formatted_tables | 50 | R66_item_segment | 0.088152124 |
| deployable | bird_sql/formatted_tables | 50 | constant_0.5 | 0.250000000 |
| deployable | bird_sql/formatted_tables | 50 | smoothed_mean_2 | 0.114774713 |
| deployable | bird_sql/formatted_tables | 50 | training_selected_smoothed | 0.122846625 |
| deployable | bird_sql/formatted_tables | 50 | source_constant_segment | 0.113762156 |
| deployable | bird_sql/formatted_tables | 50 | prior_base | 0.114176113 |
| deployable | bird_sql/formatted_tables | 50 | prior_student | 0.113940561 |
| deployable | bird_sql/formatted_tables | 50 | prior_permuted_student | 0.114551471 |
| deployable | bird_sql/formatted_tables | 50 | prior_demand_only | 0.113803106 |
| deployable | bird_sql/formatted_tables | 50 | local_student | 0.112116958 |
| deployable | bird_sql/formatted_tables | 50 | local_permuted_student | 0.112118302 |
| deployable | bird_sql/formatted_tables | 50 | local_demand_only | 0.114769338 |
| deployable | braveguard/formatted_tables | 4 | R25 | 0.152171975 |
| deployable | braveguard/formatted_tables | 4 | R56_segment | 0.149308881 |
| deployable | braveguard/formatted_tables | 4 | R66_item_segment | 0.146077321 |
| deployable | braveguard/formatted_tables | 4 | constant_0.5 | 0.250000000 |
| deployable | braveguard/formatted_tables | 4 | smoothed_mean_2 | 0.153171515 |
| deployable | braveguard/formatted_tables | 4 | training_selected_smoothed | 0.158420821 |
| deployable | braveguard/formatted_tables | 4 | source_constant_segment | 0.152728284 |
| deployable | braveguard/formatted_tables | 4 | prior_base | 0.149431638 |
| deployable | braveguard/formatted_tables | 4 | prior_student | 0.148835594 |
| deployable | braveguard/formatted_tables | 4 | prior_permuted_student | 0.149295561 |
| deployable | braveguard/formatted_tables | 4 | prior_demand_only | 0.151326681 |
| deployable | braveguard/formatted_tables | 4 | local_student | 0.152180184 |
| deployable | braveguard/formatted_tables | 4 | local_permuted_student | 0.152176643 |
| deployable | braveguard/formatted_tables | 4 | local_demand_only | 0.153180797 |
| deployable | cellverse/formatted_tables | 2 | R25 | 0.252931478 |
| deployable | cellverse/formatted_tables | 2 | R56_segment | 0.256267104 |
| deployable | cellverse/formatted_tables | 2 | R66_item_segment | 0.258345931 |
| deployable | cellverse/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| deployable | cellverse/formatted_tables | 2 | smoothed_mean_2 | 0.253008172 |
| deployable | cellverse/formatted_tables | 2 | training_selected_smoothed | 0.253008172 |
| deployable | cellverse/formatted_tables | 2 | source_constant_segment | 0.256885838 |
| deployable | cellverse/formatted_tables | 2 | prior_base | 0.257901433 |
| deployable | cellverse/formatted_tables | 2 | prior_student | 0.258141472 |
| deployable | cellverse/formatted_tables | 2 | prior_permuted_student | 0.257577873 |
| deployable | cellverse/formatted_tables | 2 | prior_demand_only | 0.256754378 |
| deployable | cellverse/formatted_tables | 2 | local_student | 0.252931365 |
| deployable | cellverse/formatted_tables | 2 | local_permuted_student | 0.252931169 |
| deployable | cellverse/formatted_tables | 2 | local_demand_only | 0.253008029 |
| deployable | ceval/formatted_tables | 22 | R25 | 0.242858102 |
| deployable | ceval/formatted_tables | 22 | R56_segment | 0.242848600 |
| deployable | ceval/formatted_tables | 22 | R66_item_segment | 0.243261451 |
| deployable | ceval/formatted_tables | 22 | constant_0.5 | 0.250000000 |
| deployable | ceval/formatted_tables | 22 | smoothed_mean_2 | 0.242576834 |
| deployable | ceval/formatted_tables | 22 | training_selected_smoothed | 0.240025296 |
| deployable | ceval/formatted_tables | 22 | source_constant_segment | 0.244223501 |
| deployable | ceval/formatted_tables | 22 | prior_base | 0.244911675 |
| deployable | ceval/formatted_tables | 22 | prior_student | 0.245202864 |
| deployable | ceval/formatted_tables | 22 | prior_permuted_student | 0.245224043 |
| deployable | ceval/formatted_tables | 22 | prior_demand_only | 0.246668127 |
| deployable | ceval/formatted_tables | 22 | local_student | 0.242850741 |
| deployable | ceval/formatted_tables | 22 | local_permuted_student | 0.242852571 |
| deployable | ceval/formatted_tables | 22 | local_demand_only | 0.242557847 |
| deployable | cmmlu/formatted_tables | 44 | R25 | 0.194623258 |
| deployable | cmmlu/formatted_tables | 44 | R56_segment | 0.192740282 |
| deployable | cmmlu/formatted_tables | 44 | R66_item_segment | 0.192856893 |
| deployable | cmmlu/formatted_tables | 44 | constant_0.5 | 0.250000000 |
| deployable | cmmlu/formatted_tables | 44 | smoothed_mean_2 | 0.193933555 |
| deployable | cmmlu/formatted_tables | 44 | training_selected_smoothed | 0.195378139 |
| deployable | cmmlu/formatted_tables | 44 | source_constant_segment | 0.198482795 |
| deployable | cmmlu/formatted_tables | 44 | prior_base | 0.203065608 |
| deployable | cmmlu/formatted_tables | 44 | prior_student | 0.202993742 |
| deployable | cmmlu/formatted_tables | 44 | prior_permuted_student | 0.203725806 |
| deployable | cmmlu/formatted_tables | 44 | prior_demand_only | 0.201553268 |
| deployable | cmmlu/formatted_tables | 44 | local_student | 0.194626769 |
| deployable | cmmlu/formatted_tables | 44 | local_permuted_student | 0.194626950 |
| deployable | cmmlu/formatted_tables | 44 | local_demand_only | 0.193936054 |
| deployable | cruxeval/formatted_tables | 50 | R25 | 0.242008632 |
| deployable | cruxeval/formatted_tables | 50 | R56_segment | 0.242361192 |
| deployable | cruxeval/formatted_tables | 50 | R66_item_segment | 0.244196346 |
| deployable | cruxeval/formatted_tables | 50 | constant_0.5 | 0.250000000 |
| deployable | cruxeval/formatted_tables | 50 | smoothed_mean_2 | 0.240643166 |
| deployable | cruxeval/formatted_tables | 50 | training_selected_smoothed | 0.240643166 |
| deployable | cruxeval/formatted_tables | 50 | source_constant_segment | 0.243781506 |
| deployable | cruxeval/formatted_tables | 50 | prior_base | 0.242643626 |
| deployable | cruxeval/formatted_tables | 50 | prior_student | 0.242491013 |
| deployable | cruxeval/formatted_tables | 50 | prior_permuted_student | 0.243214140 |
| deployable | cruxeval/formatted_tables | 50 | prior_demand_only | 0.241554791 |
| deployable | cruxeval/formatted_tables | 50 | local_student | 0.242012992 |
| deployable | cruxeval/formatted_tables | 50 | local_permuted_student | 0.242014691 |
| deployable | cruxeval/formatted_tables | 50 | local_demand_only | 0.240637656 |
| deployable | dave/formatted_tables | 16 | R25 | 0.231044847 |
| deployable | dave/formatted_tables | 16 | R56_segment | 0.235053855 |
| deployable | dave/formatted_tables | 16 | R66_item_segment | 0.234484314 |
| deployable | dave/formatted_tables | 16 | constant_0.5 | 0.250000000 |
| deployable | dave/formatted_tables | 16 | smoothed_mean_2 | 0.231390001 |
| deployable | dave/formatted_tables | 16 | training_selected_smoothed | 0.231390001 |
| deployable | dave/formatted_tables | 16 | source_constant_segment | 0.231544376 |
| deployable | dave/formatted_tables | 16 | prior_base | 0.233867342 |
| deployable | dave/formatted_tables | 16 | prior_student | 0.233767227 |
| deployable | dave/formatted_tables | 16 | prior_permuted_student | 0.233344213 |
| deployable | dave/formatted_tables | 16 | prior_demand_only | 0.231347589 |
| deployable | dave/formatted_tables | 16 | local_student | 0.231051910 |
| deployable | dave/formatted_tables | 16 | local_permuted_student | 0.231051718 |
| deployable | dave/formatted_tables | 16 | local_demand_only | 0.231396287 |
| deployable | devbench/formatted_tables | 38 | R25 | 0.218042542 |
| deployable | devbench/formatted_tables | 38 | R56_segment | 0.216289533 |
| deployable | devbench/formatted_tables | 38 | R66_item_segment | 0.216407291 |
| deployable | devbench/formatted_tables | 38 | constant_0.5 | 0.250000000 |
| deployable | devbench/formatted_tables | 38 | smoothed_mean_2 | 0.226491205 |
| deployable | devbench/formatted_tables | 38 | training_selected_smoothed | 0.224999514 |
| deployable | devbench/formatted_tables | 38 | source_constant_segment | 0.220545958 |
| deployable | devbench/formatted_tables | 38 | prior_base | 0.216553336 |
| deployable | devbench/formatted_tables | 38 | prior_student | 0.216811142 |
| deployable | devbench/formatted_tables | 38 | prior_permuted_student | 0.216547196 |
| deployable | devbench/formatted_tables | 38 | prior_demand_only | 0.217890368 |
| deployable | devbench/formatted_tables | 38 | local_student | 0.218023227 |
| deployable | devbench/formatted_tables | 38 | local_permuted_student | 0.218025033 |
| deployable | devbench/formatted_tables | 38 | local_demand_only | 0.226422619 |
| deployable | editbench/formatted_tables | 50 | R25 | 0.264414674 |
| deployable | editbench/formatted_tables | 50 | R56_segment | 0.264610064 |
| deployable | editbench/formatted_tables | 50 | R66_item_segment | 0.266322195 |
| deployable | editbench/formatted_tables | 50 | constant_0.5 | 0.250000000 |
| deployable | editbench/formatted_tables | 50 | smoothed_mean_2 | 0.265542175 |
| deployable | editbench/formatted_tables | 50 | training_selected_smoothed | 0.261730113 |
| deployable | editbench/formatted_tables | 50 | source_constant_segment | 0.264454953 |
| deployable | editbench/formatted_tables | 50 | prior_base | 0.263630576 |
| deployable | editbench/formatted_tables | 50 | prior_student | 0.263688882 |
| deployable | editbench/formatted_tables | 50 | prior_permuted_student | 0.263784277 |
| deployable | editbench/formatted_tables | 50 | prior_demand_only | 0.264330656 |
| deployable | editbench/formatted_tables | 50 | local_student | 0.264415099 |
| deployable | editbench/formatted_tables | 50 | local_permuted_student | 0.264415570 |
| deployable | editbench/formatted_tables | 50 | local_demand_only | 0.265527839 |
| deployable | edu_circuit_hw/formatted_tables | 12 | R25 | 0.214382868 |
| deployable | edu_circuit_hw/formatted_tables | 12 | R56_segment | 0.217454148 |
| deployable | edu_circuit_hw/formatted_tables | 12 | R66_item_segment | 0.215227937 |
| deployable | edu_circuit_hw/formatted_tables | 12 | constant_0.5 | 0.250000000 |
| deployable | edu_circuit_hw/formatted_tables | 12 | smoothed_mean_2 | 0.214382868 |
| deployable | edu_circuit_hw/formatted_tables | 12 | training_selected_smoothed | 0.215209949 |
| deployable | edu_circuit_hw/formatted_tables | 12 | source_constant_segment | 0.212848301 |
| deployable | edu_circuit_hw/formatted_tables | 12 | prior_base | 0.213055683 |
| deployable | edu_circuit_hw/formatted_tables | 12 | prior_student | 0.213566280 |
| deployable | edu_circuit_hw/formatted_tables | 12 | prior_permuted_student | 0.212922138 |
| deployable | edu_circuit_hw/formatted_tables | 12 | prior_demand_only | 0.214002667 |
| deployable | edu_circuit_hw/formatted_tables | 12 | local_student | 0.214382868 |
| deployable | edu_circuit_hw/formatted_tables | 12 | local_permuted_student | 0.214382868 |
| deployable | edu_circuit_hw/formatted_tables | 12 | local_demand_only | 0.214382868 |
| deployable | egoschema/formatted_tables | 22 | R25 | 0.255927254 |
| deployable | egoschema/formatted_tables | 22 | R56_segment | 0.255832964 |
| deployable | egoschema/formatted_tables | 22 | R66_item_segment | 0.258554669 |
| deployable | egoschema/formatted_tables | 22 | constant_0.5 | 0.250000000 |
| deployable | egoschema/formatted_tables | 22 | smoothed_mean_2 | 0.255823549 |
| deployable | egoschema/formatted_tables | 22 | training_selected_smoothed | 0.255823549 |
| deployable | egoschema/formatted_tables | 22 | source_constant_segment | 0.255814029 |
| deployable | egoschema/formatted_tables | 22 | prior_base | 0.255790966 |
| deployable | egoschema/formatted_tables | 22 | prior_student | 0.255764565 |
| deployable | egoschema/formatted_tables | 22 | prior_permuted_student | 0.255891854 |
| deployable | egoschema/formatted_tables | 22 | prior_demand_only | 0.255782809 |
| deployable | egoschema/formatted_tables | 22 | local_student | 0.255927586 |
| deployable | egoschema/formatted_tables | 22 | local_permuted_student | 0.255928777 |
| deployable | egoschema/formatted_tables | 22 | local_demand_only | 0.255821808 |
| deployable | emoji_attack/formatted_tables | 20 | R25 | 0.206619658 |
| deployable | emoji_attack/formatted_tables | 20 | R56_segment | 0.210386116 |
| deployable | emoji_attack/formatted_tables | 20 | R66_item_segment | 0.209684894 |
| deployable | emoji_attack/formatted_tables | 20 | constant_0.5 | 0.250000000 |
| deployable | emoji_attack/formatted_tables | 20 | smoothed_mean_2 | 0.212548476 |
| deployable | emoji_attack/formatted_tables | 20 | training_selected_smoothed | 0.212906443 |
| deployable | emoji_attack/formatted_tables | 20 | source_constant_segment | 0.209729312 |
| deployable | emoji_attack/formatted_tables | 20 | prior_base | 0.206083072 |
| deployable | emoji_attack/formatted_tables | 20 | prior_student | 0.207349727 |
| deployable | emoji_attack/formatted_tables | 20 | prior_permuted_student | 0.206157639 |
| deployable | emoji_attack/formatted_tables | 20 | prior_demand_only | 0.213019092 |
| deployable | emoji_attack/formatted_tables | 20 | local_student | 0.206620843 |
| deployable | emoji_attack/formatted_tables | 20 | local_permuted_student | 0.206622652 |
| deployable | emoji_attack/formatted_tables | 20 | local_demand_only | 0.212542373 |
| deployable | engdesign/formatted_tables | 24 | R25 | 0.226275203 |
| deployable | engdesign/formatted_tables | 24 | R56_segment | 0.218411382 |
| deployable | engdesign/formatted_tables | 24 | R66_item_segment | 0.217387073 |
| deployable | engdesign/formatted_tables | 24 | constant_0.5 | 0.250000000 |
| deployable | engdesign/formatted_tables | 24 | smoothed_mean_2 | 0.228335361 |
| deployable | engdesign/formatted_tables | 24 | training_selected_smoothed | 0.226402730 |
| deployable | engdesign/formatted_tables | 24 | source_constant_segment | 0.222076945 |
| deployable | engdesign/formatted_tables | 24 | prior_base | 0.219650053 |
| deployable | engdesign/formatted_tables | 24 | prior_student | 0.219235552 |
| deployable | engdesign/formatted_tables | 24 | prior_permuted_student | 0.219528265 |
| deployable | engdesign/formatted_tables | 24 | prior_demand_only | 0.218591862 |
| deployable | engdesign/formatted_tables | 24 | local_student | 0.226250790 |
| deployable | engdesign/formatted_tables | 24 | local_permuted_student | 0.226253276 |
| deployable | engdesign/formatted_tables | 24 | local_demand_only | 0.228272514 |
| deployable | enginemt_qa/formatted_tables | 2 | R25 | 0.173371629 |
| deployable | enginemt_qa/formatted_tables | 2 | R56_segment | 0.174786242 |
| deployable | enginemt_qa/formatted_tables | 2 | R66_item_segment | 0.177125912 |
| deployable | enginemt_qa/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| deployable | enginemt_qa/formatted_tables | 2 | smoothed_mean_2 | 0.211839786 |
| deployable | enginemt_qa/formatted_tables | 2 | training_selected_smoothed | 0.211335344 |
| deployable | enginemt_qa/formatted_tables | 2 | source_constant_segment | 0.172011950 |
| deployable | enginemt_qa/formatted_tables | 2 | prior_base | 0.173504616 |
| deployable | enginemt_qa/formatted_tables | 2 | prior_student | 0.170896109 |
| deployable | enginemt_qa/formatted_tables | 2 | prior_permuted_student | 0.173337801 |
| deployable | enginemt_qa/formatted_tables | 2 | prior_demand_only | 0.168782834 |
| deployable | enginemt_qa/formatted_tables | 2 | local_student | 0.173359695 |
| deployable | enginemt_qa/formatted_tables | 2 | local_permuted_student | 0.173355434 |
| deployable | enginemt_qa/formatted_tables | 2 | local_demand_only | 0.211762711 |
| deployable | evouna/formatted_tables | 10 | R25 | 0.210120964 |
| deployable | evouna/formatted_tables | 10 | R56_segment | 0.204715102 |
| deployable | evouna/formatted_tables | 10 | R66_item_segment | 0.206161923 |
| deployable | evouna/formatted_tables | 10 | constant_0.5 | 0.250000000 |
| deployable | evouna/formatted_tables | 10 | smoothed_mean_2 | 0.210250951 |
| deployable | evouna/formatted_tables | 10 | training_selected_smoothed | 0.210250951 |
| deployable | evouna/formatted_tables | 10 | source_constant_segment | 0.209217821 |
| deployable | evouna/formatted_tables | 10 | prior_base | 0.207564051 |
| deployable | evouna/formatted_tables | 10 | prior_student | 0.207565148 |
| deployable | evouna/formatted_tables | 10 | prior_permuted_student | 0.207466952 |
| deployable | evouna/formatted_tables | 10 | prior_demand_only | 0.209726755 |
| deployable | evouna/formatted_tables | 10 | local_student | 0.210118242 |
| deployable | evouna/formatted_tables | 10 | local_permuted_student | 0.210116797 |
| deployable | evouna/formatted_tables | 10 | local_demand_only | 0.210244798 |
| deployable | exploitgym/formatted_tables | 10 | R25 | 0.118069240 |
| deployable | exploitgym/formatted_tables | 10 | R56_segment | 0.104929090 |
| deployable | exploitgym/formatted_tables | 10 | R66_item_segment | 0.100485549 |
| deployable | exploitgym/formatted_tables | 10 | constant_0.5 | 0.250000000 |
| deployable | exploitgym/formatted_tables | 10 | smoothed_mean_2 | 0.117218541 |
| deployable | exploitgym/formatted_tables | 10 | training_selected_smoothed | 0.124593799 |
| deployable | exploitgym/formatted_tables | 10 | source_constant_segment | 0.119675333 |
| deployable | exploitgym/formatted_tables | 10 | prior_base | 0.111009996 |
| deployable | exploitgym/formatted_tables | 10 | prior_student | 0.111032621 |
| deployable | exploitgym/formatted_tables | 10 | prior_permuted_student | 0.111414473 |
| deployable | exploitgym/formatted_tables | 10 | prior_demand_only | 0.117089623 |
| deployable | exploitgym/formatted_tables | 10 | local_student | 0.118070317 |
| deployable | exploitgym/formatted_tables | 10 | local_permuted_student | 0.118069948 |
| deployable | exploitgym/formatted_tables | 10 | local_demand_only | 0.117220137 |
| deployable | faithcot/formatted_tables | 8 | R25 | 0.246164289 |
| deployable | faithcot/formatted_tables | 8 | R56_segment | 0.242768454 |
| deployable | faithcot/formatted_tables | 8 | R66_item_segment | 0.245268321 |
| deployable | faithcot/formatted_tables | 8 | constant_0.5 | 0.250000000 |
| deployable | faithcot/formatted_tables | 8 | smoothed_mean_2 | 0.243142625 |
| deployable | faithcot/formatted_tables | 8 | training_selected_smoothed | 0.239911433 |
| deployable | faithcot/formatted_tables | 8 | source_constant_segment | 0.250338730 |
| deployable | faithcot/formatted_tables | 8 | prior_base | 0.252756352 |
| deployable | faithcot/formatted_tables | 8 | prior_student | 0.252620924 |
| deployable | faithcot/formatted_tables | 8 | prior_permuted_student | 0.253542687 |
| deployable | faithcot/formatted_tables | 8 | prior_demand_only | 0.253401085 |
| deployable | faithcot/formatted_tables | 8 | local_student | 0.246176431 |
| deployable | faithcot/formatted_tables | 8 | local_permuted_student | 0.246170074 |
| deployable | faithcot/formatted_tables | 8 | local_demand_only | 0.243143406 |
| deployable | felm/formatted_tables | 2 | R25 | 0.166019589 |
| deployable | felm/formatted_tables | 2 | R56_segment | 0.157705979 |
| deployable | felm/formatted_tables | 2 | R66_item_segment | 0.158627724 |
| deployable | felm/formatted_tables | 2 | constant_0.5 | 0.250000000 |
| deployable | felm/formatted_tables | 2 | smoothed_mean_2 | 0.167098962 |
| deployable | felm/formatted_tables | 2 | training_selected_smoothed | 0.170963437 |
| deployable | felm/formatted_tables | 2 | source_constant_segment | 0.164957024 |
| deployable | felm/formatted_tables | 2 | prior_base | 0.164823246 |
| deployable | felm/formatted_tables | 2 | prior_student | 0.163911529 |
| deployable | felm/formatted_tables | 2 | prior_permuted_student | 0.163875248 |
| deployable | felm/formatted_tables | 2 | prior_demand_only | 0.163724351 |
| deployable | felm/formatted_tables | 2 | local_student | 0.166015746 |
| deployable | felm/formatted_tables | 2 | local_permuted_student | 0.166015833 |
| deployable | felm/formatted_tables | 2 | local_demand_only | 0.167095822 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | R25 | 0.255489917 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | R56_segment | 0.255179186 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | R66_item_segment | 0.255360450 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | constant_0.5 | 0.250000000 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | smoothed_mean_2 | 0.258686472 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | training_selected_smoothed | 0.255688807 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | source_constant_segment | 0.256610894 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | prior_base | 0.255809595 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | prior_student | 0.256095803 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | prior_permuted_student | 0.255887800 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | prior_demand_only | 0.259463643 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | local_student | 0.255487631 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | local_permuted_student | 0.255494021 |
| deployable | fewshot_ttt_bbh/formatted_tables | 22 | local_demand_only | 0.258634556 |
| deployable | financebench/formatted_tables | 32 | R25 | 0.203618828 |
| deployable | financebench/formatted_tables | 32 | R56_segment | 0.209389841 |
| deployable | financebench/formatted_tables | 32 | R66_item_segment | 0.204661617 |
| deployable | financebench/formatted_tables | 32 | constant_0.5 | 0.250000000 |
| deployable | financebench/formatted_tables | 32 | smoothed_mean_2 | 0.204612039 |
| deployable | financebench/formatted_tables | 32 | training_selected_smoothed | 0.204612039 |
| deployable | financebench/formatted_tables | 32 | source_constant_segment | 0.203893846 |
| deployable | financebench/formatted_tables | 32 | prior_base | 0.204010275 |
| deployable | financebench/formatted_tables | 32 | prior_student | 0.204124881 |
| deployable | financebench/formatted_tables | 32 | prior_permuted_student | 0.204290570 |
| deployable | financebench/formatted_tables | 32 | prior_demand_only | 0.203880320 |
| deployable | financebench/formatted_tables | 32 | local_student | 0.203600873 |
| deployable | financebench/formatted_tables | 32 | local_permuted_student | 0.203601914 |
| deployable | financebench/formatted_tables | 32 | local_demand_only | 0.204576781 |
| deployable | finegrain_t2i/formatted_tables | 10 | R25 | 0.273116648 |
| deployable | finegrain_t2i/formatted_tables | 10 | R56_segment | 0.274795622 |
| deployable | finegrain_t2i/formatted_tables | 10 | R66_item_segment | 0.274846258 |
| deployable | finegrain_t2i/formatted_tables | 10 | constant_0.5 | 0.250000000 |
| deployable | finegrain_t2i/formatted_tables | 10 | smoothed_mean_2 | 0.274263550 |
| deployable | finegrain_t2i/formatted_tables | 10 | training_selected_smoothed | 0.269145940 |
| deployable | finegrain_t2i/formatted_tables | 10 | source_constant_segment | 0.274150471 |
| deployable | finegrain_t2i/formatted_tables | 10 | prior_base | 0.274672363 |
| deployable | finegrain_t2i/formatted_tables | 10 | prior_student | 0.274123064 |
| deployable | finegrain_t2i/formatted_tables | 10 | prior_permuted_student | 0.274992538 |
| deployable | finegrain_t2i/formatted_tables | 10 | prior_demand_only | 0.273554871 |
| deployable | finegrain_t2i/formatted_tables | 10 | local_student | 0.273089448 |
| deployable | finegrain_t2i/formatted_tables | 10 | local_permuted_student | 0.273101781 |
| deployable | finegrain_t2i/formatted_tables | 10 | local_demand_only | 0.274205261 |
| deployable | hallusionbench/formatted_tables | 50 | R25 | 0.246902681 |
| deployable | hallusionbench/formatted_tables | 50 | R56_segment | 0.246375292 |
| deployable | hallusionbench/formatted_tables | 50 | R66_item_segment | 0.245915321 |
| deployable | hallusionbench/formatted_tables | 50 | constant_0.5 | 0.250000000 |
| deployable | hallusionbench/formatted_tables | 50 | smoothed_mean_2 | 0.244428381 |
| deployable | hallusionbench/formatted_tables | 50 | training_selected_smoothed | 0.242321977 |
| deployable | hallusionbench/formatted_tables | 50 | source_constant_segment | 0.246004691 |
| deployable | hallusionbench/formatted_tables | 50 | prior_base | 0.246570402 |
| deployable | hallusionbench/formatted_tables | 50 | prior_student | 0.246700736 |
| deployable | hallusionbench/formatted_tables | 50 | prior_permuted_student | 0.246658290 |
| deployable | hallusionbench/formatted_tables | 50 | prior_demand_only | 0.246014017 |
| deployable | hallusionbench/formatted_tables | 50 | local_student | 0.246909671 |
| deployable | hallusionbench/formatted_tables | 50 | local_permuted_student | 0.246906335 |
| deployable | hallusionbench/formatted_tables | 50 | local_demand_only | 0.244427254 |
| deployable | healthadminbench/formatted_tables | 14 | R25 | 0.245827727 |
| deployable | healthadminbench/formatted_tables | 14 | R56_segment | 0.248999557 |
| deployable | healthadminbench/formatted_tables | 14 | R66_item_segment | 0.242702060 |
| deployable | healthadminbench/formatted_tables | 14 | constant_0.5 | 0.250000000 |
| deployable | healthadminbench/formatted_tables | 14 | smoothed_mean_2 | 0.251664271 |
| deployable | healthadminbench/formatted_tables | 14 | training_selected_smoothed | 0.247706656 |
| deployable | healthadminbench/formatted_tables | 14 | source_constant_segment | 0.240711106 |
| deployable | healthadminbench/formatted_tables | 14 | prior_base | 0.238329457 |
| deployable | healthadminbench/formatted_tables | 14 | prior_student | 0.238909764 |
| deployable | healthadminbench/formatted_tables | 14 | prior_permuted_student | 0.238763354 |
| deployable | healthadminbench/formatted_tables | 14 | prior_demand_only | 0.238467622 |
| deployable | healthadminbench/formatted_tables | 14 | local_student | 0.245803330 |
| deployable | healthadminbench/formatted_tables | 14 | local_permuted_student | 0.245806309 |
| deployable | healthadminbench/formatted_tables | 14 | local_demand_only | 0.251546828 |
