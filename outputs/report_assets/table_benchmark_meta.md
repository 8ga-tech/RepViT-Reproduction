| model | input_size | batch_size | precision | backend | hardware | ort_version | runs | file_size_mb | io_nodes | mean_ms | p50_ms | p95_ms | threads_intra |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| repvit_m0_9_in1k | 224 | 1 | FP32 | CPUExecutionProvider | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 20.36 | in=NoneNone out=NoneNone | 7.446 | 7.373 | 8.002 | 4 |
| repvit_m1_0_in1k | 224 | 1 | FP32 | CPUExecutionProvider | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 27.33 | in=NoneNone out=NoneNone | 9.202 | 9.147 | 9.76 | 4 |
| repvit_m0_9_pet37 | 224 | 1 | FP32 | CPUExecutionProvider | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 18.89 | in=NoneNone out=NoneNone | 7.2 | 7.122 | 7.692 | 4 |
| repvit_m1_1_in1k | 224 | 1 | FP32 | CPUExecutionProvider | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 33.06 | in=NoneNone out=NoneNone | 10.407 | 10.234 | 11.241 | 4 |
| repvit_m1_5_in1k | 224 | 1 | FP32 | CPUExecutionProvider | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 56.35 | in=NoneNone out=NoneNone | 17.779 | 17.647 | 18.926 | 4 |
| repvit_m2_3_in1k | 224 | 1 | FP32 | CPUExecutionProvider | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 91.9 | in=NoneNone out=NoneNone | 32.65 | 32.687 | 33.322 | 4 |
