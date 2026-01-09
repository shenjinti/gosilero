# Gosilero (Go)

A pure-Go reimplementation of the Silero VAD.

## Building & Running

1. `go test ./...` exercises the inference pipeline plus the regression on `testdata/1843344-user.wav` (the 9.3‑9.8 s, 14.7‑15.2 s, and 19.2‑19.7 s ranges using a 0.5 threshold and 512-sample chunks).
2. `go run cmd/gosilero/main.go -file testdata/thankyou_16k.wav` runs the CLI, prints the detected segments, and still supports the threshold/padding flags (chunk size remains fixed at 512 samples).

## RTF measurement

`TestPerformanceRTF` runs inference over `testdata/1843344-user.wav` and logs both the real-time factor (RTF) for the full file plus the average wall-clock time to process a single 20 ms frame (assuming 320 samples per frame). 

Execute `go test -run TestPerformanceRTF -count=1` to see the output in the logs; it uses the same WAV referenced earlier so you can directly compare RTF numbers between runs or machines.

For reference, a recent [Rust Tiny Silero ](https://github.com/restsend/active-call) run reported `RTF = 0.001` and `0.03 ms` per 20 ms frame. 

The pure-Go engine logs ~`RTF = 0.0029` and `0.06 ms` per frame on the same file.

## VAD Performance

Benchmarked on 60 seconds of 16kHz audio (Release mode):

| VAD Engine | Implementation | Time (60s) | RTF (Ratio) | Note |
|------------|----------------|------------|-------------|------|
| **gosilero** | Go | ~175.0 ms | 0.0029 | Slower (Not SIMD), Pure GO |
| **TinySilero** | Rust (Optimized) | ~60.0 ms | 0.0010 | >2.5x faster than ONNX |
| **ONNX Silero** | ONNX Runtime | ~158.3 ms | 0.0026 | Standard baseline |
| **WebRTC VAD** | C/C++ (Bind) | ~3.1 ms | 0.00005 | Legacy, less accurate |