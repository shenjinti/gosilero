# Gosilero

Gosilero is a silero-vad library built using Rust, designed to provide efficient and accurate speech detection capabilities. This library leverages the performance and safety of Rust while offering seamless integration with Go through CGo bindings.


# how to build 

## build rust project first
```shell
cd gosilero-rs
# build for linux
cargo build --release --target=x86_64-unknown-linux-gnu
cp target/x86_64-unknown-linux-gnu/release/libgosilero_rs.so ../dist/

# mac
cargo build --release --target=aarch64-apple-darwin
cp target/aarch64-apple-darwin/release/libgosilero_rs.dylib ../dist/

```
## build go 


```shell
#mac 
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:`pwd`/dist

# linux
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/dist
go run cmd/gosilero/main.go -file testdata/thankyou_16k.wav

File: testdata/thankyou_16k.wav
Sample rate: 16000 Hz, Channels: 1, Bits per sample: 16
Duration: 3.611 seconds, Samples: 57783
VAD parameters: threshold=0.60, silence=200ms, pre-padding=50ms, post-padding=50ms

Detected speech segments:
-------------------------
Segment 1: 1.774s - 2.642s (duration: 0.868s), peak: 0.9987 at 2.208s
-------------------------
Total speech: 0.868 seconds (24.0% of file)
Segments detected: 1
```