Small CUDA example that captures audio via ALSA **snd-aloop** and runs a **GPU FFT** using **cuFFT**.  
This code is used in my LinkedIn article about doing FFT on Jetson with a software loopback (simulated I²S).

**What it shows**
- Deterministic capture from `hw:Loopback` (ALSA loopback card)
- Single FFT timing vs **batched (×10) FFTs** timing on the GPU
- Peak frequency detection from the spectrum

---

## Contents

- `signal_capture.cu` — main example (capture → window → 1× FFT + 10× batched FFTs, timings & peak freq)
- `README.md` — this file

---

## Requirements

- NVIDIA Jetson (Nano / Orin Nano etc.) with CUDA & cuFFT (JetPack)
- ALSA userspace dev libs: `libasound2-dev`
- ALSA utils (for testing): `alsa-utils` (for `aplay`, `speaker-test`)
- A configured **snd-aloop** device (see below)

---

## Build

```bash
nvcc -O2 -o signal_capture signal_capture.cu -lasound -lcufft

