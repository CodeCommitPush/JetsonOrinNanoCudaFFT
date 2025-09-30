#include <alsa/asoundlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <utility>

inline void CheckCuda(cudaError_t e, const char* file, int line) {
  if (e != cudaSuccess) {
    printf("Cuda Error: %d at %s:%d\n", e, file, line);
    std::exit(1);
  }
}

#define CHECK_CUDA(expr) CheckCuda((expr), __FILE__, __LINE__)

inline void CheckCuFFT(cufftResult r, const char* file, int line) {
  if (r != CUFFT_SUCCESS) {
    printf("Cuda FFT Error: %d at %s:%d\n", (int)r, file, line);
    std::exit(1);
  }
}
#define CHECK_CUFFT(expr) CheckCuFFT((expr), __FILE__, __LINE__)



/**
 * @brief Open and configure an ALSA PCM device
 *
 *
 * @param[out] pcm       snd_pcm_t* handle.
 * @param[in]  dev       ALSA device string
 * @param[in]  channels  Number of channels to capture
 * @param[in]  rate      Sampling rate in Hz 
 * @param[in]  period    Period size in frames 
 * @param[in]  buffer    Desired total buffer size in frames 
  */
static void AlsaOpenAndConfigurePcm(snd_pcm_t** pcm, const char* dev,
                                    unsigned channels, unsigned rate,
                                    snd_pcm_uframes_t period, snd_pcm_uframes_t buffer) {
                                    
  if (snd_pcm_open(pcm, dev, SND_PCM_STREAM_CAPTURE, 0) < 0)
  {
     printf("Alsa Error: snd_pcm_open");
  }

  snd_pcm_hw_params_t* hw = nullptr;
  snd_pcm_hw_params_malloc(&hw);
  snd_pcm_hw_params_any(*pcm, hw);
  snd_pcm_hw_params_set_access(*pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
  snd_pcm_hw_params_set_format(*pcm, hw, SND_PCM_FORMAT_S16_LE);
  snd_pcm_hw_params_set_channels(*pcm, hw, channels);

  unsigned int exact_rate = rate;
  int dir = 0;
  snd_pcm_hw_params_set_rate_near(*pcm, hw, &exact_rate, &dir);

  snd_pcm_hw_params_set_period_size_near(*pcm, hw, &period, &dir);
  snd_pcm_hw_params_set_buffer_size_near(*pcm, hw, &buffer);


  if (snd_pcm_hw_params(*pcm, hw) < 0) 
  {
     printf("Alsa Error: snd_pcm_hw_params");
  }

  snd_pcm_hw_params_free(hw);

  if (snd_pcm_prepare(*pcm) < 0)
  {
     printf("Alsa Error: snd_pcm_prepare");
  }
}

/**
 * @brief Check if an integer is a power of two.
 *
 * @param n  Value to test.
 * @return true  If  n is a power of two, otherwise false
 */
static bool IsPowerOfTwo(size_t n) {
  return n && ((n & (n - 1)) == 0);
}

/**
 * @brief Find the peak (maximum power) bin in a real-to-complex FFT spectrum.
 * @param X Pointer to cuFFT output  are positive-frequency bins.
 * @param N FFT size 
 * @return std::pair<int,double>
 *         - first  = index k of the max-power bin (1..N/2),
 *         - second = power at that bin (re^2 + im^2).
 */
static std::pair<int,double> FindPeakCuFFT(const cufftComplex* X, int N) {
  int best_k = 1;     // skip DC
  double best_p = 0.0;
  for (int k = 1; k <= N/2; ++k) {
    double re = X[k].x, im = X[k].y;
    double p = re*re + im*im;
    if (p > best_p) { best_p = p; best_k = k; }
  }
  return {best_k, best_p};
}

int main(int argc, char** argv) {
  const char* dev = (argc > 1) ? argv[1] : "hw:2,1,0"; // Loopback capture
  const int    N  = (argc > 2) ? std::atoi(argv[2]) : 4096;     // FFT size
  const int    SR = (argc > 3) ? std::atoi(argv[3]) : 48000;    // sample rate
  const int BATCH = 10;

  if (!IsPowerOfTwo(N) || N < 32) {
    printf("Error: Please use a power-of-two N >= 32 (e.g., 4096, 8192, 16384).\n");
    return 1;
  }

  // Capture N stereo frames
  snd_pcm_t* pcm = nullptr;
  AlsaOpenAndConfigurePcm(&pcm, dev, /*channels*/2, /*rate*/SR,
                          /*period*/512, /*buffer*/4096);

  std::vector<int16_t> interleaved(N * 2);
  size_t frames_read_total = 0;
  while (frames_read_total < (size_t)N) {
    snd_pcm_sframes_t result = snd_pcm_readi(pcm,
                          interleaved.data() + frames_read_total * 2,
                          N - frames_read_total);
    if (result == -EAGAIN) continue;
    if (result == -EPIPE) { snd_pcm_prepare(pcm); continue; } // recover overrun
    if (result < 0) {
        printf("snd_pcm_readi result is %d.", (int)result);
    }
    
    frames_read_total += (size_t)result;
  }
  snd_pcm_drain(pcm);
  snd_pcm_close(pcm);

  // LEFT channel -> float + Hann window
  std::vector<float> hL(N);
  const double pi = 3.14159265358979323846;
  const double two_pi = 2.0 * pi;
  for (int i = 0; i < N; ++i) {
    float w = 0.5f * (1.0f - std::cos((float)(two_pi * i / (N - 1))));
    hL[i] = w * (interleaved[2*i + 0] / 32768.0f); // normalize S16 -> [-1,1)
  }

  // Single FFT
  float* dL_single = nullptr;
  cufftComplex* dF_single = nullptr;
  CHECK_CUDA(cudaMalloc(&dL_single,  N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dF_single, (N/2 + 1) * sizeof(cufftComplex)));
  CHECK_CUDA(cudaMemcpy(dL_single, hL.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  cufftHandle plan_single;
  CHECK_CUFFT(cufftPlan1d(&plan_single, N, CUFFT_R2C, /*batch*/1));

  cudaEvent_t s_start, s_stop;
  CHECK_CUDA(cudaEventCreate(&s_start));
  CHECK_CUDA(cudaEventCreate(&s_stop));
  CHECK_CUDA(cudaEventRecord(s_start));
  CHECK_CUFFT(cufftExecR2C(plan_single, dL_single, dF_single));
  CHECK_CUDA(cudaEventRecord(s_stop));
  CHECK_CUDA(cudaEventSynchronize(s_stop));
  float single_ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&single_ms, s_start, s_stop));

  // Copy back for peak frequency
  std::vector<cufftComplex> hF_single(N/2 + 1);
  CHECK_CUDA(cudaMemcpy(hF_single.data(), dF_single,
                        (N/2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
  auto peak_single = FindPeakCuFFT(hF_single.data(), N);
  double bin_hz = (double)SR / (double)N;
  double f_single = peak_single.first * bin_hz;

  // Batched 10 FFTs
  std::vector<float> hL_batch(N * BATCH);
  for (int b = 0; b < BATCH; ++b) {
    std::memcpy(hL_batch.data() + b*N, hL.data(), N * sizeof(float));
  }
  float* dL_batch = nullptr;
  cufftComplex* dF_batch = nullptr;
  CHECK_CUDA(cudaMalloc(&dL_batch,  N * BATCH * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dF_batch, (N/2 + 1) * BATCH * sizeof(cufftComplex)));
  CHECK_CUDA(cudaMemcpy(dL_batch, hL_batch.data(),
                        N * BATCH * sizeof(float), cudaMemcpyHostToDevice));

  cufftHandle plan_batch;
  CHECK_CUFFT(cufftPlan1d(&plan_batch, N, CUFFT_R2C, BATCH));

  cudaEvent_t b_start, b_stop;
  CHECK_CUDA(cudaEventCreate(&b_start));
  CHECK_CUDA(cudaEventCreate(&b_stop));
  CHECK_CUDA(cudaEventRecord(b_start));
  CHECK_CUFFT(cufftExecR2C(plan_batch, dL_batch, dF_batch));
  CHECK_CUDA(cudaEventRecord(b_stop));
  CHECK_CUDA(cudaEventSynchronize(b_stop));
  float batch_ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&batch_ms, b_start, b_stop));

  // Check peak in the 10x FFT calculations
  std::vector<cufftComplex> hF_batch_first(N/2 + 1);
  CHECK_CUDA(cudaMemcpy(hF_batch_first.data(), dF_batch,
                        (N/2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
  auto peak_batch = FindPeakCuFFT(hF_batch_first.data(), N);
  double f_batch = peak_batch.first * bin_hz;

  // Cleanup
  cufftDestroy(plan_single);
  cufftDestroy(plan_batch);
  cudaEventDestroy(s_start);
  cudaEventDestroy(s_stop);
  cudaEventDestroy(b_start);
  cudaEventDestroy(b_stop);
  cudaFree(dL_single);
  cudaFree(dF_single);
  cudaFree(dL_batch);
  cudaFree(dF_batch);

  // Print results
  printf("Captured frames: %d frames.  Sample rate %d Hz (using LEFT channel)\n", N, SR);
  printf("FFT size: %d  (bin width = %.6f Hz)\n\n", N, bin_hz);

  printf("GPU single FFT:  %10.6f ms   |  peak ~ %.3f Hz (k=%d) |\n",
         single_ms, f_single, peak_single.first);
  printf("GPU 10 FFTs :    %10.6f ms   |  peak ~ %.3f Hz (k=%d) |  avg/FFT = %.6f ms\n",
         batch_ms, f_batch, peak_batch.first, batch_ms / BATCH);

  return 0;
}
