# Precision operator provenance

The optional operators are extracted and adapted from Waveform Last Mile,
Copyright (c) 2026 Mason Wang, MIT license below. This contribution factors its
static phase geometry into a portable helper and integrates opt-in preparation
with mlx-bigvgan's existing loader. Model weights, audio, the model implementation,
and the original benchmark/audit harness are not included.

The originating model layout credits yrom/mlx-bigvgan at
62a89425f07b1cc109cd7752ef50ba3c2cece06f (Copyright 2025 Yrom Wang, MIT),
and NVIDIA/BigVGAN at 7d2b454564a6c7d014227f635b7423881f14bdac
(Copyright 2024 NVIDIA CORPORATION, MIT). This patch does not copy their
activation/resampling code or replace their existing license notices.

MIT License

Copyright (c) 2026 Mason Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

