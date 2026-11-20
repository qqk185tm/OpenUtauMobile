using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Classic;
using OpenUtau.Core.DiffSinger;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.Render {
    public class ChoseiCvRenderer : IRenderer {
        public const string SMOOTHING_EXPR = "smth";

        static ChoseiCvRenderer() {
#if ANDROID
    try {
        Java.Lang.JavaSystem.LoadLibrary("onnxruntime");
    } catch { }
#endif
}

        private static readonly object lockObj = new object();
        private static InferenceSession? refinerSession;
        private static DsVocoder? vocoder;
        private static float defaultStrength = 0.7f;

        private readonly WorldlineRenderer baseRenderer = new WorldlineRenderer();

        public USingerType SingerType => USingerType.Classic;
        public bool SupportsRenderPitch => true;
        public bool SupportsRealCurve => false;
        public bool SupportsPhonemeEnvelope => true;

        public bool SupportsExpression(UExpressionDescriptor descriptor) {
            return baseRenderer.SupportsExpression(descriptor) || descriptor.abbr == SMOOTHING_EXPR;
        }

        public RenderResult Layout(RenderPhrase phrase) {
            return baseRenderer.Layout(phrase);
        }


        public RenderPitchResult LoadRenderedPitch(RenderPhrase phrase) {
            return baseRenderer.LoadRenderedPitch(phrase);
        }

        public UExpressionDescriptor[] GetSuggestedExpressions(USinger singer, URenderSettings renderSettings) {
            var list = baseRenderer.GetSuggestedExpressions(singer, renderSettings).ToList();
            if (!list.Any(e => e.abbr == SMOOTHING_EXPR)) {
                list.Add(new UExpressionDescriptor("smoothing (chosei)", SMOOTHING_EXPR, 0, 100, 70));
            }
            return list.ToArray();
        }

        private static bool TryEnsureModelsLoaded() {
            if (refinerSession != null && vocoder != null) {
                return true;
            }
            lock (lockObj) {
                string depRoot = PathManager.Inst.DependencyPath;
                if (!Directory.Exists(depRoot)) {
                    return false;
                }

                if (refinerSession == null) {
                    string? refinerOnnx = Directory.GetFiles(depRoot, "chosei_cv_kurashikku.onnx", SearchOption.AllDirectories).FirstOrDefault();
                    if (refinerOnnx == null) {
                        return false;
                    }
                    string dir = Path.GetDirectoryName(refinerOnnx)!;
                    string configPath = Path.Combine(dir, "chosei_config.json");
                    if (File.Exists(configPath)) {
                        try {
                            using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
                            if (doc.RootElement.TryGetProperty("default_smoothing_strength", out var prop)) {
                                defaultStrength = prop.GetSingle();
                            }
                        } catch { }
                    }
                    refinerSession = Onnx.getInferenceSession(File.ReadAllBytes(refinerOnnx));
                }

                if (vocoder == null) {
                    string? vocoderYaml = Directory.GetFiles(depRoot, "vocoder.yaml", SearchOption.AllDirectories)
                        .OrderByDescending(p => p.Contains("pc_nsf_hifigan") || p.Contains("2025"))
                        .FirstOrDefault();
                    if (vocoderYaml == null) {
                        return false;
                    }
                    vocoder = new DsVocoder(Path.GetDirectoryName(vocoderYaml)!);
                }
                return true;
            }
        }

        public Task<RenderResult> Render(RenderPhrase phrase, Progress progress, int trackNo, CancellationTokenSource cancellation, bool isPreRender = false) {
            return Task.Run(() => {
                // Шаг 1: Черновая склейка фразы через встроенный нативный libworldline.so
                var draftResult = baseRenderer.Render(phrase, progress, trackNo, cancellation, isPreRender).GetAwaiter().GetResult();
                float[] audio = draftResult.samples;

                // Если .oudep или вокодер еще не импортированы — возвращаем чистый Worldline без вылета
                if (audio == null || audio.Length < 2048 || !TryEnsureModelsLoaded()) {
                    return draftResult;
                }

                const int sr = 44100;
                const int nFft = 2048;
                const int hopLength = 512;
                const int nMels = 128;

                float[,] logMel = ChoseiDsp.ComputeLogMelSpectrogram(audio, sr, nFft, hopLength, nMels, 40.0, 16000.0);
                int frames = logMel.GetLength(1);
                float[] f0 = ExtractF0FromPhrase(phrase, draftResult, frames, hopLength, sr);

                float strength = defaultStrength;
                if (phrase.curves != null) {
                    var smthCurve = phrase.curves.FirstOrDefault(c => c.Item1 == SMOOTHING_EXPR);
                    if (smthCurve.Item2 != null && smthCurve.Item2.Length > 0) {
                        strength = Math.Clamp(smthCurve.Item2.Average() / 100f, 0f, 1f);
                    }
                }


                var melTensor = new DenseTensor<float>(new[] { 1, nMels, frames });
                var f0Tensor = new DenseTensor<float>(new[] { 1, 1, frames });
                for (int t = 0; t < frames; t++) {
                    f0Tensor[0, 0, t] = f0[t];
                    for (int m = 0; m < nMels; m++) {
                        melTensor[0, m, t] = logMel[m, t];
                    }
                }
                var strengthTensor = new DenseTensor<float>(new[] { strength }, new[] { 1 });

                float[] refinedAudio;
                lock (lockObj) {
                    var refinerInputs = new List<NamedOnnxValue> {
                        NamedOnnxValue.CreateFromTensor("mel", melTensor),
                        NamedOnnxValue.CreateFromTensor("f0", f0Tensor),
                        NamedOnnxValue.CreateFromTensor("strength", strengthTensor)
                    };

                    using var refinerOut = refinerSession!.Run(refinerInputs);
                    var refinedMel = refinerOut.First().AsTensor<float>();

                    var vocoderMel = new DenseTensor<float>(new[] { 1, frames, nMels });
                    var vocoderF0 = new DenseTensor<float>(new[] { 1, frames });
                    for (int t = 0; t < frames; t++) {
                        vocoderF0[0, t] = f0[t];
                        for (int m = 0; m < nMels; m++) {
                            vocoderMel[0, t, m] = refinedMel[0, m, t];
                        }
                    }

                    var vocInputs = new List<NamedOnnxValue>();
                    var inputMeta = vocoder!.session.InputMetadata.Keys.ToList();
                    vocInputs.Add(NamedOnnxValue.CreateFromTensor(inputMeta.ElementAt(0), vocoderMel));
                    vocInputs.Add(NamedOnnxValue.CreateFromTensor(inputMeta.ElementAt(1), vocoderF0));
                    if (inputMeta.Count > 2) {
                        vocInputs.Add(NamedOnnxValue.CreateFromTensor(inputMeta.ElementAt(2), new DenseTensor<float>(new[] { 0f }, new[] { 1 })));
                    }

                    using var vocOut = vocoder.session.Run(vocInputs);
                    refinedAudio = vocOut.First().AsTensor<float>().ToArray();
                }

                float[] finalSamples = new float[audio.Length];
                Array.Copy(refinedAudio, finalSamples, Math.Min(refinedAudio.Length, finalSamples.Length));

                draftResult.samples = finalSamples;
                return draftResult;
            }, cancellation.Token);
        }

        private static float[] ExtractF0FromPhrase(RenderPhrase phrase, RenderResult draftResult, int frames, int hopLength, int sr) {
            float[] f0 = new float[frames];
            if (phrase.pitches == null || phrase.pitches.Length == 0) {
                return f0;
            }
            const int pitchIntervalTick = 5;
            double startMs = draftResult.positionMs - draftResult.leadingMs;

            for (int t = 0; t < frames; t++) {
                double frameMs = startMs + (t * hopLength * 1000.0 / sr);
                int tick = phrase.timeAxis.MsPosToTickPos(frameMs);
                int idx = (tick - (phrase.position - phrase.leading)) / pitchIntervalTick;
                if (idx >= 0 && idx < phrase.pitches.Length) {
                    float midiTone = phrase.pitches[idx] / 100f;
                    f0[t] = (float)(440.0 * Math.Pow(2.0, (midiTone - 69.0) / 12.0));
                } else if (t > 0) {
                    f0[t] = f0[t - 1];
                }
            }
            return f0;
        }
    }

    internal static class ChoseiDsp {
        private static float[,]? melFilterBank;

        public static float[,] ComputeLogMelSpectrogram(float[] audio, int sr, int nFft, int hopLength, int nMels, double fMin, double fMax) {
            melFilterBank ??= CreateMelFilterBank(sr, nFft, nMels, fMin, fMax);
            int pad = nFft / 2;
            float[] padded = new float[audio.Length + 2 * pad];
            Array.Copy(audio, 0, padded, pad, audio.Length);

            int frames = 1 + (padded.Length - nFft) / hopLength;
            int nFreqs = nFft / 2 + 1;
            float[,] logMel = new float[nMels, frames];

            double[] window = new double[nFft];
            for (int i = 0; i < nFft; i++) {
                window[i] = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / nFft));
            }

            Complex[] fftBuf = new Complex[nFft];
            double[] mag = new double[nFreqs];

            for (int t = 0; t < frames; t++) {
                int offset = t * hopLength;
                for (int i = 0; i < nFft; i++) {
                    fftBuf[i] = new Complex(padded[offset + i] * window[i], 0.0);
                }
                FftInPlace(fftBuf);
                for (int k = 0; k < nFreqs; k++) {
                    mag[k] = fftBuf[k].Magnitude;
                }
                for (int m = 0; m < nMels; m++) {
                    double sum = 0.0;
                    for (int k = 0; k < nFreqs; k++) {
                        sum += melFilterBank[m, k] * mag[k];
                    }
                    logMel[m, t] = (float)Math.Log(Math.Max(1e-5, sum));
                }
            }
            return logMel;
        }

        private static float[,] CreateMelFilterBank(int sr, int nFft, int nMels, double fMin, double fMax) {
            int nFreqs = nFft / 2 + 1;
            double HzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);
            double MelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

            double melMin = HzToMel(fMin);
            double melMax = HzToMel(fMax);
            double[] melPoints = new double[nMels + 2];
            int[] bins = new int[nMels + 2];

            for (int i = 0; i < nMels + 2; i++) {
                melPoints[i] = melMin + i * (melMax - melMin) / (nMels + 1);
                double hz = MelToHz(melPoints[i]);
                bins[i] = (int)Math.Floor((nFft + 1) * hz / sr);
            }

            float[,] fb = new float[nMels, nFreqs];
            for (int m = 1; m <= nMels; m++) {
                for (int k = bins[m - 1]; k < bins[m]; k++) {
                    if (k >= 0 && k < nFreqs && bins[m] != bins[m - 1]) {
                        fb[m - 1, k] = (float)(k - bins[m - 1]) / (bins[m] - bins[m - 1]);
                    }
                }
                for (int k = bins[m]; k < bins[m + 1]; k++) {
                    if (k >= 0 && k < nFreqs && bins[m + 1] != bins[m]) {
                        fb[m - 1, k] = (float)(bins[m + 1] - k) / (bins[m + 1] - bins[m]);
                    }
                }
                double enorm = 2.0 / (MelToHz(melPoints[m + 1]) - MelToHz(melPoints[m - 1]));
                for (int k = 0; k < nFreqs; k++) {
                    fb[m - 1, k] = (float)(fb[m - 1, k] * enorm);
                }
            }
            return fb;
        }

        private static void FftInPlace(Complex[] buffer) {
            int n = buffer.Length;
            for (int j = 1, i = 0; j < n; j++) {
                int bit = n >> 1;
                for (; (i & bit) != 0; bit >>= 1) i ^= bit;
                i ^= bit;
                if (j < i) (buffer[j], buffer[i]) = (buffer[i], buffer[j]);
            }
            for (int len = 2; len <= n; len <<= 1) {
                double ang = -2.0 * Math.PI / len;
                Complex wlen = new Complex(Math.Cos(ang), Math.Sin(ang));
                for (int i = 0; i < n; i += len) {
                    Complex w = Complex.One;
                    for (int j = 0; j < len / 2; j++) {
                        Complex u = buffer[i + j];
                        Complex v = buffer[i + j + len / 2] * w;
                        buffer[i + j] = u + v;
                        buffer[i + j + len / 2] = u - v;
                        w *= wlen;
                    }
                }
            }
        }
    }
}
