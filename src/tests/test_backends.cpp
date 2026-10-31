#include "BackendFactory.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <ctime>
#include <cassert>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

int main(int argc, char** argv) {
    std::cout << "==========================================" << std::endl;
    std::cout << "Unit Test: Multi-Backend Audio & Sentry Parity" << std::endl;
    std::cout << "==========================================" << std::endl;

    std::vector<std::string> testModels;
    bool useGpu = false;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--gpu" || arg == "-g") {
                useGpu = true;
            } else {
                testModels.emplace_back(arg);
            }
        }
    }
    if (testModels.empty()) {
        testModels = {
            // TorchScript reference models
            "models/darbouka.ts",
            "models/encodec_bendable.ts",
            "models/speech.ts",
            "models/stable_audio_1_vae.ts",
            // Additional TS models if present
            "models/voice_vocalset_b2048_r48000_z16.ts",
            "models/sax_soprano_franziskaschroeder_b2048_r48000_z20.ts",
            // ONNX bendable models
            "models/darbouka_bendable.onnx",
            "models/encodec_bendable.onnx",
            "models/speech_bendable.onnx",
            "models/stable_audio_1_vae_bendable.onnx"
        };
        // Filter out non-existent TS models
        std::vector<std::string> existingModels;
        for (const auto& m : testModels) {
            std::ifstream f(m);
            if (f.good()) existingModels.push_back(m);
        }
        testModels = std::move(existingModels);

        const std::vector<std::string> pteModels = useGpu
            ? std::vector<std::string>{
                "models/darbouka_coreml.pte",
                "models/encodec_coreml.pte",
                "models/speech_coreml.pte",
                "models/stable_audio_1_vae_coreml.pte"
            }
            : std::vector<std::string>{
                "models/darbouka_xnnpack.pte",
                "models/encodec_xnnpack.pte",
                "models/speech_xnnpack.pte",
                "models/stable_audio_1_vae_xnnpack.pte"
            };
        for (const auto& pte : pteModels) {
            std::ifstream f(pte);
            if (f.good()) {
                testModels.push_back(pte);
            }
        }
    }

    int failures = 0;
    std::string reportPath = useGpu ? "test_report_gpu.md" : "test_report_cpu.md";
    std::ofstream report(reportPath);

    report << "# NetBender Audio Parity & Sentry Verification Report\n\n";
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    report << "**Timestamp**: " << std::ctime(&now) << "\n\n";

    struct BenchmarkStats {
        bool ran = false;
        double minMs = 0.0;
        double avgMs = 0.0;
        double maxMs = 0.0;
        double budgetMs = 0.0;
        double rtf = 0.0;       // Compute time / Audio duration
        double speedup = 0.0;   // Audio duration / Compute time (e.g. 5x real-time)
        bool realTimeSafe = false;
    };

    struct ModelTestResult {
        std::string path;
        std::string backendType;
        bool loadSuccess = false;
        std::string loadError;
        double sr = 0;
        int ratio = 0;
        int latentDim = 0;
        size_t numLayers = 0;
        std::vector<std::string> modes;
        std::vector<std::string> bendableLayers;
        
        // Audio Quality & Sentry Status
        std::string fwdStatus = "N/A";
        float fwdPeak = 0.0f;
        float fwdRms = 0.0f;
        bool fwdFuseTripped = false;
        bool fwdHasNan = false;
        BenchmarkStats fwdBench;

        std::string decStatus = "N/A";
        float decPeak = 0.0f;
        float decRms = 0.0f;
        bool decFuseTripped = false;
        bool decHasNan = false;
        BenchmarkStats decBench;

        std::string priorStatus = "N/A";
        float priorPeak = 0.0f;
        float priorRms = 0.0f;
        bool priorFuseTripped = false;
        bool priorHasNan = false;
        BenchmarkStats priorBench;

        std::string failureDetails;
    };

    std::vector<ModelTestResult> results;

    for (const auto& path : testModels) {
        std::cout << "\n[TESTING] Model: " << path << std::endl;
        std::cout << std::flush;

        ModelTestResult res;
        res.path = path;

        auto backend = BackendFactory::createForFile(path);
        if (!backend) {
            res.loadSuccess = false;
            res.loadError = "Failed to allocate backend from BackendFactory.";
            results.push_back(res);
            failures++;
            continue;
        }
        res.backendType = backend->get_backend_name();

        int loadRet = -1;
        try {
            loadRet = backend->load(path, 44100.0);
            if (loadRet == 0 && useGpu) {
                backend->use_gpu(true);
                res.backendType = backend->get_backend_name() + " [GPU/CoreML]";
            }
        } catch (const std::exception& e) {
            res.loadError = e.what();
            loadRet = -1;
        } catch (...) {
            res.loadError = "Unknown exception during load()";
            loadRet = -1;
        }

        if (loadRet != 0) {
            if (res.loadError.empty()) res.loadError = backend->get_last_error();
            if (res.loadError.empty()) res.loadError = "Unknown error during initialization or shape inference.";
            std::cerr << "  FAILED: load() returned " << loadRet << ": " << res.loadError << std::endl;
            failures++;
            res.loadSuccess = false;
            results.push_back(res);
            continue;
        }

        res.loadSuccess = true;
        std::cout << "  SUCCESS: Loaded " << path << " (" << res.backendType << ")" << std::endl;
        res.sr = backend->get_sample_rate();
        res.ratio = backend->get_higher_ratio();
        auto layers = backend->get_available_layers();
        res.numLayers = layers.size();
        res.bendableLayers = layers;

        res.modes = backend->get_plugin_modes();
        {
            auto decParams = backend->get_mode_params("decode");
            if (!decParams.empty()) res.latentDim = decParams[0];
        }

        int inCh = 1;
        int outCh = 1;
        if (backend->has_method("forward")) {
            auto fwdParams = backend->get_mode_params("forward");
            if (fwdParams.size() >= 3) {
                inCh = std::max(1, fwdParams[0]);
                outCh = std::max(1, fwdParams[2]);
            }
        } else if (backend->has_method("decode")) {
            auto decParams = backend->get_mode_params("decode");
            if (decParams.size() >= 3) {
                outCh = std::max(1, decParams[2]);
            }
        }
        int n_vec = 2048;
        double budgetMs = (res.sr > 0.0) ? (n_vec / res.sr * 1000.0) : (2048.0 / 44100.0 * 1000.0);

        auto computeBench = [](const std::vector<double>& times, double budget) -> BenchmarkStats {
            BenchmarkStats stats;
            if (times.empty()) return stats;
            stats.ran = true;
            stats.budgetMs = budget;
            stats.minMs = *std::min_element(times.begin(), times.end());
            stats.maxMs = *std::max_element(times.begin(), times.end());
            double sum = 0.0;
            for (double t : times) sum += t;
            stats.avgMs = sum / times.size();
            stats.rtf = (budget > 0.0) ? (stats.avgMs / budget) : 0.0;
            stats.speedup = (stats.avgMs > 0.0) ? (budget / stats.avgMs) : 0.0;
            stats.realTimeSafe = (stats.rtf < 1.0);
            return stats;
        };

        // -------------------------------------------------------------
        // 1. STREAMING FORWARD TEST & BENCHMARK
        // -------------------------------------------------------------
        int numFwdBlocks = (res.latentDim >= 8 && res.ratio >= 1024) ? 16 : 8;
        std::cout << "  Testing perform_forward (" << numFwdBlocks << "-block stream, " << inCh << "->" << outCh << " ch)..." << std::endl;
        try {
            float maxPeak = 0.0f;
            double sumSq = 0.0;
            size_t totalSamples = 0;
            bool hadNan = false;
            bool fuseTripped = false;

            float maxFwdSeamJump = 0.0f;
            std::vector<float> prevLastSample(outCh, 0.0f);
            std::vector<double> fwdTimes;

            for (int b = 0; b < numFwdBlocks; ++b) {
                std::vector<std::vector<float>> inChannels(inCh, std::vector<float>(n_vec));
                std::vector<std::vector<float>> outChannels(outCh, std::vector<float>(n_vec, 0.0f));
                for (int c = 0; c < inCh; ++c) {
                    for (int i = 0; i < n_vec; ++i) {
                        float t = (float)(b * n_vec + i) / 44100.0f;
                        inChannels[c][i] = std::sin(6.2831853f * (100.0f + c * 50.0f) * t) * 0.7f;
                        if (i % 512 == 0) inChannels[c][i] += 0.2f;
                    }
                }

                std::vector<float*> inPtrs(inCh);
                std::vector<float*> outPtrs(outCh);
                for (int c = 0; c < inCh; ++c) inPtrs[c] = inChannels[c].data();
                for (int c = 0; c < outCh; ++c) outPtrs[c] = outChannels[c].data();

                auto t0 = std::chrono::high_resolution_clock::now();
                backend->perform_forward(inPtrs, outPtrs, 1, outCh, n_vec, nullptr);
                auto t1 = std::chrono::high_resolution_clock::now();
                double blockMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (b > 0) { // skip block 0 cold warm-up
                    fwdTimes.push_back(blockMs);
                }

                for (int c = 0; c < outCh; ++c) {
                    int warmupBlocks = (res.latentDim >= 8 && res.ratio >= 1024) ? 14 : 1;
                    if (b > warmupBlocks) { // evaluate after receptive field warm-up
                        float seamJump = std::abs(outChannels[c][0] - prevLastSample[c]);
                        if (seamJump > maxFwdSeamJump) maxFwdSeamJump = seamJump;
                    }
                    prevLastSample[c] = outChannels[c][n_vec - 1];

                    for (int i = 0; i < n_vec; ++i) {
                        float s = outChannels[c][i];
                        if (std::isnan(s) || std::isinf(s)) {
                            hadNan = true;
                        } else {
                            float abs_s = std::abs(s);
                            if (abs_s > maxPeak) maxPeak = abs_s;
                            if (abs_s > 2.0f) fuseTripped = true;
                            sumSq += (double)s * (double)s;
                            totalSamples++;
                        }
                    }
                }
            }

            res.fwdPeak = maxPeak;
            res.fwdRms = totalSamples > 0 ? (float)std::sqrt(sumSq / totalSamples) : 0.0f;
            res.fwdHasNan = hadNan;
            res.fwdFuseTripped = fuseTripped;
            res.fwdBench = computeBench(fwdTimes, budgetMs);

            std::cout << "    Forward: Peak=" << std::fixed << std::setprecision(4) << maxPeak
                      << " | RMS=" << res.fwdRms << " | NaN=" << (hadNan ? "YES" : "NO")
                      << " | MaxSeamJump=" << maxFwdSeamJump
                      << " | FuseCutoff(>2.0)=" << (fuseTripped ? "BLOWN" : "OK") << std::endl;
            if (res.fwdBench.ran) {
                std::cout << "      Benchmark: avg=" << std::setprecision(2) << res.fwdBench.avgMs
                          << " ms (min=" << res.fwdBench.minMs << ", max=" << res.fwdBench.maxMs
                          << ") | Budget=" << res.fwdBench.budgetMs << " ms | RTF=" << res.fwdBench.rtf
                          << " (" << res.fwdBench.speedup << "x RT)"
                          << (res.fwdBench.realTimeSafe ? " [RT SAFE]" : " [OVERRUN!]") << std::endl;
            }

            if (hadNan) {
                res.fwdStatus = "FAIL (NaN)";
                res.failureDetails += "Forward produced NaN/Inf\n";
                failures++;
            } else if (fuseTripped) {
                res.fwdStatus = "BLOWN FUSE";
                res.failureDetails += "Forward peak (" + std::to_string(maxPeak) + ") exceeded 2.0 safety fuse\n";
                failures++;
            } else if (res.fwdRms < 1e-5f) {
                res.fwdStatus = "⚠️ SILENT";
                res.failureDetails += "Forward produced near-zero audio (RMS < 1e-5)\n";
                failures++;
            } else {
                res.fwdStatus = "✅ PASS";
            }
        } catch (const std::exception& e) {
            std::cerr << "  FAILED: perform_forward exception: " << e.what() << std::endl;
            res.fwdStatus = "FAIL";
            res.failureDetails += std::string("perform_forward error: ") + e.what() + "\n";
            failures++;
        }

        // -------------------------------------------------------------
        // 2. DECODE GEN TEST & BENCHMARK
        // -------------------------------------------------------------
        if (backend->has_method("decode")) {
            int numDecBlocks = (res.latentDim >= 8 && res.ratio >= 1024) ? 20 : 10;
            std::cout << "  Testing perform_decode_gen (" << numDecBlocks << " blocks)..." << std::endl;
            try {
                float maxPeak = 0.0f;
                double sumSq = 0.0;
                size_t totalSamples = 0;
                bool hadNan = false;
                bool fuseTripped = false;

                float maxDecSeamJump = 0.0f;
                std::vector<float> prevDecLastSample(outCh, 0.0f);
                std::vector<double> decTimes;

                for (int b = 0; b < numDecBlocks; ++b) {
                    std::vector<std::vector<float>> outChannels(outCh, std::vector<float>(n_vec, 0.0f));
                    std::vector<float*> outPtrs(outCh);
                    for (int c = 0; c < outCh; ++c) outPtrs[c] = outChannels[c].data();

                    auto t0 = std::chrono::high_resolution_clock::now();
                    backend->perform_decode_gen(outPtrs, 1, outCh, n_vec, nullptr);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double blockMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (b > 0) {
                        decTimes.push_back(blockMs);
                    }

                    for (int c = 0; c < outCh; ++c) {
                        if (b > 1) { // evaluate after initial warm-up
                            float seamJump = std::abs(outChannels[c][0] - prevDecLastSample[c]);
                            if (seamJump > maxDecSeamJump) maxDecSeamJump = seamJump;
                        }
                        prevDecLastSample[c] = outChannels[c][n_vec - 1];

                        for (int i = 0; i < n_vec; ++i) {
                            float s = outChannels[c][i];
                            if (std::isnan(s) || std::isinf(s)) {
                                hadNan = true;
                            } else {
                                float abs_s = std::abs(s);
                                if (abs_s > maxPeak) maxPeak = abs_s;
                                if (abs_s > 2.0f) fuseTripped = true;
                                sumSq += (double)s * (double)s;
                                totalSamples++;
                            }
                        }
                    }
                }

                res.decPeak = maxPeak;
                res.decRms = totalSamples > 0 ? (float)std::sqrt(sumSq / totalSamples) : 0.0f;
                res.decHasNan = hadNan;
                res.decFuseTripped = fuseTripped;
                res.decBench = computeBench(decTimes, budgetMs);

                std::cout << "    Decode Gen: Peak=" << std::fixed << std::setprecision(4) << maxPeak
                          << " | RMS=" << res.decRms << " | NaN=" << (hadNan ? "YES" : "NO")
                          << " | MaxSeamJump=" << maxDecSeamJump
                          << " | FuseCutoff(>2.0)=" << (fuseTripped ? "BLOWN" : "OK") << std::endl;
                if (res.decBench.ran) {
                    std::cout << "      Benchmark: avg=" << std::setprecision(2) << res.decBench.avgMs
                              << " ms (min=" << res.decBench.minMs << ", max=" << res.decBench.maxMs
                              << ") | Budget=" << res.decBench.budgetMs << " ms | RTF=" << res.decBench.rtf
                              << " (" << res.decBench.speedup << "x RT)"
                              << (res.decBench.realTimeSafe ? " [RT SAFE]" : " [OVERRUN!]") << std::endl;
                }

                if (hadNan) {
                    res.decStatus = "FAIL (NaN)";
                    res.failureDetails += "Decode Gen produced NaN/Inf\n";
                    failures++;
                } else if (fuseTripped) {
                    res.decStatus = "BLOWN FUSE";
                    res.failureDetails += "Decode Gen peak (" + std::to_string(maxPeak) + ") exceeded 2.0 safety fuse\n";
                    failures++;
                } else if (res.decRms < 1e-5f) {
                    res.decStatus = "⚠️ SILENT";
                    res.failureDetails += "Decode Gen produced near-zero audio (RMS < 1e-5)\n";
                    failures++;
                } else {
                    res.decStatus = "✅ PASS";
                }
            } catch (const std::exception& e) {
                std::cerr << "  FAILED: perform_decode_gen exception: " << e.what() << std::endl;
                res.decStatus = "FAIL";
                res.failureDetails += std::string("perform_decode_gen error: ") + e.what() + "\n";
                failures++;
            }
        }

        // -------------------------------------------------------------
        // 3. PRIOR DECODE TEST & BENCHMARK
        // -------------------------------------------------------------
        if (backend->has_method("prior")) {
            std::cout << "  Testing perform_prior_decode (4 blocks)..." << std::endl;
            try {
                float maxPeak = 0.0f;
                double sumSq = 0.0;
                size_t totalSamples = 0;
                bool hadNan = false;
                bool fuseTripped = false;
                std::vector<double> priorTimes;

                for (int b = 0; b < 4; ++b) {
                    std::vector<std::vector<float>> outChannels(outCh, std::vector<float>(n_vec, 0.0f));
                    std::vector<float*> outPtrs(outCh);
                    for (int c = 0; c < outCh; ++c) outPtrs[c] = outChannels[c].data();

                    auto t0 = std::chrono::high_resolution_clock::now();
                    backend->perform_prior_decode(outPtrs, 1, outCh, n_vec);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double blockMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (b > 0) {
                        priorTimes.push_back(blockMs);
                    }

                    for (int c = 0; c < outCh; ++c) {
                        for (int i = 0; i < n_vec; ++i) {
                            float s = outChannels[c][i];
                            if (std::isnan(s) || std::isinf(s)) {
                                hadNan = true;
                            } else {
                                float abs_s = std::abs(s);
                                if (abs_s > maxPeak) maxPeak = abs_s;
                                if (abs_s > 2.0f) fuseTripped = true;
                                sumSq += (double)s * (double)s;
                                totalSamples++;
                            }
                        }
                    }
                }

                res.priorPeak = maxPeak;
                res.priorRms = totalSamples > 0 ? (float)std::sqrt(sumSq / totalSamples) : 0.0f;
                res.priorHasNan = hadNan;
                res.priorFuseTripped = fuseTripped;
                res.priorBench = computeBench(priorTimes, budgetMs);

                std::cout << "    Prior: Peak=" << std::fixed << std::setprecision(4) << maxPeak
                          << " | RMS=" << res.priorRms << " | NaN=" << (hadNan ? "YES" : "NO")
                          << " | FuseCutoff(>2.0)=" << (fuseTripped ? "BLOWN" : "OK") << std::endl;
                if (res.priorBench.ran) {
                    std::cout << "      Benchmark: avg=" << std::setprecision(2) << res.priorBench.avgMs
                              << " ms (min=" << res.priorBench.minMs << ", max=" << res.priorBench.maxMs
                              << ") | Budget=" << res.priorBench.budgetMs << " ms | RTF=" << res.priorBench.rtf
                              << " (" << res.priorBench.speedup << "x RT)"
                              << (res.priorBench.realTimeSafe ? " [RT SAFE]" : " [OVERRUN!]") << std::endl;
                }

                if (hadNan) {
                    res.priorStatus = "FAIL (NaN)";
                    res.failureDetails += "Prior produced NaN/Inf\n";
                    failures++;
                } else if (fuseTripped) {
                    res.priorStatus = "BLOWN FUSE";
                    res.failureDetails += "Prior peak (" + std::to_string(maxPeak) + ") exceeded 2.0 safety fuse\n";
                    failures++;
                } else if (res.priorRms < 1e-5f) {
                    res.priorStatus = "⚠️ SILENT";
                    res.failureDetails += "Prior produced near-zero audio (RMS < 1e-5)\n";
                    failures++;
                } else {
                    res.priorStatus = "✅ PASS";
                }
            } catch (const std::exception& e) {
                std::cerr << "  FAILED: perform_prior_decode exception: " << e.what() << std::endl;
                res.priorStatus = "FAIL";
                res.failureDetails += std::string("perform_prior_decode error: ") + e.what() + "\n";
                failures++;
            }
        }

        results.push_back(res);
    }

    // Write Overview Table
    report << "| Model Path | Engine | Status | Forward Peak (RMS) | Decode Gen (RMS) | Prior (RMS) |\n";
    report << "|---|---|---|---|---|---|\n";

    for (const auto& r : results) {
        if (!r.loadSuccess) {
            report << "| `" << r.path << "` | " << r.backendType << " | LOAD FAIL | - | - | - |\n";
        } else {
            auto fmtStat = [](const std::string& st, float peak, float rms) {
                if (st == "N/A") return std::string("N/A");
                return st + " (Pk:" + std::to_string(peak).substr(0, 5) + ", RMS:" + std::to_string(rms).substr(0, 6) + ")";
            };
            report << "| `" << r.path << "` | " << r.backendType << " | "
                   << (r.failureDetails.empty() ? "✅ PASS" : "⚠️ ISSUES") << " | "
                   << fmtStat(r.fwdStatus, r.fwdPeak, r.fwdRms) << " | "
                   << fmtStat(r.decStatus, r.decPeak, r.decRms) << " | "
                   << fmtStat(r.priorStatus, r.priorPeak, r.priorRms) << " |\n";
        }
    }

    // Write Performance & Real-Time Benchmark Table
    report << "\n## Performance & Real-Time Benchmark Summary\n\n";
    report << "| Model Path | Engine | Mode | Compute Time (avg / min / max) | RT Budget (2048 smp) | RTF | Speedup | Real-Time Safe |\n";
    report << "|---|---|---|---|---|---|---|---|\n";

    for (const auto& r : results) {
        if (!r.loadSuccess) continue;
        auto writeBenchRow = [&](const std::string& mode, const BenchmarkStats& b) {
            if (!b.ran) return;
            std::ostringstream avgStr, minMaxStr, budgetStr, rtfStr, spdStr;
            avgStr << std::fixed << std::setprecision(2) << b.avgMs << " ms";
            minMaxStr << std::fixed << std::setprecision(2) << b.minMs << " / " << b.maxMs << " ms";
            budgetStr << std::fixed << std::setprecision(2) << b.budgetMs << " ms";
            rtfStr << std::fixed << std::setprecision(3) << b.rtf;
            spdStr << std::fixed << std::setprecision(1) << b.speedup << "x";

            report << "| `" << r.path << "` | " << r.backendType << " | " << mode << " | "
                   << avgStr.str() << " (" << minMaxStr.str() << ") | "
                   << budgetStr.str() << " | "
                   << rtfStr.str() << " | "
                   << spdStr.str() << " | "
                   << (b.realTimeSafe ? "✅ SAFE" : "⚠️ OVERRUN") << " |\n";
        };
        writeBenchRow("Forward", r.fwdBench);
        writeBenchRow("Decode Gen", r.decBench);
        writeBenchRow("Prior", r.priorBench);
    }

    // Write Detailed Per-Model Breakdown
    report << "\n## Detailed Model Diagnostics\n\n";
    for (const auto& r : results) {
        report << "### `" << r.path << "` (" << r.backendType << ")\n\n";
        if (!r.loadSuccess) {
            report << "> [!CAUTION]\n";
            report << "> **Load Error**: " << r.loadError << "\n\n";
            continue;
        }

        report << "- **Engine**: `" << r.backendType << "`\n";
        report << "- **Sample Rate**: " << r.sr << " Hz | **Ratio**: " << r.ratio << " | **Latent Dim**: " << r.latentDim << "\n";
        report << "- **Forward Stream**: " << r.fwdStatus
               << " (Peak: " << r.fwdPeak << ", RMS: " << r.fwdRms << ")\n";
        if (r.fwdBench.ran) {
            report << "  - Compute: " << r.fwdBench.avgMs << " ms (min: " << r.fwdBench.minMs
                   << ", max: " << r.fwdBench.maxMs << ") | RTF: " << r.fwdBench.rtf
                   << " (" << r.fwdBench.speedup << "x real-time) -> "
                   << (r.fwdBench.realTimeSafe ? "✅ Real-time safe" : "⚠️ Overrun risk") << "\n";
        }
        report << "- **Decode Gen**: " << r.decStatus
               << " (Peak: " << r.decPeak << ", RMS: " << r.decRms << ")\n";
        if (r.decBench.ran) {
            report << "  - Compute: " << r.decBench.avgMs << " ms (min: " << r.decBench.minMs
                   << ", max: " << r.decBench.maxMs << ") | RTF: " << r.decBench.rtf
                   << " (" << r.decBench.speedup << "x real-time) -> "
                   << (r.decBench.realTimeSafe ? "✅ Real-time safe" : "⚠️ Overrun risk") << "\n";
        }
        if (r.priorStatus != "N/A") {
            report << "- **Prior Mode**: " << r.priorStatus
                   << " (Peak: " << r.priorPeak << ", RMS: " << r.priorRms << ")\n";
            if (r.priorBench.ran) {
                report << "  - Compute: " << r.priorBench.avgMs << " ms (min: " << r.priorBench.minMs
                       << ", max: " << r.priorBench.maxMs << ") | RTF: " << r.priorBench.rtf
                       << " (" << r.priorBench.speedup << "x real-time) -> "
                       << (r.priorBench.realTimeSafe ? "✅ Real-time safe" : "⚠️ Overrun risk") << "\n";
            }
        }

        if (!r.failureDetails.empty()) {
            report << "\n> [!WARNING]\n";
            report << "> **Detected Audio / Sentry Issues**:\n> ```\n> " << r.failureDetails << "> ```\n";
        }
    }

    std::cout << "\n==========================================" << std::endl;
    report << "\n## Summary\n\n";
    if (failures == 0) {
        std::cout << "ALL BACKEND TESTS PASSED!" << std::endl;
        report << "**Result**: **ALL TESTS PASSED WITH REAL AUDIO VALIDATION**\n";
        report.close();
        std::cout << "Saved test report to: " << reportPath << std::endl;
        return 0;
    } else {
        std::cerr << failures << " TEST(S) FAILED OR TRIPPED SENTRY!" << std::endl;
        report << "**Result**: ⚠️ **" << failures << " TEST(S) FAILED OR TRIPPED SENTRY**\n";
        report.close();
        std::cout << "Saved test report to: " << reportPath << std::endl;
        return 1;
    }
}
