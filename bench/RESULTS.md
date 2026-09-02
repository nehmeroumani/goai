# GoAI vs Vercel AI SDK -- Benchmark Report

> Average of **3 independent sequential runs**. Both sides use in-process mock servers
> serving identical SSE fixtures (Chat Completions API) -- no real API calls, no network jitter.

## Environment

- **Machine**: Apple M2
- **OS**: macOS 26.3.1 (Darwin 25.3.0)
- **Go**: go1.26.0
- **Bun**: 1.3.10
- **AI SDK**: 5.0.124 / @ai-sdk/openai 2.0.89
- **Date**: 2026-03-17
- **Fixture**: 100 SSE chunks x 500-byte text = 50KB payload
- **Commit**: 1fa1a8f

## Results (average of 3 runs)

| Benchmark | GoAI (Go) | Vercel AI SDK (TS) | Winner |
|-----------|-----------|-------------------|--------|
| **Streaming throughput** | 1.46ms/op | 1.62ms/op | **1.1x Go** |
| **Time to first chunk** | 320.7μs | 412.3μs | **1.3x Go** |
| **Cold start** | 569.2μs | 13.89ms | **24.4x Go** |
| **Schema generation** | 3.6μs/op | 3.5μs/op | **1.0x TS** |
| **Memory (1 stream)** | 220KB | 676KB | **3.1x Go** |
| **GenerateText** | 55.7μs/op | 79.0μs/op | **1.4x Go** |

### Memory (100 streams) -- high variance, not reliable

TS memory measurement shows extreme variance across runs for 100 concurrent streams.
Go is stable. This benchmark is **not suitable for comparison** due to
`process.memoryUsage().heapUsed` unreliability with `Bun.gc()`.

## Per-Run Evidence

### Streaming throughput (100 chunks x 500B)
| Run | GoAI | Vercel AI SDK | Ratio |
|-----|------|---------------|-------|
| 1 | 1.46ms | 1.62ms | 1.11x |
| 2 | 1.46ms | 1.62ms | 1.11x |
| 3 | 1.46ms | 1.64ms | 1.12x |

### Time to first chunk
| Run | GoAI | Vercel AI SDK | Ratio |
|-----|------|---------------|-------|
| 1 | 321.7μs | 417.2μs | 1.30x |
| 2 | 315.4μs | 412.3μs | 1.31x |
| 3 | 320.7μs | 412.0μs | 1.28x |

### Cold start (median of 20 process launches)
| Run | GoAI | Vercel AI SDK | Ratio |
|-----|------|---------------|-------|
| 1 | 569.2μs | 13.89ms | 24.41x |
| 2 | 576.0μs | 13.98ms | 24.26x |
| 3 | 561.4μs | 13.88ms | 24.72x |

### Schema generation (simple struct)
| Run | GoAI | Vercel AI SDK | Ratio |
|-----|------|---------------|-------|
| 1 | 3.6μs | 3.5μs | 0.96x |
| 2 | 3.6μs | 3.5μs | 0.98x |
| 3 | 3.7μs | 3.5μs | 0.94x |

### GenerateText (non-streaming)
| Run | GoAI | Vercel AI SDK | Ratio |
|-----|------|---------------|-------|
| 1 | 55.4μs | 78.3μs | 1.41x |
| 2 | 55.7μs | 79.0μs | 1.42x |
| 3 | 56.2μs | 79.6μs | 1.42x |

### Memory (1 stream)
| Run | GoAI | Vercel AI SDK | Ratio |
|-----|------|---------------|-------|
| 1 | 231KB | 676KB | 2.93x |
| 2 | 220KB | 676KB | 3.07x |
| 3 | 145KB | 676KB | 4.65x |

## Methodology

- **Execution**: All runs are **sequential** (Go completes before TS starts) to avoid CPU contention
  on M2 cores. Running in parallel inflates TS numbers by 20-50% due to JIT sensitivity to CPU load.
- **API format**: Both sides use **Chat Completions API** (`/v1/chat/completions`) with identical
  SSE fixtures. GoAI uses `WithProviderOptions({"useResponsesAPI": false})`, TS uses `provider.chat()`.
- **Mock servers**: Go `httptest.Server`, TS `Bun.serve()` -- both in-process.
- **Streaming throughput**: Full stream lifecycle -- HTTP POST -> SSE parse -> channel/iterator -> close.
- **Time to first chunk**: `StreamText()` -> first text chunk. Drain outside timed region.
  Go uses `b.StopTimer()`/`b.StartTimer()`. TS uses `Bun.nanoseconds()` with 10-iteration warmup.
- **Cold start**: Standalone binary/process x20 runs. Includes runtime init + mock server + one `GenerateText()`.
- **Schema**: Go `SchemaFrom[T]()` (reflection) vs TS `zodSchema()` + `.jsonSchema` (Zod -> JSON Schema).
- **Memory**: Heap delta after GC. High variance on TS side -- directional only.
- **Verification**: Both sides verified to produce identical output (100 chunks, 50,000 bytes streaming; 500 bytes non-streaming).

## How to reproduce

```bash
cd bench
make bench-all    # runs both Go + TS benchmarks and generates single-run report
make bench-3x     # 3 independent runs, averaged into RESULTS.md
make bench-go     # Go only
make bench-ts     # TS only
make report       # regenerate single-run report from existing results
```
