## By source

| | n | exact match [95% CI] | structure match | token edit rate |
|---|---:|---:|---:|---:|
| All test expressions | 1665 | 59.4% [57%–62%] | 70.2% | 0.081 |
| Human handwriting (MathWriting + CROHME) | 165 | 50.9% [44%–58%] | 62.4% | 0.205 |
| Synthetic (HASYv2 glyphs) | 1500 | 60.3% [58%–63%] | 71.1% | 0.068 |
| MathWriting (human) | 67 | 53.7% [42%–66%] | 64.2% | 0.217 |
| CROHME 2014 test | 98 | 49.0% [39%–59%] | 61.2% | 0.196 |

## By structure (all sources)

| | n | exact match [95% CI] | structure match | token edit rate |
|---|---:|---:|---:|---:|
| All | 1665 | 59.4% [57%–62%] | 70.2% | 0.081 |
| Flat | 421 | 88.6% [86%–91%] | 95.0% | 0.060 |
| Scripts (single level) | 423 | 54.8% [50%–60%] | 64.3% | 0.092 |
| Nested scripts | 384 | 33.9% [29%–39%] | 48.2% | 0.106 |
| Fractions | 437 | 58.1% [53%–63%] | 71.4% | 0.069 |

## By structure (human handwriting only)

| | n | exact match [95% CI] | structure match | token edit rate |
|---|---:|---:|---:|---:|
| All | 165 | 50.9% [44%–58%] | 62.4% | 0.205 |
| Flat | 46 | 63.0% [48%–76%] | 84.8% | 0.285 |
| Scripts (single level) | 48 | 45.8% [31%–60%] | 56.2% | 0.187 |
| Nested scripts | 9 | 11.1% [0%–33%] | 11.1% | 0.344 |
| Fractions | 62 | 51.6% [40%–65%] | 58.1% | 0.139 |

## Error buckets (all test expressions)

| missing symbols | extra symbols | wrong symbols | wrong structure |
|---:|---:|---:|---:|
| 145 | 252 | 237 | 42 |

Median latency per expression (CPU, batch 1): 41.1 ms
