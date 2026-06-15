# Batched Solver Latency and QPS Throughput Profiling

**Device Profiled:** cpu

## Throughput and Latency Matrix

### Expert Registry Size $K = 4$

| Batch Size $B$ | Batch Latency ($\mu$s) | Latency per Query ($\mu$s) | Throughput (QPS) |
|---|---|---|---|
|   1 |     9.96 |               9.96 |       100417.7 |
|   4 |     9.71 |               2.43 |       411867.4 |
|  16 |     9.91 |               0.62 |      1615124.3 |
|  64 |    10.39 |               0.16 |      6157786.5 |
| 256 |    11.81 |               0.05 |     21682822.5 |

### Expert Registry Size $K = 16$

| Batch Size $B$ | Batch Latency ($\mu$s) | Latency per Query ($\mu$s) | Throughput (QPS) |
|---|---|---|---|
|   1 |    10.11 |              10.11 |        98927.0 |
|   4 |    10.14 |               2.54 |       394339.4 |
|  16 |    10.37 |               0.65 |      1543322.1 |
|  64 |    11.62 |               0.18 |      5508056.2 |
| 256 |    25.80 |               0.10 |      9923346.4 |

### Expert Registry Size $K = 64$

| Batch Size $B$ | Batch Latency ($\mu$s) | Latency per Query ($\mu$s) | Throughput (QPS) |
|---|---|---|---|
|   1 |    11.51 |              11.51 |        86880.1 |
|   4 |    14.43 |               3.61 |       277236.4 |
|  16 |    22.42 |               1.40 |       713539.7 |
|  64 |    29.72 |               0.46 |      2153478.7 |
| 256 |    57.69 |               0.23 |      4437850.2 |

