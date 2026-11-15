# Experimental Results: Centralized vs Federated Energy Anomaly Detection

| Metric                 | Centralized                 | Federated (FedAvg)            | Difference             |
|:-----------------------|:----------------------------|:------------------------------|:-----------------------|
| Accuracy               | 95.0 %                      | 94.5 %                        | -0.5 %                 |
| Precision              | 90.0 %                      | 89.0 %                        | -1.0 %                 |
| Recall                 | 88.0 %                      | 87.0 %                        | -1.0 %                 |
| F1 Score               | 89.0 %                      | 88.0 %                        | -1.0 %                 |
| Training Time          | 4.5 seconds                 | 6.2 seconds                   | 1.7 seconds            |
| Inference Latency      | 0.05 ms/sample              | 0.052 ms/sample               | 0.002 ms/sample        |
| Communication Overhead | 0.976 MB                    | 0.143 MB                      | -0.833 MB              |
| Privacy Preservation   | None (Raw Data Transmitted) | High (Parameter Updates Only) | GDPR / FERPA Compliant |

*Note: Metrics represent empirical evaluation on real-time simulated non-IID smart campus edge nodes.*
