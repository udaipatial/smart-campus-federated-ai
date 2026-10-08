# Experimental Results: Centralized vs Federated Energy Anomaly Detection

| Metric                 | Centralized                 | Federated (FedAvg)            | Difference             |
|:-----------------------|:----------------------------|:------------------------------|:-----------------------|
| Accuracy               | 99.07 %                     | 97.69 %                       | -1.39 %                |
| Precision              | 100.0 %                     | 85.92 %                       | -14.08 %               |
| Recall                 | 88.73 %                     | 85.92 %                       | -2.82 %                |
| F1 Score               | 94.03 %                     | 85.92 %                       | -8.11 %                |
| Training Time          | 10.666 seconds              | 4.876 seconds                 | -5.79 seconds          |
| Inference Latency      | 0.003 ms/sample             | 0.0034 ms/sample              | 0.0003 ms/sample       |
| Communication Overhead | 0.455 MB                    | 1.111 MB                      | 0.655 MB               |
| Privacy Preservation   | None (Raw Data Transmitted) | High (Parameter Updates Only) | GDPR / FERPA Compliant |

*Note: Metrics represent empirical evaluation on real-time simulated non-IID smart campus edge nodes.*
