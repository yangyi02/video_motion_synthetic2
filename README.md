### Endpoint Error Results Summary

| Exp  | 01   | 02   | 03   | 04   | 05   | 06   | 07   | 08   | 09   | 10   | 11   | 12   | Comp | Eval | Note |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 001  | 0.07 | 0.07 | 0.08 | 0.09 | 0.15 | 0.14 | 0.11 | 0.13 | 0.18 | 0.22 | 0.26 | 0.26 |      | Good | The simplest baseline |
| 002  | 0.01 | 0.01 | 0.03 | 0.06 | 0.08 | 0.04 | 0.04 | 0.03 | 0.12 | 0.19 | 0.18 | 0.12 | 001  | Good | Moving pixel occlude static pixel |
| 003  | 0.00 | 0.01 | 0.02 | 0.05 | 0.05 | 0.03 | 0.27 | 0.33 | 1.68 | 1.74 | 0.15 | 0.33 | 002  | Bad  | New appear pixel not in loss |
| 004  | 0.01 | 0.01 | 0.03 | 0.06 | 0.09 | 0.05 | 0.03 | 0.03 | 0.13 | 0.17 | 0.17 | 0.10 | 002  | Good | Decompose x and y |
| 005-1| 0.04 | 0.06 | 0.06 | 0.06 | 0.15 | 0.11 | 0.10 | 0.11 | 0.17 | 0.17 | ---- | ---- | 002  | Bad  | Neural net predict disappear |
| 005  | 0.19 | 0.17 | 0.26 | 0.20 | 0.12 | 0.20 | 0.43 | 0.69 | 1.87 | 2.25 | 0.81 | 0.56 | 003  | Bad  | Neural net predict disappear |
| 006  | 0.02 | 0.02 | 0.02 | 0.04 | 0.06 | 0.05 | 0.08 | 0.03 | 0.06 | 0.10 | 0.15 | 0.12 | 005-1| Good | Old pixel loss divided by total number of old pixels | 
| 006-1| 0.00 | 0.01 | 0.02 | 0.05 | 0.05 | 0.03 | 0.01 | 0.02 | 0.10 | 0.19 | 0.09 | 0.08 | 002  | Good | Old pixel loss divided by total number of old pixels |
| 007  | 0.02 | 0.02 | 0.04 | 0.05 | 0.07 | 0.05 | 0.08 | 0.09 | 0.13 | 0.15 | 0.19 | 0.18 | 006  | Bad  | Use avearge value at occlusion |
| 008  | 0.01 | 0.01 | 0.00 | 0.03 | 0.07 | 0.04 | 0.03 | 0.09 | 0.05 | 0.06 | 0.12 | 0.97 | 006  | Bad  | New appear and occlude location both not in loss |
| 008-1| 0.01 | 0.01 | 0.01 | 0.03 | 0.05 | 0.03 | 0.03 | 0.04 | 0.12 | 0.05 | 0.12 | 0.13 | 008  | Good | Extra loss for total number of new and conflicting pixels |
| 009  | 0.12 | 0.01 | 0.01 | 0.03 | 0.06 | 0.04 | 0.01 | 0.12 | 0.04 | 0.07 | 0.14 | 0.83 | 008  | Bad  | Decompose x and y |
| 010  | 0.08 | 0.01 | 0.00 | 0.02 | 0.05 | 0.04 | 0.05 | 0.22 | 0.28 | 0.05 | 0.13 | 0.94 | 008  | Bad  | Wider network, proves exp008 is bad |
| 011  | 0.01 | 0.01 | 0.02 | 0.04 | 0.06 | 0.04 | 0.04 | 0.03 | 0.07 | 0.12 | 0.12 | 0.11 | 006  | Good | Wider network |
| 012  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 000  |      | |
| 013  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 000  |      | |
| 014  | 0.04 | 0.05 | 0.03 | 0.06 | 0.13 | 0.09 | 0.09 | 0.09 | 0.14 | 0.14 | 0.27 | 0.20 | 001  | Good | Predict relative depth |
| 015  | 0.02 | 0.03 | 0.03 | 0.03 | 0.15 | 0.05 | 0.11 | 0.12 | 0.04 | 0.06 | 0.17 | 0.38 | 014  | Good | Old pixel loss divided by total number of old pixels |
| 016  | 0.02 | 0.04 | 0.04 | 0.06 | 0.08 | 0.07 | 0.07 | 0.11 | 0.11 | 0.22 | 0.19 | 0.19 | 014  | Bad  | Bidirectional model |
| 017  | 0.04 | 0.03 | 0.04 | 0.04 | 0.12 | 0.08 | 0.09 | 0.09 | 0.04 | 0.20 | 0.28 | 0.43 | 015  | Bad  | Add a few more layers at the bottom of neural net |
| 018  | 0.05 | 0.04 | 0.04 | 0.02 | 0.08 | 0.03 | 0.12 | 0.16 | 0.09 | 0.06 | 0.13 | 0.38 | 015  | Bad  | Predict depth using only one image |
| 019  | 0.00 | 0.05 | 0.00 | 0.06 | 0.07 | 0.07 | 0.01 | 0.07 | 0.02 | 0.06 | 0.22 | 0.21 | 018  | Good | Add segmentation temporal consistency loss |
| 020  | 0.00 | 0.04 | 0.01 | 0.03 | 0.09 | 0.08 | 0.03 | 0.07 | 0.02 | 0.06 | 0.20 | 0.18 | 019  | Good | Bidirectional model |
| 000  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 000  |      | |

| Exp  | 01   | 02   | 03   | 04   | 05   | 06   | 07   | 08   | 09   | 10   | 11   | 12   | Comp | Eval | Note |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 001  | 0.07 | 0.07 | 0.08 | 0.09 | 0.15 | 0.14 | 0.11 | 0.13 | 0.18 | 0.22 | 0.26 | 0.26 |      | Good | The simplest baseline |
| 002  | 0.01 | 0.01 | 0.03 | 0.06 | 0.08 | 0.04 | 0.04 | 0.03 | 0.12 | 0.19 | 0.18 | 0.12 | 001  | Good | Moving pixel occlude static pixel |
| 006-1| 0.00 | 0.01 | 0.02 | 0.05 | 0.05 | 0.03 | 0.01 | 0.02 | 0.10 | 0.19 | 0.09 | 0.08 | 002  | Good | Old pixel loss divided by total number of old pixels |
| 006  | 0.02 | 0.02 | 0.02 | 0.04 | 0.06 | 0.05 | 0.08 | 0.03 | 0.06 | 0.10 | 0.15 | 0.12 | 005-1| Good | Old pixel loss divided by total number of old pixels | 
| 011  | 0.01 | 0.01 | 0.02 | 0.04 | 0.06 | 0.04 | 0.04 | 0.03 | 0.07 | 0.12 | 0.12 | 0.11 | 006  | Good | Wider network |
| 015  | 0.02 | 0.03 | 0.03 | 0.03 | 0.15 | 0.05 | 0.11 | 0.12 | 0.04 | 0.06 | 0.17 | 0.38 | 014  | Good | Old pixel loss divided by total number of old pixels |
| 008-1| 0.01 | 0.01 | 0.01 | 0.03 | 0.05 | 0.03 | 0.03 | 0.04 | 0.12 | 0.05 | 0.12 | 0.13 | 008  | Good | Extra loss for total number of new and conflicting pixels |


### Take Home Message

- We should use: new appear pixel not in loss, loss divided by total number of old pixels
