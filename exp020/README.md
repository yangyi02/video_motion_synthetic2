## Bidirectional Model

- Baseline: exp016 and exp019
- Occlusion modeling, predict relative depth to help figure out occlusion
- Predict motion for every pixel
- Photometric loss for every pixel

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 100 | 100 | 0.00 | box, m_range=1 |
| 02 | 96 | 99 | 0.04 | mnist, m_range=1 |
| 03 | 99 | 100 | 0.01 | box, m_range=1, bg_move |
| 04 | 97 | 99 | 0.03 | mnist, m_range=1, bg_move |
| 05 | 82 | 99 | 0.09 | box, m_range=1, num_objects=2 |
| 06 | 83 | 97 | 0.08 | mnist, m_range=1, num_objects=2 |
| 07 | 98 | 100 | 0.03 | box, m_range=2 |
| 08 | 94 | 98 | 0.07 | mnist, m_range=2 |
| 09 | 98 | 98 | 0.02 | box, m_range=2, bg_move |
| 10 | 96 | 98 | 0.06 | mnist, m_range=2, bg_move |
| 11 | 81 | 97 | 0.20 | box, m_range=2, num_objects=2 |
| 12 | 81 | 94 | 0.18 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message
