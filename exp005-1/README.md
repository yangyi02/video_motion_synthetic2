## Same as Exp002 But Using Neural Nets to Predict Occlusion

- Baseline: Exp002
- Occlusion modeling, predict occlusion using neural nets instead of derived from motion
- Predict motion for every pixel
- Photometric loss for every pixel

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 82 | 80 | 0.04 | box, m_range=1 |
| 02 | 83 | 85 | 0.06 | mnist, m_range=1 |
| 03 | 88 | 85 | 0.06 | box, m_range=1, bg_move |
| 04 | 87 | 85 | 0.06 | mnist, m_range=1, bg_move |
| 05 | 70 | 66 | 0.15 | box, m_range=1, num_objects=2 |
| 06 | 75 | 76 | 0.11 | mnist, m_range=1, num_objects=2 |
| 07 | 80 | 77 | 0.10 | box, m_range=2 |
| 08 | 83 | 83 | 0.11 | mnist, m_range=2 |
| 09 | 82 | 76 | 0.17 | box, m_range=2, bg_move |
| 10 | 83 | 75 | 0.17 | mnist, m_range=2, bg_move |
| 11 | | 59 | | box, m_range=2, num_objects=2 |
| 12 | | 72 | | mnist, m_range=2, num_objects=2 |
| 13 | |  |  | box, m_range=2, image_size=64 |
| 14 | |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

