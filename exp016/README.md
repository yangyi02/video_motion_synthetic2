## Bidirectional Model

- Baseline: exp014 and exp015
- Occlusion modeling, predict relative depth to help figure out occlusion
- Predict motion for every pixel
- Photometric loss for every pixel
- Bidirectional Model

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 100 | 100 | 0.02 | box, m_range=1 |
| 02 | 96 | 99 | 0.04 | mnist, m_range=1 |
| 03 | 99 | 99 | 0.04 | box, m_range=1, bg_move |
| 04 | 98 | 99 | 0.06 | mnist, m_range=1, bg_move |
| 05 | 89 | 99 | 0.08 | box, m_range=1, num_objects=2 |
| 06 | 90 | 98 | 0.07 | mnist, m_range=1, num_objects=2 |
| 07 | 99 | 100 | 0.07 | box, m_range=2 |
| 08 | 95 | 98 | 0.11 | mnist, m_range=2 |
| 09 | 98 | 98 | 0.11 | box, m_range=2, bg_move |
| 10 | 96 | 98 | 0.22 | mnist, m_range=2, bg_move |
| 11 | 87 | 98 | 0.19 | box, m_range=2, num_objects=2 |
| 12 | 87 | 96 | 0.19 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- The bidirectional model obtains results almost same as exp015.
- Both exp016 and exp015 are slightly better than exp014, but still there exist problem.
