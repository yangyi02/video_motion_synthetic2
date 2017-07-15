## Model Relative Occlusion in Each Pixel Neighborhood

- Baseline: exp001
- Occlusion modeling, predict relative depth to help figure out occlusion
- Predict motion for every pixel
- Photometric loss for every pixel

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 82 | 80 | 0.04 | box, m_range=1 |
| 02 | 83 | 85 | 0.05 | mnist, m_range=1 |
| 03 | 89 | 89 | 0.03 | box, m_range=1, bg_move |
| 04 | 88 | 89 | 0.06 | mnist, m_range=1, bg_move |
| 05 | 71 | 68 | 0.13 | box, m_range=1, num_objects=2 |
| 06 | 75 | 80 | 0.09 | mnist, m_range=1, num_objects=2 |
| 07 | 82 | 77 | 0.09 | box, m_range=2 |
| 08 | 84 | 83 | 0.09 | mnist, m_range=2 |
| 09 | 83 | 83 | 0.14 | box, m_range=2, bg_move |
| 10 | 83 | 84 | 0.14 | mnist, m_range=2, bg_move |
| 11 | 69 | 63 | 0.27 | box, m_range=2, num_objects=2 |
| 12 | 73 | 78 | 0.20 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |    |  |      | box, m_range=2, image_size=64, bg_move |
| 16 |    |  |      | mnist, m_range=2, image_size=64, bg_move |
| 17 |    |  |      | box, m_range=2, num_objects=2, image_size=64 |
| 18 |    |  |      | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |    |  |      | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

