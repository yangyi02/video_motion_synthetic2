## Model Relative Occlusion in Each Pixel Neighborhood

- Baseline: exp014 and exp006
- Occlusion modeling, predict relative depth to help figure out occlusion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 100 | 100 | 0.02 | box, m_range=1 |
| 02 | 96 | 100 | 0.04 | mnist, m_range=1 |
| 03 | 100 | 100 | 0.04 | box, m_range=1, bg_move |
| 04 | 98 | 100 | 0.03 | mnist, m_range=1, bg_move |
| 05 | 83 | 99 | 0.15 | box, m_range=1, num_objects=2 |
| 06 | 89 | 100 | 0.05 | mnist, m_range=1, num_objects=2 |
| 07 | 97 | 100 | 0.12 | box, m_range=2 |
| 08 | 96 | 100 | 0.12 | mnist, m_range=2 |
| 09 | 99 | 100 | 0.03 | box, m_range=2, bg_move |
| 10 | 98 | 100 | 0.06 | mnist, m_range=2, bg_move |
| 11 | 87 | 99 | 0.18 | box, m_range=2, num_objects=2 |
| 12 | 89 | 99 | 0.38 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |    |  |      | box, m_range=2, image_size=64, bg_move |
| 16 |    |  |      | mnist, m_range=2, image_size=64, bg_move |
| 17 |    |  |      | box, m_range=2, num_objects=2, image_size=64 |
| 18 |    |  |      | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |    |  |      | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

