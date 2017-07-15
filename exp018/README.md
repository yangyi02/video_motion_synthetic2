## Model Relative Occlusion in Each Pixel Neighborhood

- Baseline: exp015
- Occlusion modeling, predict relative depth to help figure out occlusion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels
- Predict depth using only one image

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 94 | 100 | 0.05 | box, m_range=1 |
| 02 | 96 | 100 | 0.04 | mnist, m_range=1 |
| 03 | 100 | 100 | 0.04 | box, m_range=1, bg_move |
| 04 | 98 | 100 | 0.02 | mnist, m_range=1, bg_move |
| 05 | 86 | 99 | 0.08 | box, m_range=1, num_objects=2 |
| 06 | 91 | 100 | 0.03 | mnist, m_range=1, num_objects=2 |
| 07 | 93 | 100 | 0.12 | box, m_range=2 |
| 08 | 96 | 100 | 0.16 | mnist, m_range=2 |
| 09 | 99 | 100 | 0.09 | box, m_range=2, bg_move |
| 10 | 98 | 100 | 0.06 | mnist, m_range=2, bg_move |
| 11 | 86 | 99 | 0.13 | box, m_range=2, num_objects=2 |
| 12 | 88 | 99 | 0.38 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |    |  |      | box, m_range=2, image_size=64, bg_move |
| 16 |    |  |      | mnist, m_range=2, image_size=64, bg_move |
| 17 |    |  |      | box, m_range=2, num_objects=2, image_size=64 |
| 18 |    |  |      | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |    |  |      | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

