## Model Relative Occlusion in Each Pixel Neighborhood

- Baseline: exp018
- Occlusion modeling, predict relative depth to help figure out occlusion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels
- Predict depth using only one image
- Add segmentation temporal consistency loss

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 100 | 100 | 0.00 | box, m_range=1 |
| 02 | 96 | 100 | 0.05 | mnist, m_range=1 |
| 03 | 100 | 100 | 0.01 | box, m_range=1, bg_move |
| 04 | 98 | 100 | 0.06 | mnist, m_range=1, bg_move |
| 05 | 79 | 99 | 0.08 | box, m_range=1, num_objects=2 |
| 06 | 79 | 99 | 0.07 | mnist, m_range=1, num_objects=2 |
| 07 | 99 | 100 | 0.01 | box, m_range=2 |
| 08 | 92 | 100 | 0.08 | mnist, m_range=2 |
| 09 | 99 | 100 | 0.02 | box, m_range=2, bg_move |
| 10 | 97 | 100 | 0.06 | mnist, m_range=2, bg_move |
| 11 | 77 | 98 | 0.20 | box, m_range=2, num_objects=2 |
| 12 | 78 | 98 | 0.21 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |    |  |      | box, m_range=2, image_size=64, bg_move |
| 16 |    |  |      | mnist, m_range=2, image_size=64, bg_move |
| 17 |    |  |      | box, m_range=2, num_objects=2, image_size=64 |
| 18 |    |  |      | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |    |  |      | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

