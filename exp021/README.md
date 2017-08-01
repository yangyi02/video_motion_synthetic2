## Nearby Flow Smoothness Loss based on Segmentation 

- Baseline: exp019
- Occlusion modeling, predict relative depth to help figure out occlusion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels
- Predict depth using only one image
- Predict occlusion using rule based depth and motion
- Add segmentation temporal consistency loss
- Add segmentation based flow smoothness loss

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 98 | 100 | 0.00 | box, m_range=1 |
| 02 | 94 | 100 | 0.01 | mnist, m_range=1 |
| 03 | 100 | 100 | 0.00 | box, m_range=1, bg_move |
| 04 | 97 | 100 | 0.02 | mnist, m_range=1, bg_move |
| 05 | 80 | 99 | 0.06 | box, m_range=1, num_objects=2 |
| 06 | 83 | 99 | 0.03 | mnist, m_range=1, num_objects=2 |
| 07 | 97 | 100 | 0.02 | box, m_range=2 |
| 08 | 88 | 100 | 0.03 | mnist, m_range=2 |
| 09 | 99 | 100 | 0.01 | box, m_range=2, bg_move |
| 10 | 97 | 100 | 0.05 | mnist, m_range=2, bg_move |
| 11 | 65 | 98 | 0.22 | box, m_range=2, num_objects=2 |
| 12 | 68 | 98 | 0.17 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |    |  |      | box, m_range=2, image_size=64, bg_move |
| 16 |    |  |      | mnist, m_range=2, image_size=64, bg_move |
| 17 |    |  |      | box, m_range=2, num_objects=2, image_size=64 |
| 18 |    |  |      | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |    |  |      | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

