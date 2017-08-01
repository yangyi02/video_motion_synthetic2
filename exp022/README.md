## Nearby Flow Smoothness Loss based on Segmentation 

- Baseline: exp019
- Occlusion modeling, predict relative depth to help figure out occlusion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels
- Predict depth using only one image
- Predict occlusion using rule based depth and motion
- Add segmentation temporal consistency loss
- Add segmentation based flow smoothness loss
- Add multiple depth

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 |  | 100 |  | box, m_range=1 |
| 02 |  | 100 |  | mnist, m_range=1 |
| 03 |  | 100 |  | box, m_range=1, bg_move |
| 04 |  | 100 |  | mnist, m_range=1, bg_move |
| 05 |  | 99 | | box, m_range=1, num_objects=2 |
| 06 |  | 99 | | mnist, m_range=1, num_objects=2 |
| 07 |  | 100 |  | box, m_range=2 |
| 08 |  | 100 |  | mnist, m_range=2 |
| 09 |  | 100 |  | box, m_range=2, bg_move |
| 10 |  | 100 |  | mnist, m_range=2, bg_move |
| 11 |  | 98 | | box, m_range=2, num_objects=2 |
| 12 |  | 98 | | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |    |  |      | box, m_range=2, image_size=64, bg_move |
| 16 |    |  |      | mnist, m_range=2, image_size=64, bg_move |
| 17 |    |  |      | box, m_range=2, num_objects=2, image_size=64 |
| 18 |    |  |      | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |    |  |      | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

