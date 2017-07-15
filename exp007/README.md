## Same as Exp006 But Use Pixel Average for Occlusion Place 

- Baseline: Exp006
- No Occlusion modeling, predict conflict pixel values as the average between all coming pixels
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 94 | 81 | 0.02 | box, m_range=1 |
| 02 | 93 | 91 | 0.02 | mnist, m_range=1 |
| 03 | 99 | 96 | 0.04 | box, m_range=1, bg_move |
| 04 | 97 | 96 | 0.05 | mnist, m_range=1, bg_move |
| 05 | 87 | 80 | 0.07 | box, m_range=1, num_objects=2 |
| 06 | 90 | 90 | 0.05 | mnist, m_range=1, num_objects=2 |
| 07 | 93 | 80 | 0.08 | box, m_range=2 |
| 08 | 91 | 86 | 0.09 | mnist, m_range=2 |
| 09 | 98 | 94 | 0.13 | box, m_range=2, bg_move |
| 10 | 96 | 93 | 0.15 | mnist, m_range=2, bg_move |
| 11 | 87 | 78 | 0.19 | box, m_range=2, num_objects=2 |
| 12 | 87 | 85 | 0.18 | mnist, m_range=2, num_objects=2 |
| 13 | 94 | 82 |  | box, m_range=2, image_size=64 |
| 14 | 84 | 86 |  | mnist, m_range=2, image_size=64 |
| 15 |  | 97 |  | box, m_range=2, image_size=64, bg_move |
| 16 |  | 98 |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  | 81 |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  | 86 |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  | 80 |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Even worse at dealing occlusions, not recommend
