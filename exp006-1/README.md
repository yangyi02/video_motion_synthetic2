## Reconstruction Consider Moving Pixels Occlude Static Pixels

- Baseline: Exp002
- Occlusion modeling, moving pixels will occlude static pixels
- Predict motion for every pixel
- Photometric loss for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 100 | 100 | 0.00 | box, m_range=1 |
| 02 | 97 | 100 | 0.01 | mnist, m_range=1 |
| 03 | 100 | 96 | 0.02 | box, m_range=1, bg_move |
| 04 | 97 | 95 | 0.05 | mnist, m_range=1, bg_move |
| 05 | 89 | 97 | 0.05 | box, m_range=1, num_objects=2 |
| 06 | 91 | 95 | 0.03 | mnist, m_range=1, num_objects=2 |
| 07 | 99 | 100 | 0.01 | box, m_range=2 |
| 08 | 97 | 100 | 0.02 | mnist, m_range=2 |
| 09 | 99 | 92 | 0.10 | box, m_range=2, bg_move |
| 10 | 97 | 90 | 0.19 | mnist, m_range=2, bg_move |
| 11 | 87 | 94 | 0.09 | box, m_range=2, num_objects=2 |
| 12 | 89 | 93 | 0.08 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

