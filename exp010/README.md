## Same as Exp008 But Use Wider Model 

- Occlusion modeling, predict conflict occlusion and appear using motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded and not appeared, then divided by the total number of existing pixels
- Number of channel in each layer increase from 64 to 128

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 97 | 100 | 0.08 | box, m_range=1 |
| 02 | 98 | 100 | 0.01 | mnist, m_range=1 |
| 03 | 100 | 100 | 0.00 | box, m_range=1, bg_move |
| 04 | 99 | 100 | 0.02 | mnist, m_range=1, bg_move |
| 05 | 90 | 99 | 0.05 | box, m_range=1, num_objects=2 |
| 06 | 94 | 100 | 0.04 | mnist, m_range=1, num_objects=2 |
| 07 | 97 | 100 | 0.05 | box, m_range=2 |
| 08 | 95 | 100 | 0.22 | mnist, m_range=2 |
| 09 | 98 | 100 | 0.28 | box, m_range=2, bg_move |
| 10 | 99 | 100 | 0.05 | mnist, m_range=2, bg_move |
| 11 | 89 | 99 | 0.13 | box, m_range=2, num_objects=2 |
| 12 | 95 | 99 | 0.94 | mnist, m_range=2, num_objects=2 |
| 13 |  | 100 |  | box, m_range=2, image_size=64 |
| 14 |  | 100 |  | mnist, m_range=2, image_size=64 |
| 15 |  | 100 |  | box, m_range=2, image_size=64, bg_move |
| 16 |  | 100 |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  | 100 |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  | 100 |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  | 99 |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Wider model does not help on handle occlusion
