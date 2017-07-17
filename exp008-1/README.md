## Same as Exp008 But Add Extra Penalty Loss 

- Occlusion modeling, predict conflict occlusion and appear using motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded and not appeared, then divided by the total number of existing pixels
- Extra penalty loss for total number of conflicting and appearing pixels

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 94 | 95 | 0.01 | box, m_range=1 |
| 02 | 94 | 96 | 0.01 | mnist, m_range=1 |
| 03 | 98 | 98 | 0.01 | box, m_range=1, bg_move |
| 04 | 96 | 98 | 0.03 | mnist, m_range=1, bg_move |
| 05 | 83 | 94 | 0.05 | box, m_range=1, num_objects=2 |
| 06 | 90 | 96 | 0.03 | mnist, m_range=1, num_objects=2 |
| 07 | 93 | 94 | 0.03 | box, m_range=2 |
| 08 | 92 | 96 | 0.04 | mnist, m_range=2 |
| 09 | 95 | 97 | 0.12 | box, m_range=2, bg_move |
| 10 | 95 | 97 | 0.05 | mnist, m_range=2, bg_move |
| 11 | 83 | 93 | 0.12 | box, m_range=2, num_objects=2 |
| 12 | 87 | 95 | 0.13 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Results on Mnist overlap dataset suggest this is not a good model. 
- Perhaps we still need to reconsider bidirectional model or even deal with occlusion now.
