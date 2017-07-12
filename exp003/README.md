## Same Model as Exp002 But Loss Consider New Appear Pixels

- Occlusion modeling, moving pixels will occlude static pixels
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- |
| 01 |  |  |  | box, m_range=1 |
| 02 |  |  |  | mnist, m_range=1 |
| 03 |  |  |  | box, m_range=1, bg_move |
| 04 |  |  |  | mnist, m_range=1, bg_move |
| 05 |  |  |  | box, m_range=1, num_objects=2 |
| 06 |  |  |  | mnist, m_range=1, num_objects=2 |
| 07 |  |  |  | box, m_range=2 |
| 08 |  |  |  | mnist, m_range=2 |
| 09 |  |  |  | box, m_range=2, bg_move |
| 10 |  |  |  | mnist, m_range=2, bg_move |
| 11 |  |  |  | box, m_range=2, num_objects=2 |
| 12 |  |  |  | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 | 100 | 100 | box, m_range=1 |
| 2 | 92 | 100 | box, m_range=2 |
| 3 | 98 | 100 | mnist, m_range=2 |
| 4 | 88 | 100 | mnist, m_range=2, image_size=64 |
| 5 | 99 | 100 | box, m_range=2, image_size=64 | 
| 6 | 88 | 94 | box, m_range=2, num_objects=2 |
| 7 | 86 | 90 | mnist, m_range=2, num_objects=2 |  
| 8 |    |    | box, m_range=2, bg_move |

### Take Home Message

- Although this significantly improves test loss, the optical flow estimation actually becomes slightly worse. 
