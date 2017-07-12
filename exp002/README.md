## Reconstruction Consider Moving Pixels Occlude Static Pixels

- Occlusion modeling, moving pixels will occlude static pixels
- Predict motion for every pixel
- Photometric loss for every pixel

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 80 | 80 | 0.01 | box, m_range=1 |
| 02 | 84 | 85 | 0.01 | mnist, m_range=1 |
| 03 | 88 | 84 | 0.03 | box, m_range=1, bg_move |
| 04 | 87 | 84 | 0.06 | mnist, m_range=1, bg_move |
| 05 | 69 | 67 | 0.08 | box, m_range=1, num_objects=2 |
| 06 | 75 | 76 | 0.04 | mnist, m_range=1, num_objects=2 |
| 07 | 79 | 77 | 0.04 | box, m_range=2 |
| 08 | 82 | 83 | 0.03 | mnist, m_range=2 |
| 09 | 82 | 76 | 0.12 | box, m_range=2, bg_move |
| 10 | 81 | 75 | 0.19 | mnist, m_range=2, bg_move |
| 11 | 66 | 58 | 0.18 | box, m_range=2, num_objects=2 |
| 12 | 72 | 72 | 0.12 | mnist, m_range=2, num_objects=2 |
| 13 | 81 | 81 |  | box, m_range=2, image_size=64 |
| 14 | 81 | 84 |  | mnist, m_range=2, image_size=64 |
| 15 |  | 87 |  | box, m_range=2, image_size=64, bg_move |
| 16 |  | 90 |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  | 66 |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  | 81 |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  | 58 |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Although this is not a perfect model, but significantly than baseline exp001, even when background moves or 2 objects overlap.
- Visualization suggests this is a good baseline.
