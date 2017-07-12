## Reconstruction Consider Moving Pixels Occlude Static Pixels

- Occlusion modeling, moving pixels will occlude static pixels
- Predict motion for every pixel
- Photometric loss for every pixel

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 80 | 80 |  | box, m_range=1 |
| 02 |  | 85 |  | mnist, m_range=1 |
| 03 |  | 84 |  | box, m_range=1, bg_move |
| 04 |  | 84 |  | mnist, m_range=1, bg_move |
| 05 |  | 67 |  | box, m_range=1, num_objects=2 |
| 06 |  | 76 |  | mnist, m_range=1, num_objects=2 |
| 07 | 78 | 77 |  | box, m_range=2 |
| 08 | 82 | 83 |  | mnist, m_range=2 |
| 09 |  | 76 |  | box, m_range=2, bg_move |
| 10 |  | 75 |  | mnist, m_range=2, bg_move |
| 11 | 66 | 58 |  | box, m_range=2, num_objects=2 |
| 12 |  | 72 |  | mnist, m_range=2, num_objects=2 |
| 13 | 81 | 81 |  | box, m_range=2, image_size=64 |
| 14 | 81 | 84 |  | mnist, m_range=2, image_size=64 |
| 15 |  | 87 |  | box, m_range=2, image_size=64, bg_move |
| 16 |  | 90 |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  | 66 |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  | 81 |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  | 58 |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

