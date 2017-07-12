## Same as Exp002 But Motion Kernel Separate as x and y

- Occlusion modeling, moving pixels will occlude static pixels
- Predict motion for every pixel
- Photometric loss for every pixel
- Motion decomposed to x and y direction

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 80 | 80 | 0.01 | box, m_range=1 |
| 02 | 83 | 85 | 0.01 | mnist, m_range=1 |
| 03 | 87 | 85 | 0.03 | box, m_range=1, bg_move |
| 04 | 86 | 84 | 0.06 | mnist, m_range=1, bg_move |
| 05 | 68 | 67 | 0.09 | box, m_range=1, num_objects=2 |
| 06 | 74 | 76 | 0.05 | mnist, m_range=1, num_objects=2 |
| 07 | 78 | 77 | 0.03 | box, m_range=2 |
| 08 | 82 | 83 | 0.03 | mnist, m_range=2 |
| 09 | 82 | 77 | 0.13 | box, m_range=2, bg_move |
| 10 | 81 | 75 | 0.17 | mnist, m_range=2, bg_move |
| 11 | 66 | 59 | 0.17 | box, m_range=2, num_objects=2 |
| 12 | 71 | 72 | 0.10 | mnist, m_range=2, num_objects=2 |
| 13 | 81 | 81 |  | box, m_range=2, image_size=64 |
| 14 | 82 | 83 |  | mnist, m_range=2, image_size=64 |
| 15 |  | 87 |  | box, m_range=2, image_size=64, bg_move |
| 16 |  | 90 |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  | 66 |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  | 81 |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  | 59 |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- When motion range is small, the benefit for decomposing x and y is not evident 
