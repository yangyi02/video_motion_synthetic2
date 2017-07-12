## Same as Exp008 But Decompose Motion to x and y Direction

- Occlusion modeling, predict conflict occlusion and appear using motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded and not appeared, then divided by the total number of existing pixels
- Motion decomposed to x and y direction

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 93 | 100 | 0.12 | box, m_range=1 |
| 02 | 98 | 100 | 0.01 | mnist, m_range=1 |
| 03 | 100 | 100 | 0.01 | box, m_range=1, bg_move |
| 04 | 98 | 100 | 0.03 | mnist, m_range=1, bg_move |
| 05 | 89 | 99 | 0.06 | box, m_range=1, num_objects=2 |
| 06 | 92 | 100 | 0.04 | mnist, m_range=1, num_objects=2 |
| 07 | 100 | 100 | 0.01 | box, m_range=2 |
| 08 | 97 | 100 | 0.12 | mnist, m_range=2 |
| 09 | 99 | 100 | 0.04 | box, m_range=2, bg_move |
| 10 | 98 | 100 | 0.07 | mnist, m_range=2, bg_move |
| 11 | 87 | 99 | 0.14 | box, m_range=2, num_objects=2 |
| 12 | 92 | 99 | 0.83 | mnist, m_range=2, num_objects=2 |
| 13 |  | 100 |  | box, m_range=2, image_size=64 |
| 14 |  | 100 |  | mnist, m_range=2, image_size=64 |
| 15 |  | 100 |  | box, m_range=2, image_size=64, bg_move |
| 16 |  | 100 |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  | 100 |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  | 100 |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  | 99 |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Decomposing x and y for motion prediction does not reflect improvement on accuracy when motion range is only 2
