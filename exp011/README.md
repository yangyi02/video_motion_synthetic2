## Same as Exp006 But Use Wider Model 

- Occlusion modeling, predict occlusion using neural nets instead of derived from motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels
- Neural nets get twice deeper

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 
- The gt model should report 100% reconstruction accuracy, however, the GtNet is not since it does not really use depth

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 100 | 100 | 0.01 | box, m_range=1 |
| 02 | 97 | 100 | 0.01 | mnist, m_range=1 |
| 03 | 99 | 100 | 0.02 | box, m_range=1, bg_move |
| 04 | 98 | 100 | 0.04 | mnist, m_range=1, bg_move |
| 05 | 87 | 100 | 0.06 | box, m_range=1, num_objects=2 |
| 06 | 90 | 100 | 0.04 | mnist, m_range=1, num_objects=2 |
| 07 | 99 | 100 | 0.04 | box, m_range=2 |
| 08 | 96 | 100 | 0.03 | mnist, m_range=2 |
| 09 | 99 | 100 | 0.07 | box, m_range=2, bg_move |
| 10 | 97 | 100 | 0.12 | mnist, m_range=2, bg_move |
| 11 | 87 | 100 | 0.12 | box, m_range=2, num_objects=2 |
| 12 | 88 | 100 | 0.11 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

