## Same as Exp005-1 But Predict occlusion using neural nets 

- Occlusion modeling, predict occlusion using neural nets instead of derived from motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 
- The gt model should report 100% reconstruction accuracy, however, the GtNet is not since it does not really use depth

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 99 | 100 | 0.02 | box, m_range=1 |
| 02 | 95 | 100 | 0.02 | mnist, m_range=1 |
| 03 | 99 | 100 | 0.02 | box, m_range=1, bg_move |
| 04 | 97 | 100 | 0.04 | mnist, m_range=1, bg_move |
| 05 | 85 | 100 | 0.06 | box, m_range=1, num_objects=2 |
| 06 | 88 | 100 | 0.05 | mnist, m_range=1, num_objects=2 |
| 07 | 97 | 100 | 0.08 | box, m_range=2 |
| 08 | 96 | 100 | 0.03 | mnist, m_range=2 |
| 09 | 98 | 100 | 0.06 | box, m_range=2, bg_move |
| 10 | 96 | 100 | 0.10 | mnist, m_range=2, bg_move |
| 11 | 82 | 100 | 0.15 | box, m_range=2, num_objects=2 |
| 12 | 85 | 100 | 0.12 | mnist, m_range=2, num_objects=2 |
| 13 | 98 | 100 |  | box, m_range=2, image_size=64 |
| 14 | 96 | 100 |  | mnist, m_range=2, image_size=64 |
| 15 |  | 96 |  | box, m_range=2, image_size=64, bg_move |
| 16 |  | 97 |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  | 96 |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  | 99 |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  | 95 |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Significantly Better than exp005-1.
- But we still need a better model on occlusion.
- However, compared to Exp005 and Exp005-1, it is happy to see the model never over-estimate occlusion in the background boundary.
