## Same as Exp006 But Conflict Region and Appear Region Both no Loss

- Occlusion modeling, predict conflict occlusion and appear using motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded and not appeared, then divided by the total number of existing pixels

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 100 | 100 | 0.01 | box, m_range=1 |
| 02 | 98 | 100 | 0.01 | mnist, m_range=1 |
| 03 | 100 | 100 | 0.00 | box, m_range=1, bg_move |
| 04 | 99 | 100 | 0.03 | mnist, m_range=1, bg_move |
| 05 | 88 | 99 | 0.07 | box, m_range=1, num_objects=2 |
| 06 | 93 | 100 | 0.04 | mnist, m_range=1, num_objects=2 |
| 07 | 99 | 100 | 0.03 | box, m_range=2 |
| 08 | 97 | 100 | 0.09 | mnist, m_range=2 |
| 09 | 99 | 100 | 0.05 | box, m_range=2, bg_move |
| 10 | 98 | 100 | 0.06 | mnist, m_range=2, bg_move |
| 11 | 88 | 99 | 0.12 | box, m_range=2, num_objects=2 |
| 12 | 94 | 99 | 0.97 | mnist, m_range=2, num_objects=2 |
| 13 |  | 100 |  | box, m_range=2, image_size=64 |
| 14 |  | 100 |  | mnist, m_range=2, image_size=64 |
| 15 |  | 100 |  | box, m_range=2, image_size=64, bg_move |
| 16 |  | 100 |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  | 99 |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  | 100 |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  | 99 |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Results on Mnist overlap dataset suggest this is not a good model. 
- Perhaps we still need to reconsider bidirectional model or even deal with occlusion now.
