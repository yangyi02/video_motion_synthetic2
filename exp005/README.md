## Same as Exp003 But Using Neural Nets to Predict Occlusion

- Occlusion modeling, predict occlusion using neural nets instead of derived from motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 99 | 100 | 0.19 | box, m_range=1 |
| 02 | 95 | 100 | 0.17 | mnist, m_range=1 |
| 03 | 99 | 96 | 0.26 | box, m_range=1, bg_move |
| 04 | 98 | 96 | 0.20 | mnist, m_range=1, bg_move |
| 05 | 85 | 97 | 0.12 | box, m_range=1, num_objects=2 |
| 06 | 89 | 96 | 0.20 | mnist, m_range=1, num_objects=2 |
| 07 | 95 | 100 | 0.43 | box, m_range=2 |
| 08 | 96 | 100 | 0.69 | mnist, m_range=2 |
| 09 | 96 | 93 | 1.87 | box, m_range=2, bg_move |
| 10 | 92 | 91 | 2.25 | mnist, m_range=2, bg_move |
| 11 | 82 | 94 | 0.81 | box, m_range=2, num_objects=2 |
| 12 | 87 | 94 | 0.56 | mnist, m_range=2, num_objects=2 |
| 13 | 91 | 100 |  | box, m_range=2, image_size=64 |
| 14 | 96 | 100 |  | mnist, m_range=2, image_size=64 |
| 15 |  | 96 |  | box, m_range=2, image_size=64, bg_move |
| 16 |  | 97 |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  | 95 |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  | 99 |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  | 95 |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Does not work. 
- The neural network prefer to predict more places as occluded to decrease loss.
- The motion estimation accuracy is significantly worse. 
