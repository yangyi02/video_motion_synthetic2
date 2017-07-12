## Same Model as Exp002 But Loss Consider New Appear Pixels

- Occlusion modeling, moving pixels will occlude static pixels
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- |
| 01 | 100 | 100 | 0.00 | box, m_range=1 |
| 02 | 98 | 100 | 0.01 | mnist, m_range=1 |
| 03 | 100 | 96 | 0.02 | box, m_range=1, bg_move |
| 04 | 98 | 96 | 0.05 | mnist, m_range=1, bg_move |
| 05 | 89 | 97 | 0.05 | box, m_range=1, num_objects=2 |
| 06 | 91 | 96 | 0.03 | mnist, m_range=1, num_objects=2 |
| 07 | 94 | 100 | 0.27 | box, m_range=2 |
| 08 | 98 | 100 | 0.33 | mnist, m_range=2 |
| 09 | 86 | 93 | 1.68 | box, m_range=2, bg_move |
| 10 | 87 | 91 | 1.74 | mnist, m_range=2, bg_move |
| 11 | 88 | 94 | 0.15 | box, m_range=2, num_objects=2 |
| 12 | 90 | 93 | 0.33 | mnist, m_range=2, num_objects=2 |
| 13 | 99 | 100 |  | box, m_range=2, image_size=64 |
| 14 | 88 | 100 |  | mnist, m_range=2, image_size=64 |
| 15 |  | 96 |  | box, m_range=2, image_size=64, bg_move |
| 16 |  | 97 |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  | 95 |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  | 98 |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  | 95 |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Although this significantly improves test loss, the optical flow estimation actually becomes much worse when motion range is larger than 1.
- It is interesting to see when motion range is limited to 1, this helps on motion prediction.
- However, once motion range is increased to 2, the model significantly degenerate to a very bad local optimal.
