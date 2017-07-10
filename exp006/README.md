### Same as Exp005 But Divide Loss with Number of Existing Pixels

- Occlusion modeling
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels
- Predict occlusion using neural nets instead of motion


| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 | | 100 | box, m_range=1 |
| 2 | | 100 | box, m_range=2 |
| 3 | | 100 | mnist, m_range=2 |
| 4 | | 100 | mnist, m_range=2, image_size=64 |
| 5 | | 100 | box, m_range=2, image_size=64 | 
| 6 | | 94 | box, m_range=2, num_objects=2 |
| 7 | | 89 | mnist, m_range=2, num_objects=2 | 

Take Home Message:

