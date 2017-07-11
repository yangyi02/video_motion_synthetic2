### Same as Exp008 But Use Wider Model 

- Occlusion modeling
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded and not appeared, then divided by the total number of existing pixels
- Predict occlusion and appeara using motion


| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 | | 100 | box, m_range=1 |
| 2 | | 100 | box, m_range=2 |
| 3 | | 100 | mnist, m_range=2 |
| 4 | | 100 | mnist, m_range=2, image_size=64 |
| 5 | | 100 | box, m_range=2, image_size=64 | 
| 6 | | 99 | box, m_range=2, num_objects=2 |
| 7 | | 98 | mnist, m_range=2, num_objects=2 | 

Take Home Message:

Wider model does not help on handle occlusion