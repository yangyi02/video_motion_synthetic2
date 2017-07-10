### Same as Exp006 But Use Pixel Average for Occlusion Place 

- No Occlusion modeling
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels
- Predict occluded pixel values as the average between all coming pixels


| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 | 96 | 81 | box, m_range=1 |
| 2 | 94 | 79 | box, m_range=2 |
| 3 | 91 | 85 | mnist, m_range=2 |
| 4 | 84 | 86 | mnist, m_range=2, image_size=64 |
| 5 | 94 | 82 | box, m_range=2, image_size=64 | 
| 6 | 87 | 78 | box, m_range=2, num_objects=2 |
| 7 | 81 | 83 | mnist, m_range=2, num_objects=2 | 

Take Home Message:

Even worse at dealing occlusions, not recommend
