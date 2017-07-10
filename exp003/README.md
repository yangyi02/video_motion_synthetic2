### Same Model as Exp002 But Loss Consider New Appear Pixels

- Occlusion modeling
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded


| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 | 100 | 100 | box, m_range=1 |
| 2 | 92 | 100 | box, m_range=2 |
| 3 | 98 | 100 | mnist, m_range=2 |
| 4 | 88 | 100 | mnist, m_range=2, image_size=64 |
| 5 | 99 | 100 | box, m_range=2, image_size=64 | 
| 6 | 88 | 94 | box, m_range=2, num_objects=2 |
| 7 | 86 | 90 | mnist, m_range=2, num_objects=2 |  

Take Home Message:

Although this significantly improves test loss, the optical flow estimation actually becomes slightly worse. 
