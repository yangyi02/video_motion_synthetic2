### Same as Exp002 But Motion Kernel Separate as x and y

- Occlusion modeling
- Predict motion for every pixel
- Photometric loss for every pixel
- Motion decomposed to x and y direction


| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 | 81 | 80 | box, m_range=1 |
| 2 | 78 | 77 | box, m_range=2 |
| 3 | 82 | 84 | mnist, m_range=2 |
| 4 | 82 | 84 | mnist, m_range=2, image_size=64 |
| 5 | 81 | 81 | box, m_range=2, image_size=64 | 
| 6 | 66 | 58 | box, m_range=2, num_objects=2 |
| 7 | 63 | 65 | mnist, m_range=2, num_objects=2 | 

Take Home Message:

When motion range is small, the benefit for decomposing x and y is not evident 
