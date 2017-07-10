### Reconstruction Consider Occluded Disappear Pixels

- Occlusion modeling
- Predict motion for every pixel
- Photometric loss for every pixel


| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 | 80 | 80 | box, m_range=1 |
| 2 | 78 | 77 | box, m_range=2 |
| 3 | 82 | 84 | mnist, m_range=2 |
| 4 | 81 | 84 | mnist, m_range=2, image_size=64 |
| 5 | 81 | 81 | box, m_range=2, image_size=64 | 
| 6 | 66 | 60 | box, m_range=2, num_objects=2 |
| 7 | 65 | 64 | mnist, m_range=2, num_objects=2 | 
