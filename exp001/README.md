### The Simplest Version

- No occlusion modeling
- Predict motion for every pixel
- Photometric loss for every pixel


| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 | 82 | 61 | box, m_range=1 |
| 2 | 81 | 57 | box, m_range=2 |
| 3 | 83 | 68 | mnist, m_range=2 |
| 4 | 82 | 68 | mnist, m_range=2, image_size=64 |
| 5 | 81 | 63 | box, m_range=2, image_size=64 | 
| 6 | 68 | 37 | box, m_range=2, num_objects=2 |
| 7 | 66 | 51 | mnist, m_range=2, num_objects=2 | 
