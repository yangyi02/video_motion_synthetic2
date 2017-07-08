### The Simplest Version

- Occlusion modeling
- Predict motion for every pixel
- Photometric loss for every pixel


| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 |  | 80 | box, m_range=1 |
| 2 |  | 77 | box, m_range=2 |
| 3 |  | 84 | mnist, m_range=2 |
| 4 |  | 84 | mnist, m_range=2, image_size=64 |
| 5 |  | 81 | box, m_range=2, image_size=64 | 

