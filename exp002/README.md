### The Simplest Version

- Occlusion modeling
- Predict motion for every pixel
- Photometric loss for every pixel


| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 | 80 | 80 | box, m_range=1 |
| 2 | 78 | 77 | box, m_range=2 |
| 3 | 83 | 84 | mnist, m_range=2 |
| 4 | 80 | 84 | mnist, m_range=2, image_size=64 |
| 5 | 81 | 81 | box, m_range=2, image_size=64 | 

