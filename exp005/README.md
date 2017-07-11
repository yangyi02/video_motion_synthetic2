## Same as Exp003 But Using Neural Nets to Predict Occlusion

- Occlusion modeling, predict occlusion using neural nets instead of derived from motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded

### Results

| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 | 100 | 100 | box, m_range=1 |
| 2 | 94 | 100 | box, m_range=2 |
| 3 | 96 | 100 | mnist, m_range=2 |
| 4 | 96 | 100 | mnist, m_range=2, image_size=64 |
| 5 | 91 | 100 | box, m_range=2, image_size=64 | 
| 6 | 82 | 94 | box, m_range=2, num_objects=2 |
| 7 | 82 | 90 | mnist, m_range=2, num_objects=2 | 
| 8 |    | | box, m_range=2, bg_move |

### Take Home Message

- Does not work very good. 
- The neural network prefer to predict more places as occluded to decrease loss.
