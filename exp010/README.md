## Same as Exp008 But Use Wider Model 

- Occlusion modeling, predict conflict occlusion and appear using motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded and not appeared, then divided by the total number of existing pixels
- Number of channel in each layer increase from 64 to 128

### Results

| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 | | 100 | box, m_range=1 |
| 2 | | 100 | box, m_range=2 |
| 3 | | 100 | mnist, m_range=2 |
| 4 | | 100 | mnist, m_range=2, image_size=64 |
| 5 | | 100 | box, m_range=2, image_size=64 | 
| 6 | | 99 | box, m_range=2, num_objects=2 |
| 7 | | 98 | mnist, m_range=2, num_objects=2 | 
| 8 | |    | box, m_range=2, bg_move |

### Take Home Message

- Wider model does not help on handle occlusion
