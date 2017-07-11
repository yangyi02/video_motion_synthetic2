## Same as Exp008 But Decompose Motion to x and y Direction

- Occlusion modeling, predict conflict occlusion and appear using motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded and not appeared, then divided by the total number of existing pixels
- Motion decomposed to x and y direction

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
| 9 | | 99 | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Decomposing x and y for motion prediction does not reflect improvement on accuracy when motion range is only 2
