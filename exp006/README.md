## Same as Exp005 But Divide Loss with Number of Existing Pixels

- Occlusion modeling, predict occlusion using neural nets instead of derived from motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels

### Results

| Exp | Test Loss Improve (%) | Gt Loss Improve (%) | Note |
| ------------- | ----------- | ----------- | ----------- | 
| 1 | 100 | 100 | box, m_range=1 |
| 2 | 93 | 100 | box, m_range=2 |
| 3 | 94 | 100 | mnist, m_range=2 |
| 4 | 96 | 100 | mnist, m_range=2, image_size=64 |
| 5 | 98 | 100 | box, m_range=2, image_size=64 | 
| 6 | 84 | 94 | box, m_range=2, num_objects=2 |
| 7 | 82 | 89 | mnist, m_range=2, num_objects=2 | 
| 8 |    | | box, m_range=2, bg_move |

### Take Home Message

- Does not work. No matter how you design loss, the final result on occlusion boundary is not good.
- We still need a better model on occlusion.
- However, compared to Exp005, it is happy to see the model never over-estimate occlusion in the background boundary.
