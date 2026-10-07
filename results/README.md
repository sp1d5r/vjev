# Results

All numbers are on our test split (1,736 clips, `sha256(Id) % 100 < 5`).
They are not comparable to the VQualA 2025 leaderboard, whose test labels are withheld.

## Zero-shot Clef-flash (no training)

| Input | SROCC | PLCC | Final | Brier | ECE | Forward p50 |
|---|---|---|---|---|---|---|
| video + text | 0.128 | 0.127 | 0.127 | 0.912 | 0.064 | 188 ms |
| text only | 0.175 | 0.168 | 0.172 | 1.037 | 0.278 | 151 ms |
| video only | 0.142 | 0.139 | 0.140 | 0.910 | 0.051 | 202 ms |

A uniform guess over 10 levels has Brier 0.900.
The video runs stay close to uniform, so their low ECE means hedging, not skill.
The standard error of SROCC at n=1,736 is about 0.024, so the gaps between rows are small.

Raw numbers: `zero_shot_clef_flash_test.json`. W&B project: `sp1d5r/vjev`.
