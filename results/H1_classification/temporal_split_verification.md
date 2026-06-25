# H1 Time-Ordered Split Verification

Generated: `2026-06-25T22:32:40.970775+00:00`
Data source: `C:\Users\KLAMOS\Desktop\My Cursor Projects\AICRA\data\ember2024`
Loader: `aicra.core.data.load_ember_2024(time_ordered=True)`

## Sample counts

| Set | Count |
|-----|------:|
| Pool (train + test JSONL) | 50,005 |
| Training (time-ordered) | 40,004 |
| Testing (time-ordered) | 10,001 |

## Training timestamps

- **Earliest:** `2023-09-24T00:00:10`
- **Latest:** `2023-10-06T23:59:41`

## Testing timestamps

- **Earliest:** `2023-10-08T00:00:51`
- **Latest:** `2023-10-14T12:57:02`

## Temporal integrity

| Check | Result |
|-------|--------|
| `max(train_timestamp) < min(test_timestamp)` | **True** |
| `max(train_timestamp) <= min(test_timestamp)` | True |
| Any train row with `timestamp > min(test)` | False |
| Any test row with `timestamp < max(train)` | False |

## Boundary detail

- Boundary timestamp: `2023-10-06T23:59:41`
- Train rows at boundary: 1
- Test rows at boundary: 0
- Same sample duplicated in both sets at boundary: **False**

## Interpretation

Strict temporal gap holds (max(train) < min(test)).
