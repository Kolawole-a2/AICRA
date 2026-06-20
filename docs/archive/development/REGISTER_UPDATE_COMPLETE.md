# Risk Register Update Complete

**Date:** 2025-12-10  
**Status:** ✅ All registers updated to use $5,000,000 impact

---

## Summary

All risk register files have been successfully updated to use **$5,000,000** as the impact value for Expected Loss calculation.

### Updated Files

1. ✅ **register/risk_register_main.csv** - Updated (10,000 records)
2. ✅ **register/risk_register_main.json** - Updated
3. ✅ **register/risk_register_full.csv** - Updated (20,002 records)
4. ✅ **register/risk_register_full.json** - Updated
5. ✅ **register/risk_register_small_ember.csv** - Updated (2,000 records)
6. ✅ **register/risk_register_small_ember.json** - Updated
7. ✅ **policies/policy.json** - Updated (`impact_default: 5000000.0`)

### Expected Loss Formula

All registers now use:
```
Expected Loss = susceptibility × $5,000,000
```

Where:
- `susceptibility` = calibrated probability p(ransomware) ∈ [0, 1]
- Impact = **$5,000,000** (banking ransomware breach cost)

### Verification

All three register files verified:
- ✅ `risk_register_main.csv`: Impact = $5,000,000
- ✅ `risk_register_full.csv`: Impact = $5,000,000
- ✅ `risk_register_small_ember.csv`: Impact = $5,000,000

### Additional Updates

- ✅ `prescriptive_controls` column added to all registers (from `risk_bucket_controls.yaml`)
- ✅ Policy JSON updated with correct impact value
- ✅ Backup of original policy.json created

--**Update Complete** ✅

