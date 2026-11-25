#!/bin/bash
# Compare regenerated ~/t15r/trials against archived ~/tlv_t15/trials (byte-level).
same=0; diffn=0
cd ~/t15r/trials || exit 1
for f in *.npy; do
  if cmp -s "$f" ~/tlv_t15/trials/"$f"; then
    same=$((same+1))
  else
    diffn=$((diffn+1)); echo "DIFF $f"
  fi
done
echo "identical=$same differing=$diffn total=$((same+diffn))"
