#!/usr/bin/bash -l

bgn_date_avlb="20160104"
bgn_date_fac="20170103"
bgn_date_opt="20171009"
bgn_date="20180102"
end_date="20251031"

python main.py --bgn $bgn_date_opt --end $end_date optimize
echo "$(date +'%Y-%m-%d %H:%M:%S') weights for factors portfolios optimized"

python main.py --bgn $bgn_date --end $end_date signals --type stg
echo "$(date +'%Y-%m-%d %H:%M:%S') signals optimized"

python main.py --bgn $bgn_date --end $end_date simulations --type stg --omit
echo "$(date +'%Y-%m-%d %H:%M:%S') complex simualations done"
