#!/usr/bin/bash -l

echo "$(date +'%Y-%m-%d %H:%M:%S') begins"
pid=$(python -c $'import yaml\nwith open("config.yaml", "r") as f:_config = yaml.safe_load(f)\nprint(_config["project_id"])')
vid=$(python -c $'import yaml\nwith open("config.yaml", "r") as f:_config = yaml.safe_load(f)\nprint(_config["version_id"])')
udb=$(python -c $'import yaml\nwith open("config.yaml", "r") as f:_config = yaml.safe_load(f)\nprint(_config["dbs"]["user"])')
echo "project_id=$pid, version_id=$vid"

if [ "$#" -eq 1 ]; then
    if [ "$1" = "--auto" ]; then
        end_date=$(date +"%Y%m%d")
    else
        end_date="$1"
    fi
else
    read -p "Please input the end date, format = [YYYYMMDD]:" end_date
fi
echo "end_date = $end_date"

rm_tqdb $udb --table "$pid"_tbl_avlb_"$vid"
rm_tqdb $udb --table "$pid"_tbl_icov_"$vid"
rm_tqdb $udb --table "$pid"_tbl_css_"$vid"
rm_tqdb $udb --table "$pid"_tbl_srets_"$vid"
# rm_tqdb $udb --table "$pid"_tbl_fac_raw_"$vid"
# rm_tqdb $udb --table "$pid"_tbl_fac_nrm_"$vid"
# rm_tqdb $udb --table "$pid"_tbl_sig_fac_"$vid"
# rm_tqdb $udb --table "$pid"_tbl_sim_fac_"$vid"
# rm_tqdb $udb --table "$pid"_tbl_optimize_"$vid"
# rm_tqdb $udb --table "$pid"_tbl_sig_stg_"$vid"

cls_prv_cache
echo "$(date +'%Y-%m-%d %H:%M:%S') old data removed"

bgn_date_avlb="20160104"
bgn_date_icov="20160503"
bgn_date_fac="20170103"
bgn_date_opt="20171009"
bgn_date="20180102"

python main.py --bgn $bgn_date_avlb --end $end_date available
echo "$(date +'%Y-%m-%d %H:%M:%S') available universe generated"

python main.py --bgn $bgn_date_icov --end $end_date icov
echo "$(date +'%Y-%m-%d %H:%M:%S') icov generated"

python main.py --bgn $bgn_date_avlb --end $end_date css
echo "$(date +'%Y-%m-%d %H:%M:%S') css generated"

python main.py --bgn $bgn_date_avlb --end $end_date srets
echo "$(date +'%Y-%m-%d %H:%M:%S') srets generated"

# python main.py --bgn $bgn_date_fac --end $end_date factors --type raw
# echo "$(date +'%Y-%m-%d %H:%M:%S') factor raw generated"

# python main.py --bgn $bgn_date_fac --end $end_date factors --type nrm
# echo "$(date +'%Y-%m-%d %H:%M:%S') factor nrm generated"

# python main.py --bgn $bgn_date_fac --end $end_date signals --type fac
# echo "$(date +'%Y-%m-%d %H:%M:%S') signals generated"

# python main.py --bgn $bgn_date_fac --end $end_date simulations --type fac
# echo "$(date +'%Y-%m-%d %H:%M:%S') quick simulation for single factor"

python main.py --bgn $bgn_date_opt --end $end_date optimize
echo "$(date +'%Y-%m-%d %H:%M:%S') weights for factors portfolios optimized"

python main.py --bgn $bgn_date --end $end_date signals --type stg
echo "$(date +'%Y-%m-%d %H:%M:%S') signals optimized"

# python main.py --bgn $end_date --end $end_date operations --type opn
# python main.py --bgn $end_date --end $end_date operations --type cls
# echo "$(date +'%Y-%m-%d %H:%M:%S') signals saved to csv"

python main.py --bgn $bgn_date --end $end_date simulations --type stg --omit
echo "$(date +'%Y-%m-%d %H:%M:%S') complex simualations done"
