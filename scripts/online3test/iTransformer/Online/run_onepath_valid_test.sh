# data="ETTh2 ETTm1 Weather WeatherSelect"
# data="ECLSelect TrafficSelect nsw_elc_price Traffic"
# data="nsw_elc_demand SynA2 SynB2 ECL"
# data="ETTh2Select SynA3 SynB3"
data="ETTh2 ETTm1 Weather WeatherSelect ECLSelect TrafficSelect nsw_elc_price Traffic nsw_elc_demand SynA2 SynB2 ECL ETTh2Select SynA3 SynB3"
seq_len="336"
pred_len="24 48 96"
devices="2"
optim="Adam SGD"


bash scripts/online3test/iTransformer/Online/onepath_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"