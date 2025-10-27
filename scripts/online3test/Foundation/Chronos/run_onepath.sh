data="ETTh2 ETTm1 Weather ECLSelect TrafficSelect Traffic ECL ETTh2Select SynA3 SynB3"
seq_len="336"
pred_len="24 48 96"
devices="0"
iterations="1"

bash scripts/online3test/Foundation/Chronos/onepath_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$iterations"