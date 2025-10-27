
data="ETTh2 ETTm1 Weather ECLSelect TrafficSelect Traffic ECL ETTh2Select SynA3 SynB3"
seq_len="336"
pred_len="24 48 96"
devices="0"
optim="Adam SGD"



# Onepath
bash scripts/online3test/PatchTST/ER/onepath_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"
bash scripts/online3test/DLinear/ER/onepath_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"
bash scripts/online3test/iTransformer/ER/onepath_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"

bash scripts/online3test/PatchTST/EWC/onepath_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"
bash scripts/online3test/DLinear/EWC/onepath_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"
bash scripts/online3test/iTransformer/EWC/onepath_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"

bash scripts/online3test/PatchTST/Online/onepath_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"
bash scripts/online3test/DLinear/Online/onepath_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"
bash scripts/online3test/iTransformer/Online/onepath_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"

# Pretrain
bash scripts/online3test/PatchTST/Online/pretrain_valid_offline_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"
bash scripts/online3test/DLinear/Online/pretrain_valid_offline_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"
bash scripts/online3test/iTransformer/Online/pretrain_valid_offline_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"

bash scripts/online3test/PatchTST/Offline/offline_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"
bash scripts/online3test/DLinear/Offline/offline_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"
bash scripts/online3test/iTransformer/Offline/offline_valid_test_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$optim"

# Foundation
bash scripts/online3test/Foundation/Chronos/onepath_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$iterations"

# LinearRLS
bash scripts/online3test/LinearRLS/Online/run_onepath.sh
