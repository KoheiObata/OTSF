# Configuration file for time series forecasting models
# Manages dataset and pretrain model hyperparameter settings

# Input requirements definition by model type
need_x_y_mark = ['Autoformer', 'Transformer', 'Informer']  # Models that require both x and y marks (time features)
need_x_mark = ['TCN', 'FSNet', 'OneNet', 'iTransformer']   # Models that require only x marks
need_x_mark += [name + '_Ensemble' for name in need_x_mark]  # Add ensemble versions
no_extra_param = ['Online', 'ER', 'DERpp']  # Methods that don't require additional parameters
peft_methods = ['lora', 'adapter', 'ssf', 'mam_adapter', 'basic_tuning']  # List of PEFT methods


# Dataset-specific settings
# Define filename, target column, feature dimensions, batch size, etc. for each dataset
data_settings = {
    'wind_N2': {'data': 'wind_N2.csv', 'T':'FR51', 'M':[254, 254], 'prefetch_batch_size': 16},
    'wind': {'data': 'wind.csv', 'T':'UK', 'M':[28,28], 'prefetch_batch_size': 64},
    'ECL':{'data':'electricity.csv','T':'OT','M':[321,321],'S':[1,1],'MS':[321,1], 'prefetch_batch_size': 10},
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'Solar':{'data':'solar_AL.txt','T': 136,'M':[137,137],'S':[1,1],'MS':[137,1], 'prefetch_batch_size': 32},
    'Weather':{'data':'weather.csv','T':'OT','M':[21,21],'S':[1,1],'MS':[21,1], 'prefetch_batch_size': 64},
    'WTH':{'data':'WTH.csv','T':'OT','M':[12,12],'S':[1,1],'MS':[12,1], 'prefetch_batch_size': 64},
    'Traffic': {'data': 'traffic.csv', 'T':'OT', 'M':[862,862], 'prefetch_batch_size': 2},
    'METR_LA': {'data':'metr-la.csv','T': '773869','M':[207,207],'S':[1,1],'MS':[207,1], 'prefetch_batch_size': 16},
    'PEMS_BAY': {'data':'pems-bay.csv','T': 400001,'M':[325,325],'S':[1,1],'MS':[325,1], 'prefetch_batch_size': 10},
    'NYC_BIKE': {'data':'nyc-bike.h5','T': 0,'M':[500,500],'S':[1,1],'MS':[500,1], 'prefetch_batch_size': 4, 'feat_dim': 2},
    'NYC_TAXI': {'data':'nyc-taxi.h5','T': 0,'M':[532,532],'S':[1,1],'MS':[532,1], 'prefetch_batch_size': 4, 'feat_dim': 2},
    'PeMSD4': {'data':'PeMSD4/PeMSD4.npz','T': 0,'M':[921,921],'S':[1,1],'MS':[921,1], 'prefetch_batch_size': 2, 'feat_dim': 3},
    'PeMSD8': {'data':'PeMSD8/PeMSD8.npz','T': 0,'M':[510,510],'S':[1,1],'MS':[510,1], 'prefetch_batch_size': 6, 'feat_dim': 3},
    'Exchange': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'exchange_rate': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'Illness': {'data': 'illness.csv', 'T':'OT', 'M':[7,7], 'prefetch_batch_size': 128},
    'NOAA': {'data': 'NOAA.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'Powersupply': {'data': 'Powersupply.csv', 'T':'OT', 'M':[2,2], 'prefetch_batch_size': 128},
    'AU_Electricity': {'data': 'AU_Electricity.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'nsw_elc_price': {'data': 'nsw_elc_price.csv', 'T':'OT', 'M':[1,1], 'prefetch_batch_size': 128},
    'nsw_elc_demand': {'data': 'nsw_elc_demand.csv', 'T':'OT', 'M':[1,1], 'prefetch_batch_size': 128},
    'tetuan_power': {'data': 'tetuan_power.csv', 'T':'OT', 'M':[3,3], 'prefetch_batch_size': 128},
    'tetuan_temp': {'data': 'tetuan_temp.csv', 'T':'OT', 'M':[1,1], 'prefetch_batch_size': 128},
    'ECLSelect':{'data':'electricityselect.csv','T':'OT','M':[10,10],'S':[1,1],'MS':[10,1], 'prefetch_batch_size': 10},
    'TrafficSelect': {'data': 'trafficselect.csv', 'T':'OT', 'M':[10,10], 'prefetch_batch_size': 2},
    'ETTh2Select':{'data':'ETTh2Select.csv','T':'OT','M':[6,6],'S':[1,1],'MS':[6,1], 'prefetch_batch_size': 128},
    'SynA3': {'data': 'synthetic_A3.csv', 'T':'OT', 'M':[10,10], 'prefetch_batch_size': 2},
    'SynB3': {'data': 'synthetic_B3.csv', 'T':'OT', 'M':[10,10], 'prefetch_batch_size': 2},
}


# Default hyperparameters by model
hyperparams = {
    'PatchTST': {'e_layers': 3},
    'iTransformer': {'e_layers': 3, 'd_model': 512, 'd_ff': 512, 'activation': 'gelu', 'timeenc': 1, 'patience': 10, 'train_epochs': 100, },
}

def get_hyperparams(data, model, args):
    """
    Function to get hyperparameters according to dataset and model

    Args:
        data (str): Dataset name
        model (str): Model name
        args: Argument object

    Returns:
        dict: Adjusted hyperparameters
    """
    hyperparam: dict = hyperparams[model]

    # Dataset-specific adjustments for iTransformer model
    if model == 'iTransformer':
        if data == 'Traffic':
            hyperparam['e_layers'] = 4
        elif 'ETT' in data:
            hyperparam['e_layers'] = 2
            if data == 'ETTh1':
                hyperparam['d_model'] = 256
                hyperparam['d_ff'] = 256
            else:
                hyperparam['d_model'] = 128
                hyperparam['d_ff'] = 128

    # Dataset-specific adjustments for PatchTST model
    if model == 'PatchTST':
        if args.lradj != 'type3':
            if data in ['ETTh1', 'ETTh2', 'Weather', 'Exchange', 'wind']:
                hyperparam['lradj'] = 'type3'
            elif data in ['Illness']:
                hyperparam['lradj'] = 'constant'
            else:
                hyperparam['lradj'] = 'TST'
        if data in ['ETTh1', 'ETTh2', 'Illness']:
            hyperparam.update(**{'dropout': 0.3, 'fc_dropout': 0.3, 'n_heads': 4, 'd_model': 16, 'd_ff': 128})
        elif data in ['ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 128, 'd_ff': 256})
        else:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 64, 'd_ff': 128})

    return hyperparam

# Learning rate settings for online learning (for pretrained models)
pretrain_lr_dict = {
    'TCN': {'ETTh2': 0.001, 'ETTm1': 0.001, 'Traffic': 0.003, 'Weather': 0.001, 'ECL': 0.003},
    'TCN_RevIN': {'ETTh2': 0.001, 'ETTm1': 0.0001, 'Traffic': 0.003, 'Weather': 0.001, 'ECL': 0.003},
    'TCN_Ensemble': {'ETTh2': 0.001, 'ETTm1': 0.0001, 'Traffic': 0.003, 'Weather': 0.001, 'ECL': 0.003},
    'FSNet_RevIN': {'ETTh2': 0.001, 'ETTm1': 0.001, 'Traffic': 0.003, 'Weather': 0.001, 'ECL': 0.003},
    'PatchTST': {'ETTh1': 0.0001, 'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'iTransformer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.001, 'Weather': 0.00001, 'ECL': 0.0005},
}


def drop_last_PatchTST(args):
    """
    Data boundary adjustment function for PatchTST model
    Adjusts test data boundaries according to batch size

    Args:
        args: Argument object (contains dataset, borders, seq_len, pred_len)

    Returns:
        None: Adjusts args.borders[1][2]
    """
    # Set batch size according to dataset
    bs = 128 if args.dataset in ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Weather'] else 32
    # Calculate number of test data
    test_num = args.borders[1][2] - args.borders[0][2] - args.seq_len - args.pred_len + 1
    # Adjust boundaries according to batch size
    args.borders[1][2] -= test_num % bs
