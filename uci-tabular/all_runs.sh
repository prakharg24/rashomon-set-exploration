python3 train-tabular.py --dataset 'contrac' --method 'base'

python3 train-tabular.py --dataset 'contrac' --method 'sampling' --sampling_nmodel 100 --nepoch 20
python3 train-tabular.py --dataset 'contrac' --method 'sampling' --sampling_nmodel 100 --nepoch 25
python3 train-tabular.py --dataset 'contrac' --method 'sampling' --sampling_nmodel 100 --nepoch 30
python3 train-tabular.py --dataset 'contrac' --method 'sampling' --sampling_nmodel 100 --nepoch 35
python3 train-tabular.py --dataset 'contrac' --method 'sampling' --sampling_nmodel 100 --nepoch 40
python3 train-tabular.py --dataset 'contrac' --method 'sampling' --sampling_nmodel 100 --nepoch 45
python3 train-tabular.py --dataset 'contrac' --method 'sampling' --sampling_nmodel 100 --nepoch 50
python3 train-tabular.py --dataset 'contrac' --method 'sampling' --sampling_nmodel 100 --nepoch 55

python3 train-tabular.py --dataset 'contrac' --method 'dropout' --dropoutmethod 'bernoulli' --drp_nmodel 100 --drp_max_ratio 0.2
python3 train-tabular.py --dataset 'contrac' --method 'dropout' --dropoutmethod 'gaussian' --drp_nmodel 100 --drp_max_ratio 0.6

python3 train-tabular.py --dataset 'contrac' --method 'awp' --awp_eps 0.000,0.004,0.008,0.012,0.016,0.020,0.024,0.028,0.032,0.036,0.040