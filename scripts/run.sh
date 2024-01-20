# partition the CIFAR-10 according to Dir(0.1) for 9 clients
# python generate_data.py -d cifar10 -a 1 -cn 20

# run FedAvg on CIFAR-10 with default settings.
# Use main.py like python main.py <method> [args ...]
# ❗ Method name should be identical to the `.py` file name in `src/server`.
python main.py feddiff -d cifar10_iid --join_ratio 1.0
