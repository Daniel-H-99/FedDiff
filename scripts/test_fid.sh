# partition the CIFAR-10 according to Dir(0.1) for 9 clients
# python generate_data.py -d pathmnist -a 1 -cn 9 

# run FedAvg on CIFAR-10 with default settings.
# Use main.py like python main.py <method> [args ...]
# ❗ Method name should be identical to the `.py` file name in `src/server`.
python test_fid.py feddiff -d femnist --join_ratio 1.0 --personal_tag 'private'