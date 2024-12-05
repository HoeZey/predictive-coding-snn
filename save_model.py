from predcoding.snn.network import EnergySNN
from predcoding.utils import model_result_dict_load, save_checkpoint

use_alif_neurons = True  # whether use adaptive neuron or not
clf_alpha = 1
energy_alpha = 0.05  # - config.clf_alpha
spike_alpha = 0.0  # energy loss on spikes
one_to_one = True
lr = 1e-3
p_dropout = 0.4
is_recurrent = False
b0 = 0.1  # neural threshold baseline

# training parameters
T = 50
K = 10  # k_updates is num updates per sequence
omega = int(T / K)  # update frequency
clip = 1.0
log_interval = 20
epochs = 35
alpha = 0.2
beta = 0.5
rho = 0.0

# set input and t param
# set input and t param
d_in = 784
d_hidden = [600, 500, 500]
n_classes = 10

# define network
model = EnergySNN(
    d_in,
    d_hidden,
    d_out=n_classes,
    is_adaptive=use_alif_neurons,
    one_to_one=one_to_one,
    p_dropout=p_dropout,
    is_recurrent=is_recurrent,
    b0=b0,
    device="cuda",
)
saved_dict = model_result_dict_load("./checkpoints/best_model.pt.tar")
model.load_state_dict(saved_dict["state_dict"])
save_checkpoint(
    {"state_dict": model.to("cpu").state_dict()},
    prefix="checkpoints/",
    filename="best_model.pt.tar",
)
