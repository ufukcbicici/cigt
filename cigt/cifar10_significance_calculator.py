import numpy as np

from auxillary.bootstap_mean_comparison import BootstrapMeanComparison

cifar10_vanilla = np.array([
    0.935700000846386,
    0.93869999730587,
    0.939600001823902,
    0.938400000905991,
    0.937600000214577,
    0.939199999707937,
    0.940099999547005,
    0.937300003159046,
    0.938600001621246,
    0.935799999862909
])


cifar10_smoe_lambda_0 = np.array([
    0.941399991512299,
    0.940799951553345,
    0.941299974918366,
    0.940999984741211,
    0.941100001335144,
    0.941199958324432,
    0.940899968147278,
    0.940899968147278,
    0.941199958324432,
    0.941299974918366,
    0.941199958324432
])

# cifar10_q_cigt = np.array([
#     0.943299949169159,
#     0.942399978637695,
#     0.941799998283386,
#     0.94159996509552,
#     0.941199958324432,
#     0.941199958324432,
#     0.941199958324432,
#     0.941100001335144,
#     0.941100001335144
# ])


cifar10_q_cigt_0 = np.array([
    0.942499995231628,
    0.942499995231628,
    0.942799985408783,
    0.942799985408783,
    0.94269996881485,
    0.942200005054474,
    0.942099988460541,
    0.941999971866608,
    0.941999971866608,
    0.941999971866608
])

cifar10_q_cigt_0_mac = np.array([
    0.533286094665527,
    0.683510839939117,
    0.489854782819748,
    0.537015795707703,
    0.607287704944611,
    0.0873093754053116,
    0.228525280952454,
    0.229037389159203,
    0.224182948470116,
    0.0863984003663063])

cifar10_q_cigt_001 = np.array([
    0.942099988460541,
    0.941999971866608,
    0.941999971866608,
    0.941999971866608,
    0.941899955272675,
    0.94269996881485,
    0.942399978637695,
    0.942299962043762,
    0.942999958992004,
    0.942799985408783
])

cifar10_q_cigt_001_mac = np.array([
    0.210378125309944,
    0.215559899806976,
    0.257952153682709,
    0.210578858852386,
    0.207863181829453,
    0.601949393749237,
    0.535393714904785,
    0.595042884349823,
    0.21648845076561,
    0.574628412723541])

cifar10_q_cigt_005 = np.array([
    0.942200005054474,
    0.942299962043762,
    0.943099975585938,
    0.942799985408783,
    0.94269996881485,
    0.942399978637695,
    0.941999971866608,
    0.941799998283386,
    0.941899955272675,
    0.941799998283386,
])

cifar10_q_cigt_005_mac = np.array([
    0.107133887708187,
    0.330193251371384,
    0.550428628921509,
    0.542195796966553,
    0.574457705020905,
    0.27074259519577,
    0.144803151488304,
    0.18406730890274,
    0.269431740045547,
    0.27157273888588])

cifar10_q_cigt_01 = np.array([
    0.942500005054474,
    0.943200005054474,
    0.942800005054474,
    0.942899981689453,
    0.94299996509552,
    0.943299949169159,
    0.942699978637695,
    0.941799998283386,
    0.94199996509552,
    0.942299991512299
])

cifar10_q_cigt_01_mac = np.array([
    0.115703023970127,
    0.102349337458611,
    0.10634074151516,
    0.0822133794426918,
    0.0801154747605324,
    0.171554499864578,
    0.516794264316559,
    0.204833072423935,
    0.176567715406418,
    0.0275140646845102
])

cifar10_q_cigt_05 = np.array([
    0.94159996509552,
    0.941399991512299,
    0.941299974918366,
    0.941299974918366,
    0.941199958324432,
    0.941100001335144,
    0.941199958324432,
    0.941199958324432,
    0.941299974918366,
    0.940899968147278
])

cifar10_q_cigt_05_mac = np.array([
    0.0105783799663186,
    0.0310885086655617,
    0.0114009026437998,
    0.0244363639503717,
    0.0250946097075939,
    0.0467005223035812,
    0.0272823050618172,
    0.037947803735733,
    0.0216581858694553,
    0.0163593143224716
])

cifar10_q_cigt_1 = np.array([
    0.941100001335144,
    0.941199958324432,
    0.941199958324432,
    0.941499948501587,
    0.941199958324432,
    0.941899955272675,
    0.941799998283386,
    0.941599977970123,
    0.94049996137619,
    0.940599977970123
])

cifar10_q_cigt_1_mac = np.array([
    0.00658244686201215,
    0.0257644709199667,
    0.0411286726593971,
    0.0262639615684748,
    0.0392623580992222,
    0.0280838161706924,
    0.0273248981684446,
    0.0165490452200174,
    0.0279908869415522,
    0.0164406280964613
])

cifar10_q_cigt_15 = np.array([
    0.941699981689453,
    0.941399991512299,
    0.941399991512299,
    0.941299974918366,
    0.940999984741211,
    0.940699994564056,
    0.940699994564056,
    0.940599977970123,
    0.940599977970123,
    0.940599977970123
])

cifar10_q_cigt_15_mac = np.array([
    0.0269299522042274,
    0.0345384888350964,
    0.0347011089324951,
    0.0382943525910377,
    0.0377251654863358,
    0.0184076186269522,
    0.0185586269944906,
    0.0186321958899498,
    0.018492802977562,
    0.0185779873281717
])

print("Vanilla Cifar 10 CIGT:{0}".format(np.mean(cifar10_vanilla)))
print("cifar10_smoe_lambda_0 Accuracy:{0}".format(np.mean(cifar10_smoe_lambda_0)))
print("Q Cigt-0 Accuracy:{0} MAC:{1}".format(np.mean(cifar10_q_cigt_0), np.mean(cifar10_q_cigt_0_mac)))
print("Q Cigt-001 Accuracy:{0} MAC:{1}".format(np.mean(cifar10_q_cigt_001), np.mean(cifar10_q_cigt_001_mac)))
print("Q Cigt-005 Accuracy:{0} MAC:{1}".format(np.mean(cifar10_q_cigt_005), np.mean(cifar10_q_cigt_005_mac)))
print("Q Cigt-01 Accuracy:{0} MAC:{1}".format(np.mean(cifar10_q_cigt_01), np.mean(cifar10_q_cigt_01_mac)))
print("Q Cigt-05 Accuracy:{0} MAC:{1}".format(np.mean(cifar10_q_cigt_05), np.mean(cifar10_q_cigt_05_mac)))
print("Q Cigt-1 Accuracy:{0} MAC:{1}".format(np.mean(cifar10_q_cigt_1), np.mean(cifar10_q_cigt_1_mac)))
print("Q Cigt-15 Accuracy:{0} MAC:{1}".format(np.mean(cifar10_q_cigt_15), np.mean(cifar10_q_cigt_15_mac)))


baselines = ["cifar10_vanilla", "cifar10_smoe_lambda_0"]

data_samples = {"cifar10_vanilla": cifar10_vanilla,
                "cifar10_smoe_lambda_0": cifar10_smoe_lambda_0,
                "cifar10_q_cigt_0": cifar10_q_cigt_0,
                "cifar10_q_cigt_001": cifar10_q_cigt_001,
                "cifar10_q_cigt_005": cifar10_q_cigt_005,
                "cifar10_q_cigt_01": cifar10_q_cigt_01,
                "cifar10_q_cigt_05": cifar10_q_cigt_05,
                "cifar10_q_cigt_1": cifar10_q_cigt_1,
                "cifar10_q_cigt_15": cifar10_q_cigt_15}

proposed_models = ["cifar10_q_cigt_01"]

for method in proposed_models:
    print("**********CIGN method:{0}**********".format(method))
    for baseline_method in baselines:
        print("Comparing {0} vs {1}".format(method, baseline_method))
        cign_arr = data_samples[method]
        baseline_arr = data_samples[baseline_method]
        p_value, reject_null_hypothesis = BootstrapMeanComparison.compare(x=cign_arr, y=baseline_arr,
                                                                          boostrap_count=100000)
        print("p-value:{0} Reject H0 for equal means:{1}".format(p_value, reject_null_hypothesis))
#
# x = np.random.uniform(low=0.0, high=10.0, size=(1000,))
# y = np.random.uniform(low=20.0, high=40.0, size=(1250,))
#
# p_value, reject_null_hypothesis = BootstrapMeanComparison.compare(x=x, y=y, boostrap_count=10000)