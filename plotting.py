import pickle
from matplotlib import pyplot as plt


logged = pickle.load(open("datalogs.p","rb"))
num_epochs = 50
# 1: num workers, 2: delay, 3: lr schedule, 4: lr const
fig, ax = plt.subplots(3)


for lr, logs in logged[10][0]["default"].items():
    #     print(logs)
    gen_error = [abs(logs[i]["train_loss"] - logs[i]["test_loss"]) for i in range(num_epochs)]
    train_error = [logs[i]["train_loss"] for i in range(num_epochs)]
    test_error = [logs[i]["test_loss"] for i in range(num_epochs)]

    ax[0].plot(gen_error, label=str(lr))

    ax[0].legend()
    # ax[0].title("generalization")

    ax[1].plot(train_error, label=str(lr) + "train")
    # ax[1].plot(test_error, label=str(lr) + "test")
    ax[1].legend()
    # ax[1].title("training error")

    # train_accuracy = [logs[i]["train_acc"] for i in range(num_epochs)]
    test_accuracy = [logs[i]["test_acc"] for i in range(num_epochs)]
    # ax[2].plot(train_accuracy, label=str(lr) + "train")
    ax[2].plot(test_accuracy, label=str(lr) + "test")
    ax[2].legend()
    # ax[2].title("accuracy")




plt.show()
