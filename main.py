from util import *
from tqdm import tqdm
from scipy import sparse
from sklearn.datasets import load_svmlight_file
from random import shuffle
from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
import gc
import pickle
from multiprocessing import Pool

lr_scheduler = ["default","t","tlog"]
# lr_scheduler = ["default"]

lr_const = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

delays = [0, 10, 50, 100, 500, 1000]
# delays = [0]

num_workers = [1, 4, 10, 20]
# num_workers = [1]

dataset = ["mnist"]

num_epochs = 50

batch_size = 100

logdict = dict()

mnist = fetch_mldata('MNIST original', data_home="data/")
mnist.data = preprocessing.normalize(mnist.data)


def execute(arg):
    nw, max_delay, lr_schedule, lr = arg
    server = ParamServer(num_workers=nw, max_delay=max_delay, dim_in=784, num_classes=10, log_details=False,
                                     layers=[], activation="softmax", lr_const=lr, lr_schedule=lr_schedule,
                                     regularization=0, dataset="mnist", num_epochs=50, batch_size=100, data=mnist)
    logs = server.train()
    del server
    gc.collect()

    return logs


pool = Pool(processes=4)
for nw in tqdm(num_workers):
    logdict[nw] = dict()
    for max_delay in tqdm(delays):
        logdict[nw][max_delay] = dict()
        for lr_schedule in tqdm(lr_scheduler):
            logdict[nw][max_delay][lr_schedule] = dict()
            # for lr in tqdm(lr_const):
            #     logdict[nw][max_delay][lr_schedule][lr] = dict()
            arg = [[nw, max_delay, lr_schedule, i] for i in lr_const]
            logs = pool.map(execute, arg)
                # server = ParamServer(num_workers=nw, max_delay=max_delay, dim_in=784, num_classes=10,log_details=False,
                #                      layers=[], activation="softmax",lr_const=lr, lr_schedule=lr_schedule,
                #                      regularization=0, dataset="mnist", num_epochs=50, batch_size=100, data=mnist)
                # logs = server.train()
            for ind, i in enumerate(lr_const):
                logdict[nw][max_delay][lr_schedule][i] = logs[ind]


            # logdict[nw][max_delay][lr_schedule][lr] = logs
            # del server
            gc.collect()

pickle.dump(logdict, open("datalogs.p","wb"))

