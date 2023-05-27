import numpy as np

import torch
import torch.utils.data
from torch import linalg as LA
from torch.autograd import Variable

from model_lenet import RegressionModel, RegressionTrain
from model_resnet import MnistResNet, RegressionTrainResNet
from utils import *

import pickle
import argparse

parser = argparse.ArgumentParser(description='Multi-Fashion-MNIST')
parser.add_argument('--base', default='lenet', type=str, help='base model')
parser.add_argument('--solver', default='sdmgrad', type=str, help='which optimization algorithm to use')
parser.add_argument('--alpha', default=0.5, type=float, help='the alpha used in cagrad')
parser.add_argument('--lmbda', default=0.5, type=float, help='the lmbda used in sdmgrad')
parser.add_argument('--seed', default=0, type=int, help='the seed')
parser.add_argument('--niter', default=100, type=int, help='step of (outer) iteration')
parser.add_argument('--initer', default=20, type=int, help='step of inner itration')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def train(dataset, base_model, solver, alpha, lmbda, niter, initer):

    # generate #npref preference vectors
    n_tasks = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load dataset

    # MultiMNIST: multi_mnist.pickle
    if dataset == 'mnist':
        with open('./data/multi_mnist.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

    # MultiFashionMNIST: multi_fashion.pickle
    if dataset == 'fashion':
        with open('./data/multi_fashion.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

    # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
    if dataset == 'fashion_and_mnist':
        with open('./data/multi_fashion_and_mnist.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

    trainX = torch.from_numpy(trainX.reshape(120000, 1, 36, 36)).float()
    trainLabel = torch.from_numpy(trainLabel).long()
    testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
    testLabel = torch.from_numpy(testLabel).long()

    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
    test_set = torch.utils.data.TensorDataset(testX, testLabel)

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))

    # define the base model for ParetoMTL
    if base_model == 'lenet':
        model = RegressionModel(n_tasks).to(device)
    if base_model == 'resnet18':
        model = MnistResNet(n_tasks).to(device)

    # choose different optimizer for different base model
    if base_model == 'lenet':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75, 90], gamma=0.5)

    if base_model == 'resnet18':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    # store infomation during optimization
    task_train_losses = []
    train_accs = []

    # grad
    grad_dims = []
    for mm in model.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), n_tasks).to(device)
    w = torch.ones(n_tasks).to(device) / n_tasks

    # run niter epochs
    for t in range(niter):

        model.train()
        for it, (X, ts) in enumerate(train_loader):

            X, ts = X.to(device), ts.to(device)

            optimizer.zero_grad()
            # compute stochastic gradient
            task_loss = model.forward_loss(X, ts)

            # \nabla F, grads [n_model, n_tasks]
            for i in range(n_tasks):
                if i == 0:
                    task_loss[i].backward(retain_graph=True)
                else:
                    task_loss[i].backward()
                grad2vec(model, grads, grad_dims, i)
                model.zero_grad_shared_modules()

            if solver == 'cagrad':
                g = cagrad(grads, alpha, rescale=1)
            elif solver == 'mgd':
                g = mgd(grads)
            elif solver == 'sgd':
                g = mean_grad(grads)
            elif solver == 'sdmgrad':
                g = sdmgrad(w, grads, lmbda, initer)
            else:
                raise ValueError('Not supported solver.')
            overwrite_grad(model, g, grad_dims)

            # optimization step
            optimizer.step()
            scheduler.step()

        # calculate and record performance
        if t == 0 or (t + 1) % 2 == 0:

            model.eval()
            with torch.no_grad():

                total_train_loss = []
                train_acc = []

                correct1_train = 0
                correct2_train = 0

                for it, (X, ts) in enumerate(train_loader):

                    X, ts = X.to(device), ts.to(device)

                    valid_train_loss = model.forward_loss(X, ts)
                    total_train_loss.append(valid_train_loss)
                    output1 = model(X).max(2, keepdim=True)[1][:, 0]
                    output2 = model(X).max(2, keepdim=True)[1][:, 1]
                    correct1_train += output1.eq(ts[:, 0].view_as(output1)).sum().item()
                    correct2_train += output2.eq(ts[:, 1].view_as(output2)).sum().item()

                train_acc = np.stack([
                    1.0 * correct1_train / len(train_loader.dataset), 1.0 * correct2_train / len(train_loader.dataset)
                ])

                total_train_loss = torch.stack(total_train_loss)
                average_train_loss = torch.mean(total_train_loss, dim=0)

            # record and print
            task_train_losses.append(average_train_loss.data.cpu().numpy())
            train_accs.append(train_acc)

            print('{}/{}: train_loss={}, train_acc={}'.format(t + 1, niter, task_train_losses[-1], train_accs[-1]))

    save_path = './saved_model/%s_%s_solver_%s_niter_%d_seed_%d.pickle' % (dataset, base_model, solver, niter,
                                                                           args.seed)
    torch.save(model.state_dict(), save_path)


def run(dataset='mnist', base_model='lenet', solver='sdmgrad', alpha=0.5, lmbda=0.5, niter=100, initer=20):
    """
    run stochatic moo algorithms
    """

    train(dataset, base_model, solver, alpha, lmbda, niter, initer)


run(dataset='fashion_and_mnist',
    base_model=args.base,
    solver=args.solver,
    alpha=args.alpha,
    lmbda=args.lmbda,
    niter=args.niter,
    initer=args.initer)
