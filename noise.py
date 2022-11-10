import torch

g = torch.Generator()
g.manual_seed(2147483647)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), generator=g) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddLabelNoise(object):
    def __init__(self, eta=0., n_classes=10):
        self.eta = eta
        self.n_classes = n_classes
        
    def __call__(self, y):
        # Binary vector indicating to change the label (1) or not (0)
        flag_change = torch.bernoulli(torch.tensor([self.eta], dtype=float), generator=g).int()
        # Vector of random integers from 1 to n_class-1
        random_label = torch.randint(1,self.n_classes+1,(1,), generator=g)

        return (y + flag_change*random_label).remainder(self.n_classes).numpy()[0]
    
    def __repr__(self):
        return self.__class__.__name__ + '(eta={0})'.format(self.eta)


