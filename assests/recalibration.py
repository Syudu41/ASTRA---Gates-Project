import torch
from torch import nn, optim
from torch.nn import functional as F

import assests.metrics

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device="cpu"):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.device = torch.device(device)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input["input"], input["segment_label"], input["feat"])
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1)).to(self.device)
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        #self.cuda()
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = metrics.ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                # print("Input = ", input["input"])
                # print("Input = ", input["segment_label"])
                # print("Input = ", input["feat"])
                # input = input
                logits = self.model(input["input"].to(self.device), input["segment_label"].to(self.device), input["feat"].to(self.device))
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion.loss(logits.cpu().numpy(),labels.cpu().numpy(),15)
        #before_temperature_ece = ece_criterion(logits, labels).item()
        #ece_2 = ece_criterion_2.loss(logits,labels)
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        #print(ece_2)
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.005, max_iter=1000)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits.to(self.device)), labels.to(self.device))
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion.loss(self.temperature_scale(logits).detach().cpu().numpy(),labels.cpu().numpy(),15)
        #after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self
