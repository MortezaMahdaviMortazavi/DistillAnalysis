import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillLoss(nn.Module):
    def __init__(self,temperature):
        super().__init__()
        self.temperature = temperature
        self.KL_DIV = nn.KLDivLoss()
        # log softmax student ==> the probability distribution of P in our case
        # log softmax student ==> the probability distribution of Q in our case
        # softloss with KL_DIV ==> showing how much two distribution P and Q are different
        # KL_DIV ==> Dkl = sum(P(x)*log(P(x)/Q(x)) over x in all X)
        
    def forward(self,outputs,teacher_outputs,labels):
        log_softmax_student = F.log_softmax(outputs/self.temperature,dim=1)
        log_softmax_teacher = F.log_softmax(teacher_outputs/self.temperature,dim=1)
        soft_loss = self.KL_DIV(log_softmax_student,log_softmax_teacher) 
        hard_loss = F.cross_entropy(outputs,labels)

        return soft_loss + hard_loss
    

class AttentionTransferLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=4.0):
        super(AttentionTransferLoss, self).__init__()
        self.alpha = alpha  # Weight for classification loss
        self.beta = beta    # Weight for attention transfer loss
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_outputs, teacher_outputs, student_attn, teacher_attn, labels):
        # Classification loss (e.g., Cross-Entropy)
        cls_loss = self.cross_entropy(student_outputs, labels)

        # Attention transfer loss (e.g., Mean Squared Error)
        attn_loss = self.mse_loss(student_attn, teacher_attn)

        # Combine both losses with appropriate weights
        total_loss = self.alpha * cls_loss + self.beta * attn_loss
        return total_loss