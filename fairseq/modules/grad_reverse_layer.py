from torch.autograd import Function
import torch

class GradReverse(Function):
    scale=1.0

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_.view_as(input_)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = - grad_output*GradReverse.scale
        return grad_input


#revgrad = GradReverse.appl




def grad_reverse(x, lambd):
    GradReverse.scale=lambd
    return GradReverse.apply(x)


