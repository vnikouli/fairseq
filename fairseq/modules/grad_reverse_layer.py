from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input


revgrad = GradReverse.apply

def grad_reverse(x, lambd):
    return lambd*GradReverse.apply(x)


