import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import numpy as np
from math import pi

class Modified_MLP(nn.Cell):
    def __init__(self, dim_in, dim_out, dim_hidden, hidden_layers, init_name='XavierUniform'):
        super(Modified_MLP, self).__init__()
        self.hidden_layers = hidden_layers

        self.activation = nn.Tanh()

        self.model = nn.CellList([nn.Dense(dim_in, dim_hidden, weight_init=init_name), self.activation])
        for i in range(hidden_layers):
            self.model.append(nn.Dense(dim_hidden, dim_hidden, weight_init=init_name))
            self.model.append(self.activation)
        self.model.append(nn.Dense(dim_hidden, dim_out, weight_init=init_name))

        self.fc_U = nn.Dense(dim_in, dim_hidden, weight_init=init_name)
        self.fc_V = nn.Dense(dim_in, dim_hidden, weight_init=init_name)
        self.encoder_U = nn.SequentialCell([self.fc_U, self.activation])
        self.encoder_V = nn.SequentialCell([self.fc_V, self.activation])

        self.mul = ops.Mul()

    def construct(self, x, y, z):
        x = ms.numpy.concatenate([x, y, z], axis=1)
        U = self.encoder_U(x)
        V = self.encoder_V(x)
        for i in range(self.hidden_layers):
            x = self.model[2 * i](x)  # 调用线性层
            x = self.model[2 * i + 1](x)  # 调用激活层
            x = self.mul((1 - x), U) + self.mul(x, V)  # 特征融合
        x = self.model[-1](x)  # 调用最后一个线性层得到输出

        return x

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=False)
        self.network = network
        self.firstgrad = self.grad(self.network)

    def construct(self, x, y, z):
        gout = self.firstgrad(x, y, z)  # return dx, dy, dz
        return gout


class GradSec(nn.Cell):
    global data_length
    def __init__(self, net):
        super(GradSec, self).__init__()
        self.grad1 = ops.GradOperation(get_all=True, sens_param=False)
        self.forward_net = net
        self.first_grad = self.grad1(self.forward_net)

        self.grad2 = ops.GradOperation(get_all=True, sens_param=True)
        self.second_grad = self.grad2(self.first_grad)

        self.sens1 = ms.Tensor(np.ones([data_length, 1]).astype('float32'))
        self.sens2 = ms.Tensor(np.zeros([data_length, 1]).astype('float32'))

    def construct(self, x, y, z):
        dxdx, dxdy, dxdz = self.second_grad(x, y, z, (self.sens1, self.sens2, self.sens2))
        dydx, dydy, dydz = self.second_grad(x, y, z, (self.sens2, self.sens1, self.sens2))
        dzdx, dzdy, dzdz = self.second_grad(x, y, z, (self.sens2, self.sens2, self.sens1))
        return dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy, dzdz


class PINN_poisson(nn.Cell):
    """定义PINN的损失网络"""

    def __init__(self, backbone):
        super(PINN_poisson, self).__init__(auto_prefix=False)
        self.backbone = backbone

        self.firstgrad = Grad(backbone)  # first order
        self.secondgrad = GradSec(backbone)  # second order

        self.mul = ops.Mul()

    def construct(self, xyz, xyz_b, u_b):
        loss_r = self.mul(100, mnp.mean((self.net_r(xyz)) ** 2))
        loss_b = self.mul(10000, mnp.mean(((self.net_u(xyz_b) - u_b)) ** 2))
        loss = loss_r + loss_b

        return loss

    def net_u(self, xyz):
        x = xyz[:, [0]]
        y = xyz[:, [1]]
        z = xyz[:, [2]]
        u = self.backbone(x, y, z)
        return u

    def net_r(self, xyz):
        x = xyz[:, [0]]
        y = xyz[:, [1]]
        z = xyz[:, [2]]
        u = self.backbone(x, y, z)

        u_xx, _, _, _, u_yy, _, _, _, u_zz = self.secondgrad(x, y, z)
        residual = u_xx + u_yy + u_zz + ops.sin(self.mul(4*pi, xyz[:, [0]])) * ops.sin(
            self.mul(4*pi, xyz[:, [1]])) * ops.sin(self.mul(4*pi, xyz[:, [2]]))

        return residual


class CustomTrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer):
        """入参有两个：训练网络，优化器"""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                           # 定义前向网络
        self.network.set_grad()                          # 构建反向网络
        self.optimizer = optimizer                       # 定义优化器
        self.weights = self.optimizer.parameters         # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, *inputs):
        loss = self.network(*inputs)                            # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        self.optimizer(grads)                                   # 使用优化器更新权重参数
        return loss


class CustomTrainOneStepCell_lbfgs(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer):
        """入参有两个：训练网络，优化器"""
        super(CustomTrainOneStepCell_lbfgs, self).__init__(auto_prefix=False)
        self.network = network                           # 定义前向网络
        self.network.set_grad()                          # 构建反向网络
        self.optimizer = optimizer                       # 定义优化器
        self.weights = self.optimizer.parameters         # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, *inputs):
        loss = self.network(*inputs)                            # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        return loss