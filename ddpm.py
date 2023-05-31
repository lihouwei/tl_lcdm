import torch

# 确定超参数的值
num_steps = 1000

# 制定每一步的beta
# betas = torch.linspace(-6, 6, num_steps)
# betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_steps)

# 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
       alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape \
       == one_minus_alphas_bar_sqrt.shape
print("all the same shape", betas.shape)


# 3、确定扩散过程任意时刻的采样值
# 计算任意时刻的x采样值，基于x_0和重参数化
def q_x(x_0, t):
    """可以基于x[0]得到任意时刻t的x[t]"""
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return alphas_t * x_0 + alphas_1_m_t * noise  # 在x[0]的基础上添加噪声


# 6、编写训练的误差函数
def diffusion_loss_fn(model, x_0, depth, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """对任意时刻t进行采样计算loss"""
    batch_size = x_0.shape[0]

    # 对一个batchsize样本生成随机的时刻t
    t = torch.randint(0, n_steps, size=(batch_size,))
    t = t.unsqueeze(-1)

    # x0的系数
    a = alphas_bar_sqrt[t]
    a = a.to('cuda')
    # eps的系数
    aml = one_minus_alphas_bar_sqrt[t]
    aml = aml.to('cuda')
    # 生成随机噪音eps
    e = torch.randn_like(x_0)

    # 构造模型的输入
    x = x_0 * a + e * aml

    depth = depth.reshape([batch_size,])

    # 送入模型，得到t时刻的随机噪声预测值
    t = t.squeeze(-1)
    output = model(x, t, depth)

    # 与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()


# 7、编写逆扩散采样函数（inference）
def p_sample_loop(model, shape, n_steps, depth, betas, one_minus_alphas_bar_sqrt, device='cuda'):
    """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
    cur_x = torch.randn(shape)
    cur_x = cur_x.to(device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, depth, betas, one_minus_alphas_bar_sqrt, device)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, depth, betas, one_minus_alphas_bar_sqrt, device='cuda'):
    """从x[T]采样t时刻的重构值"""
    t = torch.tensor([t])
    betas = betas.to(device)
    one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t, depth)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return sample





