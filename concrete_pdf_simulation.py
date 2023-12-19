import numpy as np


def concrete_pdf(alpha, x, lambda_):
    n_ = len(x)
    a_ = np.math.factorial(n_ - 1)
    b_ = np.math.pow(lambda_, n_ - 1)
    c_ = [alpha[i] * np.math.pow(x[i], -lambda_) for i in range(n_)]
    c_ = np.sum(c_)
    c_ = np.math.pow(c_, -n_)
    d_ = [alpha[i] * np.math.pow(x[i], -(lambda_+1)) for i in range(n_)]
    d_ = np.prod(d_)
    res = a_ * b_ * c_ * d_
    return res


def concrete_pdf2(alpha, x, lambda_):
    n_ = len(x)
    a_ = np.math.factorial(n_ - 1)
    # b_ = np.math.pow(lambda_, n_ - 1)
    c_ = [alpha[i] * np.math.pow(x[i], -lambda_) for i in range(n_)]
    c_ = np.sum(c_)
    c_ = np.math.pow(c_, -n_)
    d_ = [alpha[i] * np.math.pow(x[i], -(lambda_ + 1)) for i in range(n_)]
    b_ = np.math.pow(lambda_, (n_ - 1) / n_)
    d_2 = [d_[i] * b_ for i in range(n_)]
    d_ = np.prod(d_2)
    res = a_ * c_ * d_
    return res


# s1 = concrete_pdf(alpha=np.array([5.0, 3.0, 6.0]), x=([0.2, 0.7, 0.1]), lambda_=0.1)
# s2 = concrete_pdf(alpha=np.array([5.0, 3.0, 6.0]), x=([0.2, 0.7, 0.1]), lambda_=0.01)
# s3 = concrete_pdf(alpha=np.array([5.0, 3.0, 6.0]), x=([0.2, 0.7, 0.1]), lambda_=0.001)
# s4 = concrete_pdf(alpha=np.array([5.0, 3.0, 6.0]), x=([0.2, 0.7, 0.1]), lambda_=0.0001)
# s5 = concrete_pdf(alpha=np.array([5.0, 3.0, 6.0]), x=([0.2, 0.7, 0.1]), lambda_=0.00001)
# s6 = concrete_pdf(alpha=np.array([5.0, 3.0, 6.0]), x=([0.2, 0.7, 0.1]), lambda_=0.000001)

eps_list = [0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

for eps in eps_list:
    s1 = concrete_pdf(alpha=np.array([1.0, 1.0, 1.0]), x=([0.2, (0.7-eps), eps]), lambda_=eps)
    s2 = concrete_pdf2(alpha=np.array([1.0, 1.0, 1.0]), x=([0.2, (0.7-eps), eps]), lambda_=eps)
    print(s1)
    print(s2)
    # s2 = concrete_pdf(alpha=np.array([5.0, 3.0, 6.0]), x=([0.2, 0.7, 0.1]), lambda_=0.01)
    # s3 = concrete_pdf(alpha=np.array([5.0, 3.0, 6.0]), x=([0.2, 0.7, 0.1]), lambda_=0.001)
    # s4 = concrete_pdf(alpha=np.array([5.0, 3.0, 6.0]), x=([0.2, 0.7, 0.1]), lambda_=0.0001)
    # s5 = concrete_pdf(alpha=np.array([5.0, 3.0, 6.0]), x=([0.2, 0.7, 0.1]), lambda_=0.00001)
    # s6 = concrete_pdf(alpha=np.array([5.0, 3.0, 6.0]), x=([0.2, 0.7, 0.1]), lambda_=0.000001)
