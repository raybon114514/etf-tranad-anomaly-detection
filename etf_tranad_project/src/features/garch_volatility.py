import numpy as np
import scipy.optimize as opt


class GARCHModels:
    """GARCH(1,1) 與 GJR-GARCH(1,1) QMLE 參數估計與條件波動率特徵萃取"""

    @staticmethod
    def fit_garch11(returns_train):
        """對稱 GARCH(1, 1) QMLE 估計"""
        r = returns_train * 100.0
        mu = np.mean(r)
        e = r - mu
        T = len(r)
        var_sample = np.var(r)

        def nll(params):
            omega, alpha, beta = params
            sigma2 = np.zeros(T)
            sigma2[0] = var_sample
            for t in range(1, T):
                sigma2[t] = omega + alpha * (e[t - 1] ** 2) + beta * sigma2[t - 1]
                if sigma2[t] <= 1e-6:
                    sigma2[t] = 1e-6
            return 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + (e ** 2) / sigma2)

        cons = ({'type': 'ineq', 'fun': lambda p: 0.9999 - p[1] - p[2]})
        bounds = [(1e-6, None), (1e-6, 1.0), (1e-6, 1.0)]

        best_res, best_fun = None, np.inf
        for a_init, b_init in [(0.05, 0.90), (0.03, 0.94), (0.10, 0.85)]:
            w_init = var_sample * (1.0 - a_init - b_init)
            res = opt.minimize(nll, [w_init, a_init, b_init], method='SLSQP', bounds=bounds, constraints=cons)
            if res.success and res.fun < best_fun:
                best_fun = res.fun
                best_res = res

        omega, alpha, beta = best_res.x
        omega /= (100.0 ** 2)
        return mu / 100.0, omega, alpha, beta

    @staticmethod
    def fit_gjr_garch11(returns_train):
        """非對稱 GJR-GARCH(1, 1) QMLE 估計"""
        r = returns_train * 100.0
        mu = np.mean(r)
        e = r - mu
        T = len(r)
        var_sample = np.var(r)

        def nll(params):
            omega, alpha, gamma, beta = params
            sigma2 = np.zeros(T)
            sigma2[0] = var_sample
            for t in range(1, T):
                I = 1.0 if e[t - 1] < 0 else 0.0
                sigma2[t] = omega + (alpha + gamma * I) * (e[t - 1] ** 2) + beta * sigma2[t - 1]
                if sigma2[t] <= 1e-6:
                    sigma2[t] = 1e-6
            return 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + (e ** 2) / sigma2)

        cons = ({'type': 'ineq', 'fun': lambda p: 0.9999 - p[1] - p[3] - p[2] / 2.0})
        bounds = [(1e-6, None), (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0)]

        best_res, best_fun = None, np.inf
        for a_init, g_init, b_init in [(0.04, 0.05, 0.88), (0.05, 0.08, 0.90), (0.06, 0.10, 0.85)]:
            w_init = var_sample * (1.0 - a_init - g_init / 2.0 - b_init)
            res = opt.minimize(nll, [w_init, a_init, g_init, b_init], method='SLSQP', bounds=bounds, constraints=cons)
            if res.success and res.fun < best_fun:
                best_fun = res.fun
                best_res = res

        omega, alpha, gamma, beta = best_res.x
        omega /= (100.0 ** 2)
        return mu / 100.0, omega, alpha, gamma, beta

    @staticmethod
    def extract_features(returns_all, model_type, params):
        """固定訓練參數，向後遞迴外推產生全樣本的條件波動度 sigma 與標準化殘差 z"""
        T = len(returns_all)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns_all[:100])

        if model_type == "GARCH":
            mu, omega, alpha, beta = params
            e = returns_all - mu
            for t in range(1, T):
                sigma2[t] = omega + alpha * (e[t - 1] ** 2) + beta * sigma2[t - 1]
        elif model_type == "GJR":
            mu, omega, alpha, gamma, beta = params
            e = returns_all - mu
            for t in range(1, T):
                I = 1.0 if e[t - 1] < 0 else 0.0
                sigma2[t] = omega + (alpha + gamma * I) * (e[t - 1] ** 2) + beta * sigma2[t - 1]

        sigma = np.sqrt(np.maximum(sigma2, 1e-8))
        z = (returns_all - params[0]) / sigma
        return sigma, z
