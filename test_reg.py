import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from scipy.stats import pearsonr
from numpy.linalg import cholesky
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score             

# 1. 新しい相関行列を持つデータの準備 (前回と同じ設定)
np.random.seed(43) 
T = 1000 

# 目標とする相関行列 (R)
target_R = np.array([
    [1.00, 0.65, 0.85],
    [0.65, 1.00, 0.70],
    [0.85, 0.70, 1.00]
])

# R のコレスキー分解とデータ生成
L = cholesky(target_R)
Z = np.random.normal(0, 1, (T, 3))
X_matrix = Z @ L.T
X_matrix = X_matrix - np.mean(X_matrix, axis=0) # センターリング

X1 = X_matrix[:, 0]
X2 = X_matrix[:, 1]
X3 = X_matrix[:, 2]
X_list = [X1, X2, X3]
N_vars = 3
TARGET_CORR = 0.9 

# -----------------------------------------------
# 2. 最適化問題の定義 (変更なし)
def create_Fi(coeffs, X):
    return X @ coeffs

def solve_Fi(target_idx, X_list, X_matrix, target_corr):
    
    Xi = X_list[target_idx]
    min_indices = [i for i in range(N_vars) if i != target_idx]
    Xj = X_list[min_indices[0]]
    Xk = X_list[min_indices[1]]

    def objective_function(coeffs):
        Fi = create_Fi(coeffs, X_matrix)
        if np.std(Fi) < 1e-6:
            return 1e10
        corr_Fi_Xj = pearsonr(Fi, Xj)[0]
        corr_Fi_Xk = pearsonr(Fi, Xk)[0]
        return corr_Fi_Xj**2 + corr_Fi_Xk**2

    def constraint_corr_Xi_ge(coeffs):
        Fi = create_Fi(coeffs, X_matrix)
        if np.std(Fi) < 1e-6:
            return -1.0 
        current_corr = pearsonr(Fi, Xi)[0]
        return current_corr - target_corr

    def constraint_variance_Fi(coeffs):
        Fi = create_Fi(coeffs, X_matrix)
        return np.var(Fi) - 1.0

    initial_guess = np.zeros(N_vars)
    initial_guess[target_idx] = 1.0

    constraints = [
        {'type': 'ineq', 'fun': constraint_corr_Xi_ge}, 
        {'type': 'eq', 'fun': constraint_variance_Fi}
    ]
    bounds = Bounds([-10.0] * N_vars, [10.0] * N_vars)

    result = minimize(
        objective_function, 
        initial_guess, 
        method='SLSQP', 
        bounds=bounds,
        constraints=constraints,
        options={'disp': False, 'ftol': 1e-6}
    )
    
    Fi_optimized = create_Fi(result.x, X_matrix)
    corr_Fi_Xi = pearsonr(Fi_optimized, Xi)[0]
    corr_Fi_Xj = pearsonr(Fi_optimized, Xj)[0]
    corr_Fi_Xk = pearsonr(Fi_optimized, Xk)[0]
    
    return {
        'success': result.success,
        'coeffs': result.x,
        'corr_target': corr_Fi_Xi,
        'corr_min_1': corr_Fi_Xj,
        'corr_min_2': corr_Fi_Xk,
        'objective_value': result.fun,
        'min_vars': (f'X{min_indices[0]+1}', f'X{min_indices[1]+1}'),
        'Fi': Fi_optimized 
    }
    
# 3. F1, F2, F3 の計算と結果出力 
print("--- データの相関行列 (新しい設定) ---")
print(pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3}).corr().round(4))
print("----------------------------------------\n")

results = {}
F_data = {} 
R2_scores = {}

for i in range(N_vars):
    target_name = f'X{i+1}'
    factor_name = f'F{i+1}'
    
    results[factor_name] = solve_Fi(i, X_list, X_matrix, TARGET_CORR)
    res = results[factor_name]
    F_data[factor_name] = res['Fi'] 

    # 単回帰: Xn を Fn で回帰 (Xn ~ Fn)
    Xi = X_list[i]
    Fi = res['Fi'].reshape(-1, 1) 
    
    model = LinearRegression()
    model.fit(Fi, Xi)
    R2 = model.score(Fi, Xi)
    R2_scores[target_name] = R2

    # 個別の最適化結果を出力
    print(f"--- {factor_name} の最適化 (Target: {target_name}) ---")
    print(f"  成功: {res['success']}")
    print(f"  最適化された係数 a (a1, a2, a3): {res['coeffs'].round(4)}")
    print(f"  {target_name} と {factor_name} の相関 (目標: >= {TARGET_CORR}): {res['corr_target']:.4f}")
    print(f"  {res['min_vars'][0]} と {factor_name} の相関: {res['corr_min_1']:.4f}")
    print(f"  {res['min_vars'][1]} と {factor_name} の相関: {res['corr_min_2']:.4f}")
    print(f"  目的関数 (相関二乗和): {res['objective_value']:.6f}")
    print(f"  【単回帰 R^2 (X{i+1} ~ F{i+1})】: {R2:.4f} (理論値 {res['corr_target']**2:.4f})")
    print("-" * 35)

# -----------------------------------------------
## 4. 最終結果のまとめ (エラー修正箇所を含む)
# -----------------------------------------------
print("\n" + "="*50)
print("=== 最終結果のまとめ ===")
print("="*50)

# 決定係数の出力
print(">> Xn を Fn で単回帰した決定係数 (R^2):")
for X_var, R2_val in R2_scores.items():
    print(f"   {X_var} ~ {X_var.replace('X', 'F')}: {R2_val:.4f}")

# 最終的な相関行列を作成 (エラー修正箇所)
F1 = F_data['F1']
F2 = F_data['F2']
F3 = F_data['F3']
F_matrix = np.vstack([F1, F2, F3]).T # F_matrix はそのまま
All_data_df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'F1': F1, 'F2': F2, 'F3': F3})
final_corr_matrix = All_data_df[['X1', 'X2', 'X3', 'F1', 'F2', 'F3']].corr().round(4)

print("\n>> 最終結果の相関行列 (X と F) :")
print(final_corr_matrix)