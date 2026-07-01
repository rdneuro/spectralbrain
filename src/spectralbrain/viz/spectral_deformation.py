# -*- coding: utf-8 -*-
"""
spectral_deformation.py  —  SpectralBrain
=========================================

Quantificação e visualização de deformação NÃO-ISOMÉTRICA entre duas superfícies
via variação do espectro de Laplace-Beltrami, *sem registro e sem correspondência*
(Hu, Hamidian, Zhong & Hua, IEEE TVCG 2017,
"Visualizing Shape Deformations with Variation of Geometric Spectrum").

Caso de uso clínico (MTLE-HS): lateralização intrínseca esquerda-vs-direita do
hipocampo. O método é *correspondence-free* e o espectro é invariante a isometrias
e a reflexões — então L-vs-R sai direto, sem espelhar nem registrar, mesmo com
malhas de contagens de vértices diferentes.

────────────────────────────────────────────────────────────────────────────────
A MATEMÁTICA (resumo fiel ao paper)
────────────────────────────────────────────────────────────────────────────────
LBO discreto:  W v = λ S v        (W = stiffness cotangente; S = massa Voronoi/lumped)

Deformação = uma função de escala positiva por-vértice ω (matriz diagonal Ω) na
métrica de Riemann: gω = ω·g. O problema de autovalor ponderado fica

        W v_i = λ_i (Ω S) v_i,     <v_i, v_i>_{ΩS} = 1.

Teorema 2 (derivada do autovalor):   λ̇_i = −λ_i · v_iᵀ Ω̇ S v_i.

Interpolamos os autovalores de N linearmente até os de M, o que dá um sistema
linear na derivada da escala (via produto de Hadamard, Eq. 18 do paper):

        (m_N ∘ v_i ∘ v_i)ᵀ · d  =  (λ_i(q) − λ_M,i) / λ_i(q),     i = 1..k1

onde d = diag(Ω̇) é a incógnita por-vértice e m_N = diag(S_N) são as áreas.
Empilhando i=1..k1:  A·d = b.  O sistema é sub-determinado (k1 ≪ n), então
escolhemos a solução de menor energia de suavização (Eq. 25):

        min_d  dᵀ W d + 2 cᵀ d ,    c = W·v_Ω
        s.a.   A d = b              (alinhamento dos autovalores)
               h_l ≤ v_Ω + d ≤ h_u  (escala positiva e limitada)

Integração linear em K passos, re-inicializando o espectro a cada passo (Eq. 31):

        Ω(q+1) = Ω(q) + 1/(K−q) · Ω̇(q),   Ω(0) = I.

Saída: a função de escala ω por-vértice em N. Visualiza-se log(ω): positivo =
N precisa expandir localmente para casar com M; negativo = contrair. As nodal
lines da deformação (log ω = 0) são o que importa — "topology over magnitude".

────────────────────────────────────────────────────────────────────────────────
Decisões de engenharia (vs. o paper de 2017)
────────────────────────────────────────────────────────────────────────────────
• Operadores cotangente + massa Voronoi-lumped implementados aqui (numpy/scipy),
  sem dependência externa para o núcleo.
• O QP é resolvido com OSQP (esparso). A restrição de alinhamento entra como
  igualdade no solver — NUNCA formamos AᵀA denso (n×n estouraria a RAM em malhas
  de 8k vértices). Há um `align_tol` para relaxar a igualdade numa banda (modo
  "soft"), mais estável quando os limites apertam.
• `multiscale`: varrer k1 crescente revela deformações de frequência crescente
  (k1 baixo → global; k1 alto → fino), no espírito multiescala da SB.

Dependências: numpy, scipy, osqp (pip install osqp). Opcional: vedo (render).

Autor: Rodrigo Debona (Velho Mago) — com Claudinho.
"""

# %% ───────────────────────────── imports & ambiente ─────────────────────────
import os
import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

ArrayF = np.ndarray
ArrayI = np.ndarray
PathLike = Union[str, os.PathLike]


# %% ───────────────────── operadores LBO: stiffness & massa ───────────────────
def cotangent_stiffness(V: ArrayF, F: ArrayI) -> sp.csc_matrix:
    """Matriz de rigidez cotangente W (n×n), convenção do paper (Eq. 4).

    Off-diagonal W_ij = −(cot α_ij + cot β_ij)/2 ; diagonal W_ii = −Σ_j W_ij > 0.
    Forma quadrática vᵀ W v = energia de Dirichlet ≥ 0 (PSD em malhas não-obtusas).
    """
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=int)
    n = V.shape[0]
    i1, i2, i3 = F[:, 0], F[:, 1], F[:, 2]
    v1, v2, v3 = V[i1], V[i2], V[i3]

    # área dupla por face (norma do produto vetorial)
    dblA = np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
    dblA = np.maximum(dblA, 1e-12)

    # cotangentes dos ângulos em cada vértice da face
    cot1 = np.einsum("ij,ij->i", v2 - v1, v3 - v1) / dblA  # ângulo em i1
    cot2 = np.einsum("ij,ij->i", v3 - v2, v1 - v2) / dblA  # ângulo em i2
    cot3 = np.einsum("ij,ij->i", v1 - v3, v2 - v3) / dblA  # ângulo em i3

    # o cot do ângulo em i pondera a aresta OPOSTA (Eq. 2/4)
    I = np.concatenate([i2, i3, i3, i1, i1, i2])
    J = np.concatenate([i3, i2, i1, i3, i2, i1])
    vals = 0.5 * np.concatenate([cot1, cot1, cot2, cot2, cot3, cot3])

    W = sp.csr_matrix((-vals, (I, J)), shape=(n, n))     # off-diagonais (negativas)
    diag = -np.asarray(W.sum(axis=1)).ravel()            # diagonal = −Σ off-diag (>0)
    W = (W + sp.diags(diag)).tocsc()
    W = 0.5 * (W + W.T)                                   # simetriza erros numéricos
    return W.tocsc()


def lumped_mass(V: ArrayF, F: ArrayI) -> ArrayF:
    """Massa lumped (barycêntrica) como vetor diagonal m (n,).

    m_i = (1/3) Σ_{faces incidentes} área. Estável e padrão para LBO lumped; é a
    diagonal exigida pela linearização de Hadamard (Eq. 18).
    """
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=int)
    n = V.shape[0]
    v1, v2, v3 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
    m = np.zeros(n)
    for c in range(3):
        np.add.at(m, F[:, c], area / 3.0)
    m = np.maximum(m, 1e-12)
    return m


# %% ─────────────────── autoproblema generalizado ponderado ───────────────────
def _weighted_eigs(
    W: sp.csc_matrix,
    mass_diag: ArrayF,
    k_nonzero: int,
    sigma: float = -1e-6,
) -> Tuple[ArrayF, ArrayF]:
    """Resolve W v = λ (B) v, B = diag(mass_diag), retornando os k_nonzero
    menores autopares NÃO-NULOS (pula o modo constante λ≈0).

    Autovetores vêm B-ortonormais (vᵀ B v = I), que é a normalização <·,·>_{ΩS}=1.
    """
    B = sp.diags(mass_diag).tocsc()
    vals, vecs = eigsh(W, k=k_nonzero + 1, M=B, sigma=sigma, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    # descarta o primeiro (constante, λ≈0)
    return vals[1:k_nonzero + 1], vecs[:, 1:k_nonzero + 1]


# %% ──────────────────────────── passo de QP (OSQP) ───────────────────────────
def _qp_step(
    W: sp.csc_matrix,
    v_omega: ArrayF,
    A: ArrayF,
    b: ArrayF,
    h_low: ArrayF,
    h_high: ArrayF,
    align_tol: float,
    ridge: float = 1e-8,
) -> ArrayF:
    """Um passo do QP (Eq. 19/25/27) via OSQP esparso.

        min_d  dᵀ W d + 2 cᵀ d ,  c = W v_Ω
        s.a.   b−tol ≤ A d ≤ b+tol      (alinhamento; tol=0 => igualdade dura)
               h_l−v_Ω ≤ d ≤ h_u−v_Ω    (limites de escala)

    A igualdade é mantida como restrição do solver — AᵀA denso NUNCA é formado.
    """
    try:
        import osqp
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "OSQP não encontrado. Instale com `pip install osqp`."
        ) from exc

    n = W.shape[0]
    P = (2.0 * W + ridge * sp.identity(n, format="csc")).tocsc()
    q = 2.0 * (W.dot(v_omega))

    A_align = sp.csc_matrix(A)                      # (k1 × n)
    C = sp.vstack([A_align, sp.identity(n, format="csc")], format="csc")
    lo = np.concatenate([b - align_tol, h_low - v_omega])
    hi = np.concatenate([b + align_tol, h_high - v_omega])

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=C, l=lo, u=hi, verbose=False,
               polish=True, eps_abs=1e-6, eps_rel=1e-6, max_iter=8000)
    res = prob.solve()
    status = res.info.status_val
    if status not in (1, 2):  # 1=solved, 2=solved_inaccurate
        warnings.warn(
            f"OSQP não convergiu plenamente (status={res.info.status}). "
            "Considere aumentar `align_tol` (modo soft) ou afrouxar os limites."
        )
    d = res.x
    if d is None or not np.all(np.isfinite(d)):
        warnings.warn("QP retornou solução inválida; passo zerado.")
        return np.zeros(n)
    return np.asarray(d, dtype=float)


# %% ─────────────────────── algoritmo principal de deformação ─────────────────
def spectral_deformation(
    V_N: ArrayF, F_N: ArrayI,           # malha N (fonte; a escala vive aqui)
    V_M: ArrayF, F_M: ArrayI,           # malha M (alvo)
    k1: int = 100,
    n_steps: int = 10,
    bounds: Tuple[float, float] = (0.1, 10.0),
    align_tol: float = 0.0,
    eig_sigma: float = -1e-6,
    return_history: bool = True,
) -> Dict[str, object]:
    """Alinha os primeiros k1 autovalores de N aos de M via função de escala ω.

    Parameters
    ----------
    V_N, F_N : vértices/faces da malha FONTE N (a função de escala é definida nela).
    V_M, F_M : vértices/faces da malha ALVO M (só seus autovalores são usados).
    k1 : nº de autovalores não-nulos a alinhar. Baixo → deformação global/grosseira;
         alto → detecta deformações de frequência mais fina (paper: k1=100).
    n_steps : K passos da integração linear (paper: K=10 já é suficiente).
    bounds : (h_l, h_u) limites da escala por-vértice (positiva, não-infinita).
    align_tol : 0.0 = igualdade dura (fiel ao paper). >0 = banda (modo "soft",
                mais estável quando os limites apertam).
    eig_sigma : shift do shift-invert do eigsh (pequeno negativo evita singular).

    Returns
    -------
    dict com:
      'scale'      : (n_N,) função de escala ω por-vértice em N.
      'log_scale'  : (n_N,) log(ω) — campo a visualizar (diverge em torno de 0).
      'lambda_M'   : (k1,) autovalores-alvo.
      'lambda_N0'  : (k1,) autovalores iniciais de N.
      'lambda_final': (k1,) autovalores de N após a deformação (≈ lambda_M).
      'align_error': (K,) erro relativo médio de alinhamento por passo (se history).
      'k1','n_steps' : parâmetros usados.

    Notes
    -----
    Custo: O(n²) por iteração de autovalor, linear em k1 e em K. Em malhas
    hipocampais (~5k–8k vértices) com k1=100, K=10 roda em segundos a poucos
    minutos na workstation. NÃO rode isto no sandbox — entregue para a RTX 3090.
    """
    V_N = np.asarray(V_N, float); F_N = np.asarray(F_N, int)
    V_M = np.asarray(V_M, float); F_M = np.asarray(F_M, int)
    n_N = V_N.shape[0]
    h_l, h_u = float(bounds[0]), float(bounds[1])
    h_low = np.full(n_N, h_l)
    h_high = np.full(n_N, h_u)

    # operadores de N (massa lumped fixa; só a escala Ω muda entre passos)
    W_N = cotangent_stiffness(V_N, F_N)
    m_N = lumped_mass(V_N, F_N)

    # autovalores-alvo de M (uma vez)
    W_M = cotangent_stiffness(V_M, F_M)
    m_M = lumped_mass(V_M, F_M)
    lam_M, _ = _weighted_eigs(W_M, m_M, k1, sigma=eig_sigma)

    # estado inicial: Ω(0) = I  →  v_Ω = 1
    v_omega = np.ones(n_N)
    lam_N0, _ = _weighted_eigs(W_N, m_N, k1, sigma=eig_sigma)

    align_err = []
    K = int(n_steps)
    for q in range(K):
        # autopares atuais de N com a métrica escalada Ω(q) S  (B = diag(v_Ω * m_N))
        b_diag = v_omega * m_N
        lam_q, vecs_q = _weighted_eigs(W_N, b_diag, k1, sigma=eig_sigma)

        if return_history:
            rel = np.mean(np.abs(lam_q - lam_M) / np.maximum(np.abs(lam_M), 1e-12))
            align_err.append(float(rel))

        # sistema linear A d = b  (Eq. 18/19): A_i = m_N ∘ v_i ∘ v_i
        A = (m_N[None, :] * (vecs_q.T ** 2))          # (k1 × n_N)
        b = (lam_q - lam_M) / np.maximum(lam_q, 1e-12)

        d = _qp_step(W_N, v_omega, A, b, h_low, h_high, align_tol)

        # integração linear (Eq. 31) + projeção nos limites por segurança
        v_omega = v_omega + d / (K - q)
        v_omega = np.clip(v_omega, h_l, h_u)

    # autovalores finais de N após deformação
    lam_final, _ = _weighted_eigs(W_N, v_omega * m_N, k1, sigma=eig_sigma)

    out: Dict[str, object] = {
        "scale": v_omega,
        "log_scale": np.log(np.clip(v_omega, 1e-12, None)),
        "lambda_M": lam_M,
        "lambda_N0": lam_N0,
        "lambda_final": lam_final,
        "k1": k1,
        "n_steps": K,
    }
    if return_history:
        out["align_error"] = np.asarray(align_err, dtype=float)
    return out


# %% ─────────────────── wrapper clínico: lateralização MTLE-HS ────────────────
def lateralization_map(
    V_ipsi: ArrayF, F_ipsi: ArrayI,
    V_contra: ArrayF, F_contra: ArrayI,
    k1: int = 100,
    n_steps: int = 10,
    bounds: Tuple[float, float] = (0.1, 10.0),
    align_tol: float = 0.0,
) -> Dict[str, object]:
    """Mapa de lateralização L-vs-R do hipocampo, definido na malha IPSILATERAL.

    Convenção MTLE-HS: N = ipsilateral (lado da esclerose), M = contralateral.
    log(ω) > 0 marca onde o ipsi precisa expandir para casar com o contra — ou
    seja, regiões de ATROFIA relativa do ipsi (precisa "crescer" para igualar).
    log(ω) < 0 marca regiões onde o ipsi está relativamente maior.

    Como o método é correspondence-free e reflection-invariant, NÃO espelhe nem
    registre as malhas; passe-as como saem do HippUnfold/segmentação.
    """
    res = spectral_deformation(
        V_ipsi, F_ipsi, V_contra, F_contra,
        k1=k1, n_steps=n_steps, bounds=bounds, align_tol=align_tol,
    )
    res["convention"] = "N=ipsilateral, M=contralateral; log(ω)>0 => atrofia relativa do ipsi"
    return res


# %% ─────────────────────────── render (vedo, offscreen) ──────────────────────
def render_scale_on_mesh(
    V: ArrayF, F: ArrayI,
    scalar: ArrayF,
    out_path: PathLike = "spectral_deformation.png",
    cmap: str = "RdBu_r",
    percentile_clim: float = 98.0,
    symmetric: bool = True,
    title: Optional[str] = None,
    size: Tuple[int, int] = (1200, 900),
    zoom: float = 1.3,
) -> str:
    """Pinta um campo escalar por-vértice (tipicamente log ω) sobre a malha.

    Divergente azul-branco-vermelho, clim simétrico por percentil (robusto a
    outliers), render offscreen via vedo. Serve para qualquer estrutura
    (hipocampo, subcortical, etc.) — não exige template fsLR.
    """
    try:
        import vedo
    except Exception as exc:  # pragma: no cover
        raise ImportError("vedo não encontrado. Instale com `pip install vedo`.") from exc

    s = np.asarray(scalar, dtype=float)
    finite = s[np.isfinite(s)]
    vmax = np.percentile(np.abs(finite), percentile_clim) if finite.size else 1.0
    vmax = float(vmax) if vmax > 0 else 1.0
    vmin = -vmax if symmetric else float(np.percentile(finite, 100 - percentile_clim))

    mesh = vedo.Mesh([np.asarray(V, float), np.asarray(F, int)])
    mesh.cmap(cmap, s, vmin=vmin, vmax=vmax).add_scalarbar(title=title or "log ω")

    plt = vedo.Plotter(offscreen=True, size=size)
    plt.show(mesh, zoom=zoom, axes=0)
    plt.screenshot(str(out_path))
    plt.close()
    print(f"render salvo em: {out_path}")
    return str(out_path)


# %% ─────────────────────────────── exemplo de uso ───────────────────────────
if __name__ == "__main__":
    # Carregue suas malhas hipocampais (V: (n,3) float, F: (m,3) int).
    # Ex.: via nibabel GIfTI, trimesh, ou o I/O da SpectralBrain.
    #
    #   import nibabel as nib
    #   g = nib.load("sub-XXX_hemi-L_hipp.surf.gii")
    #   V_L = g.darrays[0].data; F_L = g.darrays[1].data
    #   (idem para o hemi-R)
    #
    # Convenção: N = ipsilateral, M = contralateral.

    # --- placeholder mínimo só para o módulo ser executável sem dados ---
    # (substitua por V_ipsi/F_ipsi e V_contra/F_contra reais)
    print("Carregue as malhas reais e chame lateralization_map(...). Exemplo:")
    print(
        "res = lateralization_map(V_ipsi, F_ipsi, V_contra, F_contra, k1=100)\n"
        "render_scale_on_mesh(V_ipsi, F_ipsi, res['log_scale'],\n"
        "                     out_path='lat_ipsi.png', title='log ω (ipsi→contra)')"
    )
    # Diagnóstico esperado: res['align_error'] deve decair monotonicamente ao
    # longo dos K passos (autovalores de N convergindo aos de M).
