"""
==============================================================================
OBLICZENIA PARAMETRÓW SILNIKA RAKIETOWEGO: H2 / H2O2 98%
WARIANT: OPTYMALNY WSPÓŁCZYNNIK EKWIWALENCJI φ = 2.86
==============================================================================
Skrypt wyznacza parametry termodynamiczne i geometryczne silnika rakietowego
z dyszą zbieżno-rozbieżną, pracującego w atmosferze.

BIBLIOTEKI:
  - RocketCEA : interfejs do NASA CEA (Chemical Equilibrium with Applications)
                → wyznacza gamma, Tc, Mw dla mieszaniny paliwowej
  - scipy     : metoda brentq do iteracyjnego wyznaczenia Pc
  - numpy     : obliczenia matematyczne

PALIWO:    H2  (ciekły wodór)
UTLENIACZ: H2O2 98% (98 % H2O2 + 2 % H2O wagowo)

DOBÓR STOSUNKU MIESZANKI:
  Stechiometria: H2 + H2O2 → 2 H2O
    MR_stoich = M(H2O2) / M(H2) = 34 / 2 = 17.0
  Współczynnik ekwiwalencji optymalny (max Isp z analizy sweepowej):
    φ_opt = 2.86
  Wynikowy stosunek O/F:
    MR = MR_stoich / φ_opt = 17.0 / 2.86 ≈ 5.944
  (porównaj z wariantem projektowym: MR = 7.94, φ = 2.14)

ZAŁOŻENIA MODELU:
  1. Przepływ 1D (jednowymiarowy)
  2. Przepływ isentropowy (bez strat tarcia i ciepła)
  3. Parametry komory spalania = parametry spiętrzenia (Mc ≈ 0):
       Pc = P0,  Tc = T0,  ρc = ρ0
  4. Dysza zbieżno-rozbieżna: M < 1 przed gardłem, M = 1 w gardzieli,
     M > 1 za gardłem
  5. Gaz doskonały, stałe właściwości termodynamiczne (γ = const)
  6. Projektowe ciśnienie na wylocie Pe = Pa (dopasowanie do atmosfery)

WZORY:
  ve   = sqrt(2γ/(γ-1) · R̄/M̄ · T0 · [1 - (Pe/P0)^((γ-1)/γ)])
  T    = T0 · [1 + (γ-1)/2 · M²]⁻¹
  p    = p0 · [1 + (γ-1)/2 · M²]^(-γ/(γ-1))
  ρ    = ρ0 · [1 + (γ-1)/2 · M²]^(-1/(γ-1))
  A/A* = 1/M · [2/(γ+1) · (1 + (γ-1)/2 · M²)]^((γ+1)/(2(γ-1)))
  A*   = ṁ / (ρt · vt)
==============================================================================
"""

import numpy as np
from scipy.optimize import brentq
from rocketcea.cea_obj import CEA_Obj
from rocketcea.blends import newOxBlend


# ==============================================================================
# 0. STAŁE FIZYCZNE
# ==============================================================================
R_UNIV     = 8314.46      # [J/(kmol·K)] – uniwersalna stała gazowa
G0         = 9.80665      # [m/s²]       – standardowe przyspieszenie ziemskie
PSI_PER_PA = 1 / 6894.76  # przelicznik Pa → psi


# ==============================================================================
# 1. DANE WEJŚCIOWE
# ==============================================================================
F     = 701.8     # [N]    – zadany projektowy ciąg silnika
M_DOT = 0.2       # [kg/s] – zadany strumień masy spalin
PA    = 101325.0  # [Pa]   – ciśnienie otoczenia (1 atm)
EPS_C = 8.0       # [-]    – stosunek Ac/At dla komory spalania

# Stosunek mieszanki – wyznaczony z optymalnego φ
MR_STOICH = 17.0                        # stechiometria H2/H2O2 (czysty)
PHI_OPT   = 2.86                        # optymalny współczynnik ekwiwalencji
MR        = MR_STOICH / PHI_OPT        # O/F ≈ 5.944 – optymalny Isp


# ==============================================================================
# 2. INICJALIZACJA ROCKETCEA
# ==============================================================================
ox_blend = newOxBlend(oxL=['H2O2', 'H2O'], oxPcentL=[98, 2])
cea = CEA_Obj(fuelName='H2', oxName=ox_blend)


# ==============================================================================
# 3. FUNKCJE POMOCNICZE
# ==============================================================================

def get_cea_params(Pc_bar: float, MR: float) -> tuple:
    """
    Pobiera z RocketCEA podstawowe parametry termodynamiczne komory.
    Zwraca: Tc [K], Mw [g/mol], gamma [-], R_sp [J/(kg·K)]
    """
    Pc_psia = Pc_bar * 1e5 * PSI_PER_PA
    Tc      = cea.get_Tcomb(Pc=Pc_psia, MR=MR)
    Mw, gamma = cea.get_Chamber_MolWt_gamma(Pc=Pc_psia, MR=MR)
    R_sp    = R_UNIV / Mw
    return Tc, Mw, gamma, R_sp


def ve_isentropic(Pc_bar: float, MR: float) -> float:
    """
    Isentropowa efektywna prędkość wylotowa przy Pe = Pa.
      ve = sqrt( 2γ/(γ-1) · R_sp · T0 · [1 - (Pe/P0)^((γ-1)/γ)] )
    """
    Tc, Mw, gamma, R_sp = get_cea_params(Pc_bar, MR)
    Pc_Pa = Pc_bar * 1e5
    Pe_Pc = PA / Pc_Pa
    if Pe_Pc >= 1.0:
        return 0.0
    ve = np.sqrt(
        2 * gamma / (gamma - 1)
        * R_sp * Tc
        * (1 - Pe_Pc ** ((gamma - 1) / gamma))
    )
    return ve


def area_ratio(M: float, gamma: float) -> float:
    """
    Stosunek A/A* jako funkcja liczby Macha i gamma.
      A/A* = 1/M · [2/(γ+1) · (1 + (γ-1)/2 · M²)]^((γ+1)/(2(γ-1)))
    """
    factor   = (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M**2)
    exponent = (gamma + 1) / (2 * (gamma - 1))
    return (1 / M) * factor**exponent


# ==============================================================================
# 4. KROK 1 – ve Z RÓWNANIA CIĄGU (założenie Pe = Pa)
# ==============================================================================
# F = ṁ·ve + Ae·(Pe - Pa),  przy Pe = Pa → ve = F / ṁ

ve  = F / M_DOT    # [m/s]
Isp = ve / G0      # [s]


# ==============================================================================
# 5. KROK 2 – WYZNACZENIE Pc Z ROCKETCEA (iteracyjnie, metoda Brenta)
# ==============================================================================
# Szukamy Pc takiego, by ve_isentropic(Pc) = ve = F/ṁ.
# Przedział [2, 20] bar – szerszy niż w wariancie MR=7.94, ponieważ
# przy MR ≈ 5.944 Tc jest wyższe, Mw niższe → wyższe ve_iso → niższe Pc.

def residual(Pc_bar):
    return ve_isentropic(Pc_bar, MR) - ve

Pc_bar_sol = brentq(residual, 2.0, 20.0, xtol=1e-4, rtol=1e-8)

Pc = Pc_bar_sol * 1e5    # [Pa]
Tc, Mw, gamma, R_sp = get_cea_params(Pc_bar_sol, MR)

Cstar       = (np.sqrt(R_sp * Tc / gamma)
               * ((gamma + 1) / 2) ** ((gamma + 1) / (2 * (gamma - 1))))


# ==============================================================================
# 6. PARAMETRY KOMORY SPALANIA (spiętrzenia, Mc ≈ 0)
# ==============================================================================
rho_c = Pc / (R_sp * Tc)   # [kg/m³]


# ==============================================================================
# 7. PARAMETRY W GARDZIELI (M = 1, isentropowe)
# ==============================================================================
Tt    = Tc * 2 / (gamma + 1)
Pt    = Pc * (2 / (gamma + 1)) ** (gamma / (gamma - 1))
rho_t = rho_c * (2 / (gamma + 1)) ** (1 / (gamma - 1))
vt    = np.sqrt(gamma * R_sp * Tt)
At    = M_DOT / (rho_t * vt)
Dt    = np.sqrt(4 * At / np.pi)

Cstar_check = Pc * At / M_DOT


# ==============================================================================
# 8. PARAMETRY NA WYLOCIE DYSZY (isentropowe, Pe = Pa)
# ==============================================================================
Pe      = PA
Te      = Tc * (Pe / Pc) ** ((gamma - 1) / gamma)
Me      = np.sqrt(2 / (gamma - 1) * (Tc / Te - 1))
rho_e   = rho_c * (Pe / Pc) ** (1 / gamma)
ve_isen = np.sqrt(
    2 * gamma / (gamma - 1) * R_sp * Tc
    * (1 - (Pe / Pc) ** ((gamma - 1) / gamma))
)
eps_e   = area_ratio(Me, gamma)
Ae      = eps_e * At
De      = np.sqrt(4 * Ae / np.pi)
F_calc  = M_DOT * ve_isen + Ae * (Pe - PA)


# ==============================================================================
# 9. PARAMETRY W KOMORZE – DOKŁADNE (Ac/At = 8, bez założenia Mc ≈ 0)
# ==============================================================================
def mach_residual_subsonic(Mc_):
    return area_ratio(Mc_, gamma) - EPS_C

Mc        = brentq(mach_residual_subsonic, 1e-8, 1.0 - 1e-10)
Tc_loc    = Tc / (1 + (gamma - 1) / 2 * Mc**2)
Pc_loc    = Pc * (1 + (gamma - 1) / 2 * Mc**2) ** (-(gamma / (gamma - 1)))
rho_c_loc = Pc_loc / (R_sp * Tc_loc)
vc        = Mc * np.sqrt(gamma * R_sp * Tc_loc)
Ac        = EPS_C * At
Dc        = np.sqrt(4 * Ac / np.pi)


# ==============================================================================
# 10. WYNIKI – PODSUMOWANIE
# ==============================================================================
SEP = "=" * 68

print(SEP)
print("  WYNIKI: SILNIK RAKIETOWY H2 / H2O2 98%  [φ_opt = 2.86]")
print(SEP)

print(f"\n  DANE WEJŚCIOWE:")
print(f"    Ciąg          F     = {F} N")
print(f"    Strumień masy ṁ     = {M_DOT} kg/s")
print(f"    Ciśn. otocz.  Pa    = {PA:.0f} Pa")
print(f"    φ (ekwiwalencja)    = {PHI_OPT:.2f}  (optymalny Isp)")
print(f"    MR = O/F            = {MR:.4f}  (= {MR_STOICH}/{PHI_OPT})")
print(f"    Ac/At (komora)      = {EPS_C}")

print(f"\n  PARAMETRY TERMODYNAMICZNE (z RocketCEA):")
print(f"    γ (gamma)           = {gamma:.4f}")
print(f"    Mw                  = {Mw:.4f} g/mol")
print(f"    R_spec              = {R_sp:.4f} J/(kg·K)")
print(f"    C*                  = {Cstar:.4f} m/s")
print(f"    C* (weryfik. Pc·At/ṁ) = {Cstar_check:.4f} m/s")

print(f"\n  KROK 1 – ve z równania ciągu (zakł. Pe = Pa):")
print(f"    ve = F/ṁ            = {ve:.4f} m/s")
print(f"    Isp = ve/g0         = {Isp:.4f} s")

print(f"\n  KOMORA SPALANIA (parametry spiętrzenia, Mc ≈ 0):")
print(f"    Pc = P0             = {Pc:>14.4f} Pa  =  {Pc/1e5:.4f} bar")
print(f"    Tc = T0             = {Tc:>14.4f} K")
print(f"    ρc = ρ0             = {rho_c:>14.6f} kg/m³")

print(f"\n  KOMORA – dokładna (Ac/At = {EPS_C}, bez zakł. Mc ≈ 0):")
print(f"    Ac                  = {Ac:.6e} m²")
print(f"    Dc                  = {Dc*1e3:.4f} mm")
print(f"    Mc                  = {Mc:.6f}")
print(f"    vc                  = {vc:.4f} m/s")
print(f"    Tc_lok              = {Tc_loc:.4f} K")
print(f"    Pc_lok              = {Pc_loc:.4f} Pa")
print(f"    ρc_lok              = {rho_c_loc:.6f} kg/m³")

print(f"\n  GARDŁO (M = 1):")
print(f"    Pt                  = {Pt:>14.4f} Pa  =  {Pt/1e5:.4f} bar")
print(f"    Tt                  = {Tt:>14.4f} K")
print(f"    ρt                  = {rho_t:>14.6f} kg/m³")
print(f"    vt = at (pr. dźwięku) = {vt:.4f} m/s")
print(f"    At = A*             = {At:.6e} m²")
print(f"    Dt                  = {Dt*1e3:.4f} mm")

print(f"\n  WYLOT DYSZY (Pe = Pa):")
print(f"    Pe                  = {Pe:>14.4f} Pa  =  {Pe/1e5:.6f} bar")
print(f"    Te                  = {Te:>14.4f} K")
print(f"    ρe                  = {rho_e:>14.6f} kg/m³")
print(f"    Me                  = {Me:>14.4f}")
print(f"    ve (isentropowe)    = {ve_isen:>14.4f} m/s  (cel: {ve:.4f} m/s)")
print(f"    Ae                  = {Ae:.6e} m²")
print(f"    Ae/At = ε           = {eps_e:.4f}")
print(f"    De                  = {De*1e3:.4f} mm")

print(f"\n  WERYFIKACJA:")
print(f"    F = ṁ·ve + Ae·(Pe−Pa)")
print(f"      = {M_DOT}×{ve_isen:.4f} + {Ae:.6f}×({Pe:.0f}−{PA:.0f})")
print(f"      = {F_calc:.4f} N   (zadany: {F} N,  błąd: {abs(F_calc-F)/F*100:.4f} %)")

print(f"\n{SEP}")


# ==============================================================================
# 11. OBJĘTOŚĆ KOMORY SPALANIA (metoda L*)
# ==============================================================================
theta_conv = 30.0
theta_rad  = np.radians(theta_conv)

Rc = np.sqrt(Ac / np.pi)
Rt = np.sqrt(At / np.pi)

L_conv = (Rc - Rt) / np.tan(theta_rad)
V_conv = (np.pi / 3) * L_conv * (Rc**2 + Rc * Rt + Rt**2)

L_star_values = [0.60, 0.80, 1.00]

print(f"\n  OBJĘTOŚĆ KOMORY SPALANIA (L* = charakterystyczna długość komory):")
print(f"  {'─'*62}")
print(f"  Promień komory   Rc      = {Rc*1e3:.2f} mm")
print(f"  Promień gardła   Rt      = {Rt*1e3:.2f} mm")
print(f"  Kąt sekcji zbieżn. θ    = {theta_conv:.0f}°")
print(f"  Długość sekcji zbieżn.  = {L_conv*1e3:.2f} mm")
print(f"  Objętość sekcji zbieżn. = {V_conv*1e6:.4f} cm³")
print()
print(f"  {'L* [m]':>8}  {'Vc [cm³]':>10}  {'Vc [L]':>8}  {'V_cyl [cm³]':>12}  "
      f"{'L_cyl [mm]':>11}  {'L_tot [mm]':>11}  {'tau [ms]':>9}")
print(f"  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*12}  {'─'*11}  {'─'*11}  {'─'*9}")

for L_star in L_star_values:
    Vc    = L_star * At
    V_cyl = Vc - V_conv
    L_cyl = V_cyl / Ac if V_cyl > 0 else 0.0
    L_tot = L_cyl + L_conv
    tau   = (rho_c * Vc / M_DOT) * 1e3
    flag  = "  ← nominalne" if L_star == 0.80 else ""
    print(f"  {L_star:>8.2f}  {Vc*1e6:>10.2f}  {Vc*1e3:>8.4f}  {V_cyl*1e6:>12.2f}  "
          f"{L_cyl*1e3:>11.2f}  {L_tot*1e3:>11.2f}  {tau:>9.3f}{flag}")

print(f"\n  Uwaga: L* dla H₂/H₂O₂ wg Sutton RPE: 0.5 – 1.0 m")
print(f"         Zalecana wartość projektowa: 0.8 m")
print(f"\n{SEP}")


# ==============================================================================
# 12. PORÓWNANIE Z WARIANTEM PROJEKTOWYM (MR = 7.94, φ = 2.14)
# ==============================================================================
# Uwaga: ponowne zapytanie do CEA dla wariantu referencyjnego (MR = 7.94)
MR_ref    = 7.94
PHI_ref   = MR_STOICH / MR_ref   # ≈ 2.14

# Pc_ref wyznaczane iteracyjnie
def residual_ref(Pc_bar):
    return ve_isentropic(Pc_bar, MR_ref) - ve

Pc_bar_ref = brentq(residual_ref, 5.0, 14.0, xtol=1e-4, rtol=1e-8)
Tc_ref, Mw_ref, gamma_ref, R_sp_ref = get_cea_params(Pc_bar_ref, MR_ref)
Pc_ref = Pc_bar_ref * 1e5
At_ref = M_DOT / (
    (Pc_ref / (R_sp_ref * Tc_ref))
    * (2 / (gamma_ref + 1)) ** (1 / (gamma_ref - 1))
    * np.sqrt(gamma_ref * R_sp_ref * Tc_ref * 2 / (gamma_ref + 1))
)

print(f"\n  PORÓWNANIE WARIANTÓW (ten sam ciąg F = {F} N, ṁ = {M_DOT} kg/s):")
print(f"  {'─'*62}")
print(f"  {'Parametr':<30}  {'φ = 2.86 (optymalny)':>20}  {'φ = 2.14 (projektowy)':>22}")
print(f"  {'─'*30}  {'─'*20}  {'─'*22}")
print(f"  {'O/F  (MR)':<30}  {MR:>20.4f}  {MR_ref:>22.4f}")
print(f"  {'Tc  [K]':<30}  {Tc:>20.2f}  {Tc_ref:>22.2f}")
print(f"  {'Mw  [g/mol]':<30}  {Mw:>20.4f}  {Mw_ref:>22.4f}")
print(f"  {'γ  [-]':<30}  {gamma:>20.4f}  {gamma_ref:>22.4f}")
print(f"  {'Pc  [bar]':<30}  {Pc/1e5:>20.4f}  {Pc_ref/1e5:>22.4f}")
print(f"  {'At  [mm²]':<30}  {At*1e6:>20.4f}  {At_ref*1e6:>22.4f}")
print(f"  {'Dt  [mm]':<30}  {Dt*1e3:>20.4f}  {np.sqrt(4*At_ref/np.pi)*1e3:>22.4f}")
print(f"  {'Isp [s]':<30}  {Isp:>20.4f}  {ve/G0:>22.4f}")
print(f"\n  Isp jest identyczny (oba warianty: {Isp:.2f} s) – wynika ze stałego")
print(f"  zadania: F = {F} N, ṁ = {M_DOT} kg/s → ve = F/ṁ = {ve:.1f} m/s.")
print(f"  Różnica między wariantami leży w ciśnieniu komory Pc (a zatem w")
print(f"  wymiarach gardła At), temperaturze Tc i składzie spalin Mw.")
print(f"\n{SEP}")
