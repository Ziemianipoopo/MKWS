"""
==============================================================================
OBLICZENIA PARAMETRÓW SILNIKA RAKIETOWEGO: H2 / H2O2 98%
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

ZAŁOŻENIA MODELU:
  1. Przepływ 1D (jednowymiarowy)
  2. Przepływ isentropowy (bez strat tarcia i ciepła)
  3. Parametry komory spalania = parametry spiętrzenia (Mc ≈ 0):
       Pc = P0,  Tc = T0,  ρc = ρ0
  4. Dysza zbieżno-rozbieżna: M < 1 przed gardłem, M = 1 w gardzieli,
     M > 1 za gardłem
  5. Gaz doskonały, stałe właściwości termodynamiczne (γ = const)
  6. Projektowe ciśnienie na wylocie Pe = Pa (dopasowanie do atmosfery)

WZORY (zgodnie z podanym zestawem równań):
  ve   = sqrt(2γ/(γ-1) · R̄/M̄ · T0 · [1 - (Pe/P0)^((γ-1)/γ)])
  T    = T0 · [1 + (γ-1)/2 · M²]⁻¹
  p    = p0 · [1 + (γ-1)/2 · M²]^(-γ/(γ-1))
  ρ    = ρ0 · [1 + (γ-1)/2 · M²]^(-1/(γ-1))
  A/A* = 1/M · [2/(γ+1) · (1 + (γ-1)/2 · M²)]^((γ+1)/(2(γ-1)))
  A*   = ṁ/P0 · sqrt(T0·R̄/γ) · (1 + (γ-1)/2)^((γ+1)/(2(γ-1)))
         (lub prościej: A* = ṁ / (ρt · vt))
==============================================================================
"""

import numpy as np
from scipy.optimize import brentq
from rocketcea.cea_obj import CEA_Obj
from rocketcea.blends import newOxBlend


# ==============================================================================
# 0. STAŁE FIZYCZNE
# ==============================================================================
R_UNIV   = 8314.46    # [J/(kmol·K)] – uniwersalna stała gazowa
G0       = 9.80665    # [m/s²]       – standardowe przyspieszenie ziemskie
PSI_PER_PA = 1/6894.76  # przelicznik Pa → psi (RocketCEA używa psi wewnętrznie)


# ==============================================================================
# 1. DANE WEJŚCIOWE
# ==============================================================================
F         = 701.8     # [N]    – zadany projektowy ciąg silnika
M_DOT     = 0.2       # [kg/s] – zadany strumień masy spalin
PA        = 101325.0  # [Pa]   – ciśnienie otoczenia (1 atm)
MR        = 7.94      # [-]    – stosunek masy utleniacza do paliwa (O/F)
                      #          7.94 ≈ stoichiometria H2/H2O2, daje max Tc
EPS_C     = 8.0       # [-]    – przyjęty stosunek Ac/At dla komory spalania
                      #          typowa wartość dla małych silników rakietowych


# ==============================================================================
# 2. INICJALIZACJA ROCKETCEA
# ==============================================================================
# Tworzymy blend utleniacza: 98% H2O2 + 2% H2O (wagowo)
ox_blend = newOxBlend(oxL=['H2O2', 'H2O'], oxPcentL=[98, 2])

# Obiekt CEA – RocketCEA zwraca wyniki w jednostkach imperialnych (ft/s, psi itp.)
# Przeliczenia wykonujemy ręcznie tam, gdzie potrzebne
cea = CEA_Obj(fuelName='H2', oxName=ox_blend)


# ==============================================================================
# 3. FUNKCJE POMOCNICZE
# ==============================================================================

def get_cea_params(Pc_bar: float, MR: float) -> tuple:
    """
    Pobiera z RocketCEA podstawowe parametry termodynamiczne komory
    dla zadanego ciśnienia Pc i stosunku O/F.

    Parametry wejściowe:
        Pc_bar  – ciśnienie komory [bar]
        MR      – stosunek masy utleniacza do paliwa [-]

    Zwraca:
        Tc    – temperatura w komorze (= T0 spiętrzenia) [K]
        Mw    – masa molowa spalin [g/mol]
        gamma – wykładnik adiabaty (izoentropy) [-]
        R_sp  – właściwa stała gazowa spalin [J/(kg·K)]
    """
    Pc_psia = Pc_bar * 1e5 * PSI_PER_PA        # konwersja bar → psi
    Tc    = cea.get_Tcomb(Pc=Pc_psia, MR=MR)   # [K]
    Mw, gamma = cea.get_Chamber_MolWt_gamma(Pc=Pc_psia, MR=MR)  # [g/mol], [-]
    R_sp  = R_UNIV / Mw                        # [J/(kg·K)] = R_univ / M_molarna
    return Tc, Mw, gamma, R_sp


def ve_isentropic(Pc_bar: float, MR: float) -> float:
    """
    Oblicza isentropową efektywną prędkość wylotową ve dla dyszy
    dopasowanej do Pe = Pa (ekspansja do ciśnienia atmosferycznego).

    Wzór (ze schematu na zdjęciu):
        ve = sqrt( 2γ/(γ-1) · R̄/M̄ · T0 · [1 - (Pe/P0)^((γ-1)/γ)] )

    gdzie T0 = Tc (parametr spiętrzenia), P0 = Pc.
    """
    Tc, Mw, gamma, R_sp = get_cea_params(Pc_bar, MR)
    Pc_Pa = Pc_bar * 1e5
    Pe_Pc = PA / Pc_Pa                         # stosunek Pe/Pc

    if Pe_Pc >= 1.0:
        return 0.0   # Pc musiałoby być niższe niż Pa – fizycznie niemożliwe

    ve = np.sqrt(
        2 * gamma / (gamma - 1)
        * R_sp * Tc
        * (1 - Pe_Pc ** ((gamma - 1) / gamma))
    )
    return ve


def area_ratio(M: float, gamma: float) -> float:
    """
    Stosunek przekroju A do przekroju krytycznego A* (gardła)
    jako funkcja liczby Macha M i wykładnika adiabaty gamma.

    Wzór (ze schematu):
        A/A* = 1/M · [2/(γ+1) · (1 + (γ-1)/2 · M²)]^((γ+1)/(2(γ-1)))

    Wzór obowiązuje zarówno dla przepływu poddźwiękowego (M < 1)
    jak i naddźwiękowego (M > 1).
    """
    factor = (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M**2)
    exponent = (gamma + 1) / (2 * (gamma - 1))
    return (1 / M) * factor**exponent


# ==============================================================================
# 4. KROK 1 – VE Z RÓWNANIA NA CIĄG (założenie Pe = Pa)
# ==============================================================================
# Przy założeniu, że ciśnienie na wylocie Pe = Pa (projektowe dopasowanie dyszy),
# człon Ae*(Pe - Pa) = 0, więc równanie ciągu upraszcza się do:
#
#   F = ṁ · ve   →   ve = F / ṁ
#
# To daje nam efektywną prędkość wylotową – daną do dalszych obliczeń geometrii.
# Później NIE stosujemy założenia Pe = Pa przy obliczaniu Ae, Ac, vc.

ve = F / M_DOT           # [m/s]
Isp = ve / G0            # [s] – specyficzny impuls


# ==============================================================================
# 5. KROK 2 – WYZNACZENIE Pc Z ROCKETCEA (iteracyjnie)
# ==============================================================================
# Szukamy ciśnienia komory Pc takie, by isentropowa ve (wzór z kroku 4)
# zgadzała się z ve wyznaczonym z ciągu.
# Iteracja metodą bisekcji (Brent) w przedziale [5, 14] bar.
#
# Fizyczne uzasadnienie przedziału:
#   – Przy Pc < Pa: nie ma ekspansji (Pa/Pc > 1) – niemożliwe
#   – Przy Pc ~ 5 bar:  ve_iso ≈ 2883 m/s  (za mało)
#   – Przy Pc ~ 15 bar: ve_iso ≈ 3588 m/s  (za dużo)
#   Rozwiązanie jest gdzieś pośrodku.

def residual(Pc_bar):
    return ve_isentropic(Pc_bar, MR) - ve

Pc_bar_sol = brentq(residual, 5.0, 14.0, xtol=1e-4, rtol=1e-8)

# Pobieramy parametry termodynamiczne dla znalezionego Pc
Pc = Pc_bar_sol * 1e5    # [Pa]
Tc, Mw, gamma, R_sp = get_cea_params(Pc_bar_sol, MR)

# Charakterystyczna prędkość wyrzutu C* (z definicji isentropowej, spójna z gamma):
#   C* = sqrt(R·T0/γ) · ((γ+1)/2)^((γ+1)/(2(γ-1)))
# Uwaga: RocketCEA zwraca własne C* w ft/s z innego modelu (uwzględnia
# nieidealne efekty przepływu). Tu używamy wzoru spójnego z przyjętym gamma.
Cstar = np.sqrt(R_sp * Tc / gamma) * ((gamma + 1) / 2) ** ((gamma + 1) / (2 * (gamma - 1)))


# ==============================================================================
# 6. PARAMETRY KOMORY SPALANIA (spiętrzenia, Mc ≈ 0)
# ==============================================================================
# Założenie: Mc ≈ 0 → parametry lokalne w komorze = parametry spiętrzenia
#   Pc = P0,  Tc = T0,  ρc = ρ0
#
# Gęstość z równania stanu gazu doskonałego:
#   ρ = p / (R · T)

rho_c = Pc / (R_sp * Tc)   # [kg/m³] – gęstość spiętrzenia w komorze

# Prędkość vc ≈ 0 (z założenia Mc ≈ 0), dokładna wartość liczymy w kroku 9


# ==============================================================================
# 7. PARAMETRY W GARDZIELI (M = 1, isentropowe)
# ==============================================================================
# W gardzieli M = 1 (warunek krytyczny), co daje:
#
#   T_t = T0 · [1 + (γ-1)/2 · 1²]⁻¹ = T0 · 2/(γ+1)
#   P_t = P0 · (2/(γ+1))^(γ/(γ-1))
#   ρ_t = ρ0 · (2/(γ+1))^(1/(γ-1))
#   v_t = sqrt(γ · R · T_t)   ← prędkość dźwięku w gardzieli = Mach 1
#
# Przekrój gardła z równania ciągłości:
#   ṁ = ρ_t · v_t · A_t  →  A_t = ṁ / (ρ_t · v_t)
#
# Tożsame z wzorem na A* z obrazu:
#   A* = ṁ/P0 · sqrt(T0·R/γ) · ((γ+1)/2)^((γ+1)/(2(γ-1)))
#   (co jest dokładnie C*/Pc · ṁ = A*)

Tt    = Tc * 2 / (gamma + 1)                                    # [K]
Pt    = Pc * (2 / (gamma + 1)) ** (gamma / (gamma - 1))        # [Pa]
rho_t = rho_c * (2 / (gamma + 1)) ** (1 / (gamma - 1))        # [kg/m³]
vt    = np.sqrt(gamma * R_sp * Tt)                              # [m/s]
At    = M_DOT / (rho_t * vt)                                    # [m²]  = A*
Dt    = np.sqrt(4 * At / np.pi)                                 # [m]   – średnica gardła

# Weryfikacja C*: C* = Pc·At/ṁ (powinno zgadzać się z Cstar z kroku 5)
Cstar_check = Pc * At / M_DOT


# ==============================================================================
# 8. PARAMETRY NA WYLOCIE DYSZY (isentropowe, Pe = Pa)
# ==============================================================================
# Ciśnienie na wylocie = atmosferyczne (projektowe dopasowanie):
Pe = PA   # [Pa]

# Temperatura na wylocie (relacja isentropowa):
#   T_e = T0 · (Pe/Pc)^((γ-1)/γ)
Te = Tc * (Pe / Pc) ** ((gamma - 1) / gamma)           # [K]

# Liczba Macha na wylocie (z relacji T/T0):
#   T/T0 = [1 + (γ-1)/2 · M²]⁻¹  →  M = sqrt(2/(γ-1) · (T0/Te - 1))
Me = np.sqrt(2 / (gamma - 1) * (Tc / Te - 1))          # [-]

# Gęstość na wylocie (isentropowa):
#   ρ_e = ρ0 · (Pe/Pc)^(1/γ)
rho_e = rho_c * (Pe / Pc) ** (1 / gamma)               # [kg/m³]

# Prędkość wylotowa (isentropowa) – weryfikacja zgodności z ve z kroku 4:
#   ve = sqrt(2γ/(γ-1) · R · T0 · [1 - (Pe/Pc)^((γ-1)/γ)])
ve_isen = np.sqrt(
    2 * gamma / (gamma - 1) * R_sp * Tc
    * (1 - (Pe / Pc) ** ((gamma - 1) / gamma))
)   # [m/s]

# Stosunek Ae/At z równania obszarów (wzór A/A* z obrazu) przy M = Me:
eps_e = area_ratio(Me, gamma)                            # Ae/At [-]
Ae    = eps_e * At                                       # [m²]
De    = np.sqrt(4 * Ae / np.pi)                         # [m]

# Weryfikacja ciągu: F = ṁ·ve + Ae·(Pe - Pa)
# Przy Pe = Pa człon Ae·(Pe-Pa) = 0, więc F = ṁ·ve (jak w założeniu kroku 1)
F_calc = M_DOT * ve_isen + Ae * (Pe - PA)               # [N]


# ==============================================================================
# 9. PARAMETRY W KOMORZE – DOKŁADNE (bez założenia Mc ≈ 0, z Ac/At = 8)
# ==============================================================================
# Przyjmujemy stosunek przekrojów Ac/At = EPS_C = 8 (typowe dla małych silników).
# Wyznaczamy liczbę Macha w komorze Mc z równania A/A* = Ac/At,
# korzystając z gałęzi PODDŹWIĘKOWEJ (0 < Mc < 1).
#
# Prędkość i temperatura lokalna w komorze (bez założenia spiętrzenia):
#   T_loc = T0 / (1 + (γ-1)/2 · Mc²)    ← ze wzoru na T z obrazu
#   P_loc = P0 · (1 + (γ-1)/2 · Mc²)^(-γ/(γ-1))
#   ρ_loc = P_loc / (R · T_loc)
#   v_c   = Mc · sqrt(γ · R · T_loc)

def mach_residual_subsonic(Mc_):
    """Równanie do wyznaczenia Mc: A/A*(Mc) - eps_c = 0, gałąź poddźwiękowa."""
    return area_ratio(Mc_, gamma) - EPS_C

# bisekcja na gałęzi poddźwiękowej: Mc ∈ (0, 1)
Mc = brentq(mach_residual_subsonic, 1e-8, 1.0 - 1e-10)

Tc_loc  = Tc / (1 + (gamma - 1) / 2 * Mc**2)                           # [K]
Pc_loc  = Pc * (1 + (gamma - 1) / 2 * Mc**2) ** (-(gamma / (gamma - 1)))  # [Pa]
rho_c_loc = Pc_loc / (R_sp * Tc_loc)                                    # [kg/m³]
vc      = Mc * np.sqrt(gamma * R_sp * Tc_loc)                           # [m/s]
Ac      = EPS_C * At                                                     # [m²]
Dc      = np.sqrt(4 * Ac / np.pi)                                       # [m]


# ==============================================================================
# 10. WYNIKI – PODSUMOWANIE
# ==============================================================================

SEP = "=" * 68

print(SEP)
print("  WYNIKI: SILNIK RAKIETOWY H2 / H2O2 98%")
print(SEP)

print("\n  DANE WEJŚCIOWE:")
print(f"    Ciąg          F     = {F} N")
print(f"    Strumień masy ṁ     = {M_DOT} kg/s")
print(f"    Ciśn. otocz.  Pa    = {PA:.0f} Pa")
print(f"    Stosunek O/F  MR    = {MR}")
print(f"    Ac/At (komora)      = {EPS_C}")

print("\n  PARAMETRY TERMODYNAMICZNE (z RocketCEA):")
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
print(f"    vc                  ≈ 0 m/s  (Mc ≈ 0)")

print(f"\n  KOMORA – dokładna (Ac/At = {EPS_C}, bez zakł. Mc ≈ 0):")
print(f"    Ac                  = {Ac:.6e} m²")
print(f"    Dc                  = {Dc*1e3:.4f} mm")
print(f"    Mc                  = {Mc:.6f}")
print(f"    vc                  = {vc:.4f} m/s")
print(f"    Tc_lok (nie spiętrzenia) = {Tc_loc:.4f} K")
print(f"    Pc_lok (nie spiętrzenia) = {Pc_loc:.4f} Pa")
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
# 11. OBJĘTOŚĆ KOMORY SPALANIA
# ==============================================================================
#
# Metodologia: charakterystyczna długość komory L*  (L-star)
# ─────────────────────────────────────────────────────────────────────────────
# L* jest empirycznym parametrem gwarantującym wystarczający czas przebywania
# mieszanki w komorze dla pełnego spalania. Definiuje się go jako:
#
#   L* = Vc / A*   [m]
#
# Wartości typowe (wg Sutton, "Rocket Propulsion Elements"):
#   H₂/O₂           : 0,56 – 0,71 m
#   H₂O₂/HC         : 0,76 – 1,02 m
#   H₂/H₂O₂         : 0,50 – 1,00 m  ← nasz przypadek
#
# Przyjmujemy zakres L* = 0.6, 0.8, 1.0 m i liczymy dla wszystkich trzech.
#
# Geometria komory (cylindryczna + stożkowa część zbieżna dyszy):
# ─────────────────────────────────────────────────────────────────────────────
# Przyjmujemy standardowy kąt półrozwartości sekcji zbieżnej: θ_conv = 30°
#
# Długość sekcji zbieżnej (stożkowej, od Ac do At):
#   L_conv = (Rc - Rt) / tan(θ_conv)
#
# Objętość stożka (truncated cone):
#   V_conv = π/3 · L_conv · (Rc² + Rc·Rt + Rt²)
#
# Objętość cylindrycznej części komory:
#   V_cyl = Vc - V_conv
#
# Długość cylindrycznej części komory:
#   L_cyl = V_cyl / Ac   (musi być > 0; jeśli nie – L* jest za małe)
#
# Całkowita długość komory (od dna do gardła):
#   L_chamber = L_cyl + L_conv

theta_conv = 30.0                          # [°] kąt półrozwartości sekcji zbieżnej
theta_rad  = np.radians(theta_conv)

Rc = np.sqrt(Ac / np.pi)                   # [m] promień komory
Rt = np.sqrt(At / np.pi)                   # [m] promień gardła

L_conv = (Rc - Rt) / np.tan(theta_rad)     # [m] długość sekcji zbieżnej
V_conv = (np.pi / 3) * L_conv * (Rc**2 + Rc * Rt + Rt**2)   # [m³]

# Czas charakterystyczny (residence time) przy L* = 0.8 m (nominalne):
# tau = rho_c * Vc / m_dot  [s]

L_star_values = [0.60, 0.80, 1.00]        # [m]

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
    Vc      = L_star * At                          # [m³]
    V_cyl   = Vc - V_conv                          # [m³] część cylindryczna
    L_cyl   = V_cyl / Ac if V_cyl > 0 else 0.0   # [m]
    L_tot   = L_cyl + L_conv                       # [m]
    tau_ms  = (rho_c * Vc / M_DOT) * 1e3          # [ms] czas przebywania

    flag = "  ← nominalne" if L_star == 0.80 else ""
    print(f"  {L_star:>8.2f}  {Vc*1e6:>10.2f}  {Vc*1e3:>8.4f}  {V_cyl*1e6:>12.2f}  "
          f"{L_cyl*1e3:>11.2f}  {L_tot*1e3:>11.2f}  {tau_ms:>9.3f}{flag}")

print(f"\n  Uwaga: L* dla H₂/H₂O₂ wg Sutton RPE: 0.5 – 1.0 m")
print(f"         Zalecana wartość projektowa: 0.8 m")
print(f"\n{SEP}")
