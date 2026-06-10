# MKWS
Projekt na MKWS


# MKWS – Obliczenia Parametrów Silnika Rakietowego H₂/H₂O₂ 98%

Projekt obliczeniowy wyznaczający parametry termodynamiczne i geometryczne silnika rakietowego zasilanego ciekłym wodorem (H₂) i nadtlenkiem wodoru w stężeniu 98% (H₂O₂ 98%).

---

## Opis projektu

Skrypty realizują pełny cykl obliczeń projektowych silnika rakietowego z dyszą zbieżno-rozbieżną, pracującego w atmosferze (Pe = Pa). Obliczenia oparte są na modelu 1D przepływu isentropowego przy założeniu gazu doskonałego ze stałymi właściwościami termodynamicznymi (γ = const). Parametry termodynamiczne mieszaniny spalin wyznaczane są przy użyciu interfejsu **RocketCEA** (NASA Chemical Equilibrium with Applications).

### Dane projektowe

| Parametr | Wartość |
|---|---|
| Ciąg F | 701,8 N |
| Strumień masy ṁ | 0,2 kg/s |
| Paliwo | H₂ (ciekły wodór) |
| Utleniacz | H₂O₂ 98% (98% H₂O₂ + 2% H₂O wagowo) |
| Ciśnienie otoczenia Pa | 101 325 Pa (1 atm) |
| Stosunek Ac/At (komora) | 8,0 |

---

## Zawartość repozytorium

### Skrypty Python

| Plik | Opis |
|---|---|
| `silnik_rakietowy_H2_H2O2.py` | Główny skrypt obliczeniowy – wyznacza pełne parametry termodynamiczne i geometryczne silnika (Pc, Tc, geometria gardła, wylotu i komory, objętość komory) |
| `silnik_rakietowy_H2_H2O2_phi286.py` | Wariant obliczeniowy dla φ = 2,86 |
| `Isp_vs_phi.py` | Analiza impulsu właściwego Isp jako funkcji współczynnika ekwiwalencji φ przy stałym Pc |
| `analiza_dławienia.py` | Analiza pracy silnika w zakresie dławienia (F = 100–800 N) przy stałej geometrii dyszy – wyznaczanie optymalnego MR*(ṁ), Pc(ṁ), Pe/Pa(ṁ) |
| `iteracja_Pc_phi_opt.py` | Iteracyjne wyznaczenie optymalnej pary (Pc\*, φ\*) maksymalizującej Isp przy zadanym ciągu i strumieniu masy |

### Wykresy

| Plik | Opis |
|---|---|
| `Isp_vs_phi.png` | Impuls właściwy vs. współczynnik ekwiwalencji φ |
| `analiza_dławienia.png` | Wykresy F(ṁ), Pc(ṁ), φ\*(ṁ), Pe/Pa(ṁ) – analiza dławienia |
| `analiza_dlaw_optymalny.png` | Analiza dławienia w punkcie optymalnym |
| `iteracja_Pc_phi_opt.png` | Krzywe Isp(φ) dla różnych Pc oraz linia maksimów φ_opt(Pc) |
| `W1_parametry_napedowe.png` – `W7_gestosci_masy.png` | Zestawy wykresów parametrów napędowych, termodynamicznych, LBV, strumienia masy, ciągu, zestawienia i gęstości |

### Dokumentacja

| Plik | Opis |
|---|---|
| `Raport_Silnik_Rakietowy.docx` | Pełny raport techniczny projektu |

---

## Model obliczeniowy

Obliczenia opierają się na następujących wzorach i założeniach:

**Prędkość wylotowa (isentropowa):**
$$v_e = \sqrt{\frac{2\gamma}{\gamma-1} \cdot \frac{\bar{R}}{\bar{M}} \cdot T_0 \cdot \left[1 - \left(\frac{P_e}{P_0}\right)^{\frac{\gamma-1}{\gamma}}\right]}$$

**Stosunek przekrojów (równanie obszarów):**
$$\frac{A}{A^*} = \frac{1}{M} \left[\frac{2}{\gamma+1}\left(1 + \frac{\gamma-1}{2}M^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}$$

**Przekrój gardła:**
$$A^* = \frac{\dot{m}}{\rho_t \cdot v_t}$$

Wyznaczenie ciśnienia komory Pc odbywa się iteracyjnie metodą bisekcji (Brent) z wykorzystaniem warunku zgodności prędkości wylotowej isentropowej z prędkością wynikającą z równania ciągu.

---

## Wymagania

```
rocketcea
scipy
numpy
matplotlib
```

Instalacja:

```bash
pip install rocketcea scipy numpy matplotlib
```

> **Uwaga:** RocketCEA wymaga zainstalowanego Fortranu oraz biblioteki NASA CEA. Szczegóły instalacji: [https://rocketcea.readthedocs.io](https://rocketcea.readthedocs.io)

---

## Uruchomienie

```bash
# Obliczenia główne (punkt projektowy)
python silnik_rakietowy_H2_H2O2.py

# Analiza Isp vs. φ
python Isp_vs_phi.py

# Analiza dławienia
python analiza_dławienia.py

# Iteracyjna optymalizacja Pc i φ
python iteracja_Pc_phi_opt.py
```

---

## Wybrane wyniki (dla MR = 7,94)

| Parametr | Komora | Gardło | Wylot |
|---|---|---|---|
| Ciśnienie P | ~12,94 bar | ~7,26 bar | 1,013 bar |
| Temperatura T | ~4335 K | ~3626 K | ~1740 K |
| Liczba Macha M | ≈ 0 | 1,0 | ~3,24 |
| Impuls właściwy Isp | — | — | ~357,8 s |

---
