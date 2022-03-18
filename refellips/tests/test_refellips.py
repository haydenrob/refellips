import numpy as np
import os.path
from refellips import (
    RI,
    Cauchy,
    DataSE,
    ReflectModelSE,
    ObjectiveSE,
    SlabSE,
    StructureSE,
)
from numpy.testing import assert_allclose


def test_bare_against_wvase():
    # Check bare interface ellipsometry calcualtion against wvase
    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "TestData_bareSI.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/void.csv")
    void = RI(_f)

    struc = void() | si()
    assert isinstance(struc[0], SlabSE)
    assert isinstance(struc, StructureSE)

    model = ReflectModelSE(struc)

    wav, aois, psi_d, delta_d = data.data
    wavelength_aoi = np.c_[wav, aois]
    psi, delta = model(wavelength_aoi)

    assert_allclose(psi, psi_d, rtol=0.002)
    assert_allclose(delta, delta_d, rtol=0.003)

    _ = struc.ri_profile()


def test_refellips_against_wvase2():
    # Check Cauchy model with VWASE and theory

    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "Cauchynk_fromWVASE.txt")
    wvase_output = np.loadtxt(dname)

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/void.csv")
    void = RI(_f)

    A = 1.47
    B = 0.00495
    C = 0

    wavs = wvase_output[:, 0]
    refin = A + B / ((wavs / 1000) ** 2) + C / ((wavs / 1000) ** 4)

    cauchy = Cauchy(A=A, B=B, C=C)
    struc = void() | cauchy(1000) | si()
    assert isinstance(struc, StructureSE)

    model = ReflectModelSE(struc)
    model._flip_delta = True

    refellips_RI = []
    for wav in wavs:
        refellips_RI.append(cauchy.complex(wav).real)

    assert_allclose(refellips_RI, refin, rtol=5e-7)
    assert_allclose(refellips_RI, wvase_output[:,1], rtol=5e-7)


def test_refellips_against_wvase3():
    # Three layer stack(cauchy & from file) compared to wvase
    # Also tests behaviour of Delta around 180°

    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "TestData_cauchy.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/void.csv")
    void = RI(_f)

    cauchy = Cauchy(A=1.47, B=0.00495, C=0)
    struc = void() | cauchy(1000) | si()
    assert isinstance(struc, StructureSE)

    model = ReflectModelSE(struc)
    model._flip_delta = True

    wav, aois, psi_d, delta_d = data.data
    wavelength_aoi = np.c_[wav, aois]
    psi, delta = model(wavelength_aoi)

    # assert_allclose(psi, psi_d, rtol=0.0005)
    assert_allclose(delta, delta_d, rtol=0.003)

def test_refellips_against_wvase4():
    # Three layer stack (RI from file) compared to WVASE
    # of a 30 nm SiO2 film with ambient as water.

    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "WVASE_example_30nmSiO2_Water_MultiWavelength.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/silica.csv")
    sio2 = RI(_f)

    _f = os.path.join(pth, "../materials/water.csv")
    h2o = RI(_f)

    struc = h2o() | sio2(300) | si()
    struc.solvent = h2o

    model = ReflectModelSE(struc, delta_offset=0)
    model._flip_delta = True

    wavelength, aoi, d_psi, d_delta = data.data
    psi, delta = model(np.c_[wavelength, np.ones_like(wavelength) * aoi])

    assert_allclose(psi, d_psi, rtol=5e-4)
    assert_allclose(delta, d_delta, rtol=5e-5)


def test_refellips_against_wvase5():
    # Three layer stack (RI from file) compared to WVASE
    # of a 3 nm SiO2 film with 30% VF of water.

    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "WVASE_example_3nmSiO2_30EMA_MultiWavelength.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/silica.csv")
    sio2 = RI(_f)

    _f = os.path.join(pth, "../materials/water.csv")
    h2o = RI(_f)

    silica = sio2(30)
    silica.name = 'Silica'
    silica.vfsolv.setp(value=0.3)

    struc = h2o() | silica | si()
    struc.solvent = h2o
    struc.ema_method = 'linear'

    model = ReflectModelSE(struc, delta_offset=0)
    model._flip_delta = True

    wavelength, aoi, d_psi, d_delta = data.data
    psi, delta = model(np.c_[wavelength, np.ones_like(wavelength) * aoi])

    assert_allclose(psi, d_psi, rtol=6e-4)
    assert_allclose(delta, d_delta, rtol=1e-4)

def test_refellips_against_wvase6():
    # Four layer stack (Cauchy & RI from file) compared to WVASE
    # of a 2 nm SiO2 film and 20 nm polymer film system with
    # 50% VF solvent in the polymer film

    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "WVASE_example_2nmSiO2_20nmPNIPAM_50EMA_MultiWavelength.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/silica.csv")
    sio2 = RI(_f)

    polymer = Cauchy(A=1.47, B=0.00495)
    h2o = Cauchy(A=1.3242, B=0.003064)

    polymer_layer = polymer(200)
    polymer_layer.name = 'PNIPAM'
    polymer_layer.vfsolv.setp(value=0.5)

    struc = h2o() | polymer_layer | sio2(20) | si()
    struc.solvent = h2o

    model = ReflectModelSE(struc, delta_offset=0)
    model._flip_delta = True

    wavelength, aoi, d_psi, d_delta = data.data
    psi, delta = model(np.c_[wavelength, np.ones_like(wavelength) * aoi])

    assert_allclose(psi, d_psi, rtol=6e-4)
    assert_allclose(delta, d_delta, rtol=5e-5)

def test_refellips_against_wvase7():
    # Three layer stack (Cauchy & RI from file) compared to WVASE
    # of a 2 nm SiO2 film and 117 nm gold film in air

    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "WVASE_example_2nmSiO2_117nmAu_MultiWavelength.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/silica.csv")
    sio2 = RI(_f)

    _f = os.path.join(pth, "../materials/gold.csv")
    gold = RI(_f)

    _f = os.path.join(pth, "../materials/air.csv")
    air = RI(_f)

    struc = air() | gold(1170) | sio2(20) | si()

    model = ReflectModelSE(struc, delta_offset=0)
    model._flip_delta = True

    wavelength, aoi, d_psi, d_delta = data.data
    psi, delta = model(np.c_[wavelength, np.ones_like(wavelength) * aoi])

    assert_allclose(psi, d_psi, rtol=3e-5)
    assert_allclose(delta, d_delta, rtol=3e-4)


def test_refellips_against_wvase8():
    # Three layer stack (Cauchy & RI from file) compared to WVASE
    # of a 2 nm SiO2 film and 65 nm aluminium oxide film in air

    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "WVASE_example_2nmSiO2_65nmAl2O3_MultiWavelength.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/silica.csv")
    sio2 = RI(_f)

    _f = os.path.join(pth, "../materials/aluminium_oxide.csv")
    al2o3 = RI(_f)

    _f = os.path.join(pth, "../materials/air.csv")
    air = RI(_f)

    struc = air() | al2o3(650) | sio2(20) | si()

    model = ReflectModelSE(struc, delta_offset=0)
    model._flip_delta = True

    wavelength, aoi, d_psi, d_delta = data.data
    psi, delta = model(np.c_[wavelength, np.ones_like(wavelength) * aoi])

    assert_allclose(psi, d_psi, rtol=4e-5)
    assert_allclose(delta, d_delta, rtol=2e-4)


def test_refellips_against_wvase9():
    # Three layer stack (Cauchy & RI from file) compared to WVASE
    # of a 2 nm SiO2 film and 325 nm aluminium oxide film in air

    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "WVASE_example_10nmSiO2_325nmAl2O3_MultiWavelength.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/silica.csv")
    sio2 = RI(_f)

    _f = os.path.join(pth, "../materials/aluminium_oxide.csv")
    al2o3 = RI(_f)

    _f = os.path.join(pth, "../materials/air.csv")
    air = RI(_f)

    struc = air() | al2o3(3250) | sio2(100) | si()

    model = ReflectModelSE(struc, delta_offset=0)
    model._flip_delta = True

    wavelength, aoi, d_psi, d_delta = data.data
    psi, delta = model(np.c_[wavelength, np.ones_like(wavelength) * aoi])

    assert_allclose(psi, d_psi, rtol=4e-5)
    assert_allclose(delta, d_delta, rtol=2e-4)