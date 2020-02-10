import penquins.models as models

def reducedTunnelModel(vb, gammaL, gammaR, deltaE1, eta,sigma):
    T = 300
    c = 0
    vg = 0   
    gammaC = gammaL*gammaR
    gammaW = gammaL+gammaR
    
    return models.tunnelmodel_singleLevel(vb,gammaC,gammaW, deltaE1,eta,sigma,c,vg,T)

def reducedTunnelModel_NoGauss(vb, gammaL, gammaR, deltaE1, eta):
    T = 300
    c = 0
    vg = 0
    sigma = 0
    gammaC = gammaL*gammaR
    gammaW = gammaL+gammaR
    
    return models.tunnelmodel_singleLevel(vb,gammaC,gammaW, deltaE1,eta,sigma,c,vg,T)
    

def test_compare_with_mathematica():
    f = reducedTunnelModel_NoGauss(4,0.5,0.5,1.1,0.5)
    assert abs((f-0.0000476775)/f) <0.05

def test_compare_with_mathematica2():
    mathematica = [-8.20749E-8, -3.17403E-8, -5.23501E-9, -5.90173E-10,\
                   2.90532E-10, 6.28546E-10, 1.30025E-9,3.02873E-9]
    vb = [-1, -.75, -.5, -.25,.25, .50, .75, 1]
    for i in range(len(vb)):
        f = reducedTunnelModel(vb[i],.0005, 0.02,\
                                                        0.62, 0.7, 0.14)
        assert abs((f-mathematica[i])/f) < 0.05