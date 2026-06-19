**17268** 

_J. Phys. Chem. B_ **2006,** _110,_ 17268-17281 

# **Intermolecular Coulomb Couplings from Ab Initio Electrostatic Potentials: Application to Optical Transitions of Strongly Coupled Pigments in Photosynthetic Antennae and Reaction Centers** 

## **M. E. Madjet, A. Abdurahman, and T. Renger*** 

_Freie Uni_ V _ersitaet Berlin, Institut fuer Chemie (Kristallographie), Takustrasse 6 D-14195 Berlin, Germany Recei_ V _ed: March 13, 2006; In Final Form: June 28, 2006_ 

An accurate and numerically efficient method for the calculation of intermolecular Coulomb couplings between charge densities of electronic states and between transition densities of electronic excitations is presented. The coupling of transition densities yields the Fo¨rster type excitation energy transfer coupling, and from the charge density coupling, a shift in molecular excitation energies results. Starting from an ab initio calculation of the charge and transition densities, atomic partial charges are determined such as to fit the resulting electrostatic potentials of the different states and the transition. The different intermolecular couplings are then obtained from the Coulomb couplings between the respective atomic partial charges. The excitation energy transfer couplings obtained in the present TrEsp (transition charge from electrostatic potential) method are compared with couplings obtained from the simple point-dipole and extended dipole approximations and with those from the ab initio transition density cube method of Kru¨ger, Scholes, and Fleming. The present method is of the same accuracy as the latter but computationally more efficient. The method is applied to study strongly coupled pigments in the light-harvesting complexes of green sulfur bacteria (FMO), purple bacteria (LH2), and higher plants (LHC-II) and the “special pairs” of bacterial reaction centers and reaction centers of photosystems I and II. For the pigment dimers in the antennae, it is found that the mutual orientation of the pigments is optimized for maximum excitonic coupling. A driving force for this orientation is the Coulomb coupling between ground-state charge densities. In the case of excitonic couplings in the “special pairs”, a breakdown of the point-dipole approximation is found for all three reaction centers, but the extended dipole approximation works surprisingly well, if the extent of the transition dipole is chosen larger than assumed previously. For the “special pairs”, a large shift in local transition energies is found due to charge density coupling. 

## **Ι. Introduction** 

Half a century ago, Fo¨rster suggested a mechanism, by which an excitation can be transferred between molecules and related the rate constant of this process to the overlap of donor fluorescence and acceptor absorption spectra.[1] According to Fo¨rster, the Coulomb coupling between the electrons of the excited molecule with those of the molecule in the ground state gives rise to the excitation energy transfer and the respective coupling matrix element is given in terms of the coupling between the transition dipole moments of the molecules. This idea had an enormous impact on many areas of physics, chemistry, and biology. In particular, excitation energy transfer in photosynthesis can be understood in molecular detail, since structural information is available. Often the intermolecular distances between photosynthetic pigments (chlorophylls, bacteriochlorophylls, carotenoids) are comparable or even smaller than the extension of the pigments. In this case, on one hand, additional short-range contributions to the matrix element can be expected,[2,3] which rely on electron exchange between the molecules, as was first pointed out by Dexter.[2] On the other hand, the Fo¨rster type coupling matrix element cannot be evaluated in point-dipole approximation (PDA) and more sophisticated methods have to be used. 

* To whom correspondence should be addressed. E-mail: rth@ chemie.fu-berlin.de. 

Since the pioneering work of Weiss[4] and Chang,[5] it became possible to quantify the corrections of the very practical PDA for the Fo¨rster type coupling matrix element. The PDA results from a multipole expansion of the transition densities of the interacting molecules and becomes invalid if the extension of the molecular wave functions is on the same order as the distance between the molecular centers. Whereas for the PDA, just the geometry and strength of the molecular transition dipole elements and the intermolecular distances need to be known, in the case of the more accurate transition monopole approximations (TMAs) of Weiss[4] and Chang,[5] so-called transition monopole charges are introduced which are obtained from - _semiempirical_ quantum chemical calculations, using a Pariser Parr-Pople self-consistent field method. The transition monopoles reflect the transition density of a molecule, which has no classical analogue, since it contains a product of ground and excited-state wave functions, rather than an absolute square of a wave function as a usual charge density. The excitation energy transfer coupling is simply given as the Coulomb coupling between the transition charges of one molecule with those of the other and so can readily be applied, once the transition charges are known. 

An extended analysis that contains the TMA, but includes also exchange of electrons between different molecules, was used by Warshel and Parson[6] and by the Schulten group.[7] 

10.1021/jp0615398 CCC: $33.50 © 2006 American Chemical Society Published on Web 08/04/2006 

_J. Phys. Chem. B, Vol. 110, No. 34, 2006_ **17269** 

Intermolecular Coulomb Couplings 

Warshel and Parson applied the method to study excitonic and charge-transfer couplings in the special pair of the bacterial reaction center of _Rhodopseudomonas_ V _iridis_ and demonstrated that the PDA overestimates the excitonic coupling by about a factor of 5 and that a mixing of excited states with intermolecular charge-transfer states is responsible for the strongly red shifted low-energy absorption band of the special pair. Damjanovic et al.[7] applied the TMA to study excitation energy transfer between an optically dark (one photon forbidden) transition of a carotenoid and a bacteriochlorophyll molecule. Since the distance between the bacteriochlorophyll and the carotenoid is much smaller than their extension, the bacteriochlorophyll transition does not interact with the transition dipole moment of the carotenoid, which is zero, but with individual transition charges. Therefore, excitation energy transfer based on Coulomb interaction can occur and was shown to be more efficient than the alternative Dexter mechanism.[7] 

It was pointed out[8,9] that there is no unique way to determine the atomic partial charges and attempts were made to obtain a more realistic representation of the transition density of the molecules, which should in some way take into account the three-dimensional (3-D) structure of the molecular orbitals. A first improvement for chlorophylls and bacteriochlorophylls was suggested by Sauer and co-workers[10] who used the atomic transition monopole charges from Weiss[4] and split them in half putting one part 1 Å above the macrocycle and one part 1 Å below. This more 3-D representation was found helpful by Pearlstein to explain optical spectra of the photosynthetic reaction center of _Rhodopseudomonas_ V _iridis_ . Sauer et al.[11] applied this method to the light-harvesting complex 2 (LH2) of the purple bacterium _Rhodopseudomonas acidophila_ , where they found a 25% smaller coupling for the strongest coupled bacteriochlorophylls compared with the simple PDA. A semiempirical method that uses analytical two-center repulsion integrals for the p-type orbitals and in this way takes into account the 3-D structure of the transition density was used by Alden et al.[12] for the calculation of optical spectra of the photosynthetical bacterial antenna complex LH2. 

A complete three-dimensional ab initio approach for the calculation of excitation energy transfer couplings was developed by Kru¨ger, Scholes, and Fleming.[9] In their transition density cube (TDC) method, the 3-D space around each molecule is divided into small volume elements, the so-called cubes, and the Coulomb coupling between the transition densities of the molecules is calculated directly from the Coulomb interaction between the cubes of the two molecules. Although on one hand it might be an advantage to use information of ≈500 000 cubes instead of reduced information from ≈50 atoms as in the TMA, the TDC method cannot readily be used by others without repeating the quantum chemical calculations to get the transition density in the cubes. An aim of the present paper is to combine the advantages of both methods, TMA and TDC. We argue that there is a unique way to determine atomic partial charges that takes into account the full 3-D properties of the electronic transition, namely, by fitting the electrostatic potential of the transition density on a 3-D grid using atomic partial charges. It will be shown that it is possible to distill the complete information of the transition density of an electronic transition in atomic partial charges. It is shown that the excitonic couplings obtained from the TDC method, in the limit of small cube volumes, converge with the couplings from the present transition charges from the electrostatic potential (TrEsp) method. As a result, instead of ≈10[10] pairwise cube couplings in the TDC method, the sum runs only over 

≈10[3] pairwise transition charge couplings in the TrEsp method, to evaluate the excitonic coupling between two chlorophylls or bacteriochlorophylls. The gain in numerical efficiency can be used to study the dependence of the excitonic coupling on the mutual orientations of the pigments. 

In photosynthetic pigment-protein complexes, the optical properties of the pigments are influenced, in addition to the excitonic coupling, by (i) charged amino acid residues,[13][-][15] (ii) hydrogen bonds between the pigments and the protein,[16][-][18] (iii) different conformations of the pigments,[12,19] and (iv) the charge density of the protein[20] and other pigments[21] in the local environment. To provide a structure based explanation of the optical transition energies of the pigments in a protein, the socalled site energies, is a challenge,[12,14,15,19] because of the complexity of the above interactions. A purely quantum chemical approach can take into account only a small part of the protein environment of the pigments. 

We have recently presented a simple electrostatic calculation of site energy shifts of the seven bacteriochlorophyll _a_ (BChla) - molecules of the monomeric subunit of the FMO antenna complex of green sulfur bacteria, which included all charged amino acids of the protein as point charges, assuming a standard protonation pattern of the titratable amino acid residues.[14] The electrochromic shifts were obtained from the Coulomb coupling of those point charges with the difference vector of the permanent dipole moments of the BChl’s excited and ground state, estimated from Stark experiments. The present electrostatic potential based determination of atomic partial charges of the ground and excited state of the pigments provides the basis for a quantum chemical/electrostatic calculation of site energies in atomic detail. In a forthcoming publication, we present a description of the site energies of the FMO complex that includes the heterogeneous charge distribution of the whole - pigment protein complex and an average over the possible protonation states of the titratable amino acid residues in the protein. In the present study, the partial charges of the ground and excited state are applied to calculate site energy shifts due to the charge density coupling between neighboring pigments. It will be shown that for some strongly coupled dimers the respective relative shift in transition energies can be comparable or even larger than the excitonic coupling. 

The paper is organized in the following way. In part II, the method is introduced starting from the _N_ -electron wave function and providing an intuitive explanation why the electrostatic potential of the transition or the charge density needs to be fitted by atomic partial charges. Calculations of ab initio transition densities and charge densities and the respective fits by atomic partial charges are presented in part III for bacteriochlorophyll _a_ and chlorophyll _a_ . In part IV, the atomic partial charges are used to investigate Coulomb couplings of strongly coupled pigments of photosynthetic antennae (FMO complex of _Chlorobium tepidum_ ,[22] LH2 complex of _Rhodopseudomonas acidophila_ ,[23] and LHC-II complex of spinach[24] ) and reaction center complexes (purple bacterium _Rhodobacter sphaeroides_ ,[25] photosystem I[26] and photosystem II[27] of cyanobacterium _Thermosynechococcus elongatus_ ). Finally, the results are discussed and summarized in parts V and VI. 

## **II. Method** 

We consider the Coulomb coupling between a molecule A and a molecule B with electronic states _a_ , _a_ ′ and _b_ , _b_ ′ , respectively 

**17270** _J. Phys. Chem. B, Vol. 110, No. 34, 2006_ 

Madjet et al. 

**==> picture [212 x 107] intentionally omitted <==**

Here, the integration is over the spatial coordinates of the _N_ electrons **r** 1, ..., **r** _N_ of molecule A and **r** j1, ..., **r** j _N_ of molecule B. The integration over the respective spin variables is also included but not explicitly denoted, for simplicity. The Coulomb coupling contains the intermolecular coupling between electrons, between electrons and nuclei, and between nuclei. The position of the _I_ th nucleus with atomic number _ZI_ of molecule A is given by **R** molecule _I_ and that _B_ byof **R** hthe _J_ . _J_ th nucleus with atomic number _ZJ_ of 

By using Pauli’s principle for the exchange of electrons and changing names of integration variables, the above matrix element is written as 

**==> picture [241 x 144] intentionally omitted <==**

**==> picture [223 x 15] intentionally omitted <==**

and similarly Fbb(B) ′ (j **r** 1) of molecule B. From the orthogonality of the wave functions of states _a_ and _a_ ′ , it follows 

**==> picture [165 x 15] intentionally omitted <==**

Before we return to the Coulomb coupling, another quantity, the dipole moment **d** a(A) **,** a ′ of molecule A, is introduced30 

**==> picture [201 x 39] intentionally omitted <==**

and is seen, using the same line of arguments as above and the density in eq 3, to equal 

**==> picture [200 x 20] intentionally omitted <==**

The matrix element of the Coulomb coupling in eq 2 is expressed similarly in terms of the densities 

**==> picture [206 x 128] intentionally omitted <==**

The matrix element is written in a more compact form 

**==> picture [229 x 21] intentionally omitted <==**

where the potential _φ_ bb ′ ( **r** ) of molecule B is obtained from 

**==> picture [227 x 67] intentionally omitted <==**

The key idea is to approximate this potential by atomic partial charges _q_ (B) _J_ ( _b_ , _b_ ′ ) as indicated in the second line of eq 9. With this approximation, eq 8 becomes 

**==> picture [218 x 83] intentionally omitted <==**

The expression in the bracket of the first line is expressed similarlyof the atomic partial chargeto eq 9 as the potential **R** h _J_ of molecule B,of molecule _A φ_ at _aa_ (A) ′ the( **r** )position **R** h _J_ ). If (A) ′ the latter is approximated by atomic partial charges _qI_ ( _a_ , _a_ ), the matrix element of the Coulomb coupling is obtained as the Coulomb coupling of atomic partial charges 

**==> picture [184 x 33] intentionally omitted <==**

which are obtained by fitting the electrostatic potentials of molecule B in eq 9 and in the same way for molecule A. 

In the following, Coulomb couplings _V_ 00,00 between the charge densities of the ground states of the molecules, couplings _V_ 01,01 and _V_ 10,10 between the charge densities of the ground state of one molecule and the excited state of the other molecule, and excitation energy transfer couplings _V_ 10,01 ) _V_ 01,10 between the transition densities of the two molecules are calculated. In the latter case, we have _a_ * _a_ ′ and _b_ * _b_ ′ , and therefore, no nuclear contribution occurs as can be seen, for example, in eq 7. 

To recover the standard PDA of excitation energy transfer (excitonic) coupling, the | **r** 1 - **r** j1|[-][1] value in eq 7 is written by expressing **r** 1 relative to the center **R** A of molecule A, that is, **r** 1 ) **R** A + **x** , and similarly j **r** 1 ) **R** B + **x** j for molecule B. Applying a Taylor expansion for small **x** - **x** j values, which assumes that the extension of the molecular transition densities is small compared with the distance _R_ ) | **R** | ) | **R** A - **R** B| between the centers of the molecules, one obtains[30] 

_J. Phys. Chem. B, Vol. 110, No. 34, 2006_ **17271** 

Intermolecular Coulomb Couplings 

**==> picture [241 x 81] intentionally omitted <==**

The above expansion is inserted into eq 7, and the integration variables are changed to **x** for molecule A and **x** j for molecule B. Because the integral over the transition density vanishes (eq 4), only those terms in the above sum that depend on both variables **x** and **x** j survive. Taking into account eq 6 for the transition dipole moment, the excitation energy transfer coupling _V_ 10,01 is obtained in point-dipole approximation as 

**==> picture [192 x 27] intentionally omitted <==**

The simplest possible extension of the above PDA is the extended dipole approximation (EDA). In this case, the excitonic coupling is obtained from eq 11, but by taking into account just two transition charges per molecule[8] 

**==> picture [186 x 32] intentionally omitted <==**

The two transition charges are placed along the direction of the transition dipole moment at a distance that should reflect the extent of the transition density of the molecule. The amount of the two opposite charges is determined such that the resulting dipole moment resembles the experimental transition dipole moment. In the calculations below, the extent of the dipole, that is, the distance between the two charges, will be varied and the amount of the two charges is adjusted for every distance according to the experimental transition dipole moment. 

The excitonic coupling resulting from the atomic transition charges _q_ (A) _I_ (1,0) and _q_ (B) _J_ (0,1), that fit the electrostatic potentials _φ_ (A)10 and _φ_ (B)01 of the transition densities F(A)10 and F(B)01 (eq 9), is obtained from eq 11 as 

**==> picture [185 x 33] intentionally omitted <==**

There exist standard programs for the fitting of the electrostatic potentials in eq 9 by atomic partial charges. In the past, these programs were used for an evaluation of the ground-state molecular potential _φ_ 00. Here, those programs are applied to fit, in addition, the electrostatic potentials _φ_ 11 of the excited state and _φ_ 10 of the optical transition. 

## **III. Electrostatic Potentials of the Ground and Excited State and of the Qy Transition of Chlorophyll** _**a**_ **and Bacteriochlorophyll** _**a**_ 

In the following, we describe how the atomic partial charges for the ground-state S0, the excited-state S1 and the optical Q _y_ transition S0 f S1 of chlorophyll _a_ (Chla) and bacteriochlorophyll _a_ (BChla) are obtained. For this purpose, the electrostatic potential of the charge densities of the S0 and S1 states and the transition density of the S0 f S1 excitation are calculated by - an ab initio method, based on Hartree Fock (HF) and configuration interaction with single excitations (CIS) and, alterna- 

tively, using a time-dependent density functional theory (TDDFT) with the B3LYP exchange correlation (XC) functional. The electrostatic potentials of the densities for the ground and exited state and the optical transition are then fitted by different sets of atomic partial charges, _q_ I(0,0), _q_ I(1,1), and _q_ I(1,0), respectively. 

The calculation scheme consists of the following steps: (i) A geometry optimization is performed by a density functional theory (DFT) method with the B3LYP XC functional and a 6-31G** basis set, using the program Jaguar.[31] (ii) The optimized coordinates are used by the program CHELP-BOW[32] to create a 3-D grid around the molecule. The grid is calculated for 2500 points per atom, sampled randomly from 0 to a maximum distance of 8 Å from any atom. (iii) Electronic structure calculations are carried out with the program QChem[33] based on the optimized coordinates from step (i), providing the electrostatic potential for each point of the 3-D grid of step (ii). (iv) The partial charges for each atom are calculated by a weighted least-squares fit to these potentials by using the CHELP-BOW program.[32] The least-squares equations that appear in the fit are solved with the pseudoinverses calculated with the stable singular-value decomposition method (avoiding rank problems).[34] The charges were constrained to reproduce the total charge zero of the molecule, and the charges of hydrogens were set to be zero. The dipole moments, which are obtained directly from the charge densities (eq 6), are identical with those calculated from the atomic partial charges, **d** a,a(A) ′ ) ∑ _I qI_ ( _a_ , _a_ ′ ) **R** _I_ . More details about the CHELP-BOW method can be found in ref 32. 

It is known that ab initio calculations often do not reproduce the exact magnitude of the transition dipole moments. Therefore, the transition density and transition charges are scaled by the ratio of the experimental to calculated transition dipole moment. Recently, Knox and Spring found a simple empirical relation between the dipole strengths of BChla and Chla and the optical dielectric constant of the solvent.[35] When the results of the linear fits were compared with those obtained in two cavity models, the empty cavity model was found to give the best agreement.[35] Within this model, the vacuum dipole strength of BChla is 6.1 D and the one of Chla is 4.6 D. A slightly larger empty cavity vacuum dipole strength for BChla of 6.3 D was reported earlier by Alden et al.,[12] analyzing the same experimental data as Knox and Spring. The difference perhaps is due to differences in - estimating the contribution of the 0 0 transition to the measured dipole strength. The linear fit of Knox and Spring yields a somewhat larger vacuum dipole strength of BChla of 6.6 D, whereas for Chla a value of 4.5 D was reported, very close to the empty cavity result. In the calculations presented here, we use the empty cavity vacuum dipole moment magnitudes of Knox and Spring, that is, 4.6 D and 6.1 D for Chla and BChla, respectively. 

The empty cavity model was used to estimate the influence of the dielectric medium on the excitonic coupling.[14,36] If two molecules are close, they will not form two separate cavities in the dielectric of the protein, but will rather be situated in the same cavity. Recently, it was demonstrated by Hsu et al.[36] that the excitation energy transfer coupling between two molecules in the same cavity can be either enhanced or decreased by the dielectric, depending on the mutual geometry of the transition dipole moments. For optical dielectric constant _ϵ_ ) 2, the coupling amounts to about 80% of the vacuum coupling for dipoles in sandwich geometry and 110% for in line geometry.[36] In the present work, we do not consider the effect of the dielectric but have in mind the above estimate. 

**17272** _J. Phys. Chem. B, Vol. 110, No. 34, 2006_ 

Madjet et al. 

**Figure 1.** Surface plot of the transition density F10( **r** ) of the S0 f S1 transition of Chla (upper part) obtained with the HF-CIS method (left half) and the TDDFT/B3LYP method (right half). The corresponding electrostatic potentials, _φ_ 10( **r** ), are shown in the middle and lower parts. The middle parts contain surface plots and the lower part contour plots in the mean plane of the molecule. The program Molekel (Fluekiger, P.; Luethi, H. P.; Portmann, S.; Weber, J. _Molekel 4.3_ ; Swiss Center for Scientific Computing: Manno, Switzerland, 2000-2001.) was used for the visualization. Negative values of the densities and potentials are represented by red colors and positive values by blue colors. 

The transition dipole moments calculated with HF-CIS and TDDFT/B3LYP were typically too large by a factor of about 1.6 and 1.3, respectively. In the HF-CIS as well as in the TDDFT/B3LYP calculations of excited-state wave functions, the lowest excited state was identified as the S1 state on the basis of the orientation of the transition dipole moment, which was found along the NB-ND (NI-NIII) axis, as expected. The transition densities F10( **r** ) obtained from the HF-CIS and TDDFT/B3LYP calculations, using a 6-31G* basis set, and the corresponding electrostatic potentials _φ_ 10( **r** ) are shown in Figures 1 and 2 for Chla and BChla, respectively. The use of a larger basis set has only a very minor influence on the results, whereas a smaller basis set yields larger deviations, as shown in the Supporting Information. In particular, the electrostatic potential in Figures 1 and 2 shows nicely the direction and extent of the - transition dipole along the NB ND axis of the two molecules. We note that the transition density and hence the transition dipole moment can only be determined up to a factor of minus one. However, a rotation of any transition dipole by 180 ° has no influence on the optical spectrum of a system of excitonically 

coupled pigments and therefore is not significant. The atomic transition charges that result from the fit of the electrostatic potentials in Figures 1 (Chla) and 2 (BChla) are given in the Supporting Information. 

The electrostatic potentials _φ_ 00( **r** ) and _φ_ 11( **r** ) of the ground and excited-state charge densities, respectively, are obtained in a similar way. In Figures 3 and 4, the differences in charge density ∆F( **r** ) ) F11( **r** ) - F00( **r** ) between the excited and the ground state and the corresponding difference in electrostatic potentials ∆ _φ_ ( **r** ) ) _φ_ 11( **r** ) - _φ_ 00( **r** ) are shown for Chla and BChla, respectively. The respective partial charges, that fit the potentials of the ground and excited state, are also given in the Supporting Information. Similar orientations of the difference dipole moment vector ∆ **d** ) **d** 11 - **d** 00 are obtained for Chla and BChla with HF-CIS and TDDFT/B3LYP. The ∆ **d** vector points from ring I (NB) to ring III (ND). A maximum deviation from this direction of 34 ° away from ring V is obtained for Chla with HF-CIS, whereas with TDDFT/B3LYP this vector is found to be rotated by 16 ° in the opposite direction. A magnitude of ∆ _d_ ) 0.6 D is obtained with TDDFT/B3LYP and 

_J. Phys. Chem. B, Vol. 110, No. 34, 2006_ **17273** 

Intermolecular Coulomb Couplings 

**Figure 2.** Surface plot of the transition density F10( **r** ) of the S0 f S1 transition of BChla (upper part) obtained with the HF-CIS method (left half) and the TDDFT/B3LYP method (right half). The corresponding electrostatic potentials, _φ_ 10( **r** ), are shown in the middle and lower parts. The middle parts contain surface plots and the lower part contour plots in the mean plane of the molecule. Negative values of the densities and potentials are represented by red colors and positive values by blue colors. 

∆ _d_ ) 1.2 D with HF-CIS. The respective values for BChla are ∆ _d_ ) 1.3 D with TDDFT/B3LYP and ∆ _d_ ) 2.8 D with HFCIS. For BChla, the ∆ **d** vector is rotated in the same direction by angles of 8 ° and 18 ° in HF-CIS and TDDFT/B3LYP, respectively, away from ring V. 

## **IV. Application to Pigment Dimers of Photosynthetic Antennae and Reaction Centers** 

**A. Excitonic Couplings.** The present method is applied to study Coulomb couplings between the strongest coupled pigments in several light-harvesting and reaction center complexes. In Figure 5, the structure of the BChla dimer in the R- B subunit of the LH2 complex[23] and of the special pair of the reaction center of purple bacteria (bRC)[25] are shown. The respective excitonic couplings _V_ 10,01 ) _V_ 01,10 are calculated with the TrEsp method described above and are compared with results of PDA and EDA. To investigate the validity of the latter approximations, different configurations of the pigment dimers are investigated that are obtained from the native configuration either by an in-plane rotation of one pigment around its center or a translation of the pigment along a direction normal to the plane of the other pigment, as indicated in Figure 5. The 

corresponding results are shown in Figure 6. Whereas for the LH2 complex the PDA provides at least a qualitatively correct picture, in the case of the special pair of bRC it breaks down. At the native orientation of the latter, that is, at rotation angle zero, the point-dipole coupling is about five times larger than the TrEsp result. The maximum coupling does not occur at the native orientation but at an angle of about 30 ° , whereas the lightharvesting pigments in LH2 are arranged in a position of strongest excitonic coupling. If extended dipoles are used for the calculation of the coupling in the special pair of bRC, the extent of the dipole can be adjusted (8.8 Å, | _q_ 10| ) 0.15 e) to give an almost quantitative agreement with the TrEsp values over the whole range of rotation angles and interplane distances. In the case of LH2, the extent of the transition dipole has to be chosen somewhat larger (10.2 Å, | _q_ 10| ) 0.13 e). 

Another antenna system that uses BChla molecules for light - - harvesting is the so-called Fenna Matthews Olson (FMO) complex of green sulfur bacteria.[22] The strongest coupling in the FMO complex occurs between BChl1 and BChl2. In the upper part of Figure 7, the excitonic coupling is shown with dependence on the in-plane rotation angle of BChl1. In the present case, the PDA agrees with the exact TrEsp result at all 

**17274** _J. Phys. Chem. B, Vol. 110, No. 34, 2006_ 

Madjet et al. 

**Figure 3.** Surface plot of the difference in charge density ∆F( **r** ) ) F11( **r** ) - F00( **r** ) between the excited-state S1 and the ground-state S0 of Chla (upper part) obtained with HF-CIS (left half) and TDDFT/B3LYP (right half). The corresponding electrostatic potentials ∆ _φ_ ( **r** ) ) _φ_ 11( **r** ) - _φ_ 00( **r** ) are shown in the middle and lower parts. The middle parts contain surface plots and the lower part contour plots in the mean plane of the molecule. Negative values of the densities and potentials are represented by red colors and positive values by blue colors. 

rotation angles. As in the case of LH2, the largest coupling is obtained for zero rotation, that is, at the native geometry. To elucidate the driving force for reaching the geometry of maximum excitonic coupling, we calculated the Coulomb coupling between the ground-state charge densities of the two pigments as a function of the in-plane rotation angle. Indeed, as shown in the lower part of Figure 7, both, DFT and HF calculations yield the largest stabilization of the dimer for the native geometry. 

We turn now to those antenna and reaction center complexes - which contain Chla. In Figure 8, the Chla dimer Chla611 Chla612 (in the nomenclature of Liu et al.) of the LHC-II from spinach[24] is shown together with the “special pairs” of the reaction centers of photosystem I[26] (PS-I) and photosystem II[27] (PS-II) from _T. elongatus_ . In the case of PS-I, the “special pair” consists of a chlorophyll _a_ (PB) and an epimer chlorophyll _a_ ′ (PA). The same type of analysis of the excitonic coupling as for the BChla complexes was performed. The resulting excitonic couplings are shown in Figure 9. Whereas for the LHC-II dimer, the TrEsp values are reproduced by PDA and EDA, in the case of the reaction center “special pairs”, again strong deviations of the point-dipole couplings are found. As for FMO and LH2, the 

geometry of the pigment dimer in LHC-II is found to be optimized for strong excitonic coupling. In contrast, for the “special pairs” of the reaction centers of PS-I and PS-II the strongest excitonic coupling occurs at a non-native geometry. The EDA results resemble the TrEsp calculations over the whole range of rotation angles. The extent of the dipole was chosen such as to provide a good description of the TrEsp results. It amounts 8.7 Å (| _q_ 10| ) 0.11 e) for the “special pairs” of PS-I and PS-II, that is, very close to the 8.8 Å found above for the bacterial reaction center. 

The values obtained for the excitonic coupling _V_ 10,01 ) _V_ 01,10 at the native geometry for all systems studied are summarized in Table 1. 

**B. Comparison with Other Methods.** The “special pair” of the PS-I reaction center was chosen to compare different methods for the calculation of excitonic couplings. To investigate, in addition, the effect of the different pigment conformations on the transition charges, a partial geometry optimization (keeping some torsional angles constant), was carried out for the two pigments PA and PB, and the transition densities and partial charges, resulting from those densities, were obtained separately for each pigment. The values of those charges are 

_J. Phys. Chem. B, Vol. 110, No. 34, 2006_ **17275** 

Intermolecular Coulomb Couplings 

**Figure 4.** Surface plot of the difference in charge density ∆F( **r** ) ) F11( **r** ) - F00( **r** ) between the excited-state S1 and the ground-state S0 of BChla (upper part) obtained with HF-CIS (left half) and TDDFT/B3LYP (right half). The corresponding electrostatic potentials ∆ _φ_ ( **r** ) ) _φ_ 11( **r** ) - _φ_ 00( **r** ) are shown in the middle and lower parts. The middle parts contain surface plots and the lower part contour plots in the mean plane of the molecule. Negative values of the densities and potentials are represented by red colors and positive values by blue colors. 

**Figure 5.** Structure of pigment dimers BChlaR-BChla B of the R- B subunit of the light-harvesting complex LH2[23] (file 4RC ‚pdb) and the special pair PL-PM in the bacterial reaction center[25] (file 1NKZ‚ pdb). One of the two BChls is rotated in plane and translated as indicated in the calculation of the excitonic couplings in Figure 6. 

given in the Supporting Information. The couplings were calculated as functions of the interplane distance, where PA was translated along the normal of the plane of PB starting with the native dimer. The coupling that results from the TDC method for different grid sizes is compared in Figure 10 to the TrEsp coupling. The TDC couplings converge against the TrEsp values for grid sizes smaller than 0.2 Å, which corresponds to a total number of about 500 000 cubes used per pigment. In both calculations, a HF-CIS quantum chemical computation of the transition density was used. TDDFT/B3LYP quantum chemical calculations yield the same agreement between TrEsp and TDC 

in the limit of small cubes, as shown in the Supporting Information. We note that, in the TDC method, if the cubes are chosen too large, the resulting transition dipole moment vector can deviate strongly from the vector obtained directly from the transition density. For example, an angle of 28 ° results between the two vectors for the largest cube size ( _δ_ ) 0.4 Å) in Figure 10, explaining the deviations in the excitonic coupling obtained between TDC and TrEsp, at large distances, where the PDA can be assumed to be valid. 

In Figure 11, excitonic couplings calculated with TMA with transition monopole charges from Chang[5] and Weiss[4] are compared with couplings obtained with TrEsp atomic partial charges for planar Chla using HF-CIS and, alternatively, TDDFT/B3LYP. The couplings are again shown as functions of the in-plane rotation angle of PA. The transition monopole charges of Chang[5] and Weiss[4] give similar couplings. If BChla is used as a model for Chla as was done in ref 37, that is, if instead of partial charges of Chla those of the corresponding atoms of BChla are used, the coupling at the native angle decreases and a qualitative difference in the dependence on the rotation angle is obtained. Instead of two maxima at around 25 ° and 125 ° just a single maximum of the coupling results at 

**17276** _J. Phys. Chem. B, Vol. 110, No. 34, 2006_ 

Madjet et al. 

**Figure 6.** Excitonic coupling for the BChla dimers of LH2 and bRC shown in Figure 5 obtained by using TrEsp with TDDFT/B3LYP (thick solid line) and HF-CIS (thin solid line), point-dipole approximation (dotted line), and extended dipole approximation (dashed line, dipole extent, 10.2 Å for LH2 and 8.8 Å for bRC) as functions of the inplane rotation angle of one BChla relative to the native position at R ) 0 and the plane-to-plane distance. 

**Figure 9.** Excitonic coupling for the Chla dimers of LHC-II and the reaction centers of PS-I and PS-II shown in Figure 8 obtained by using TrEsp with TDDFT/B3LYP (thick solid line) and HF-CIS (thin solid line), point-dipole approximation (dotted line), and extended dipole - - approximation (dashed line, dipole extent, LHC II, 9.6 Å; PS I, 8.7 - Å; PS II, 8.7 Å) as functions of the in-plane rotation angle of the first Chla relative to the native orientation (R ) 0) and the plane-to-plane distance. 

**Figure 10.** Comparison of excitonic couplings obtained for the special pair in photosystem I in the transition density cube (TDC) method with those obtained with the TrEsp method. The couplings are shown as a function of the interplane distance of the two pigments, which was increased as explained in the text, starting with the native structure. The results of the TDC calculations are shown for different cube sizes _δ_[3] . A geometry optimization constraining some torsional angles was performed for the two pigments. The transition densities were obtained from HF-CIS quantum chemical calculations. The respective TDDFT/ B3LYP results are shown in the Supporting Information. 

**Figure 7.** Excitonic coupling _V_ 10,01 (upper part) and ground-state charge density coupling _V_ 00,00 (lower part) of BChl1 and BChl2 of the FMO complex of _C. tepidum_[22] (file 1M50.pdb) depending on the in-plane rotation angle of BChla1 relative to the native geometry. 

**Figure 8.** Structure of pigment dimers Chla611-Chla612 of the lightharvesting complex LHC-II[24] (file 1RWT.pdb) and of the “special pairs” in the reaction centers of photosystem I (PS-I)[26] (file 1JB0.pdb) and photosystem II (PS-II)[27] (file 2AXT.pdb). 

about 75 ° . The TrEsp couplings obtained from TDDFT/B3LYP calculations behave qualitatively similar to the TrEsp/HF-CIS results and the TMA couplings of Weiss and Chang, but the coupling at the native geometry becomes almost zero in TDDFT/ B3LYP (a somewhat larger value of 30 cm[-][1] results for this coupling if the conformations of the pigments are taken into 

**TABLE 1: Excitonic Couplings in Units of cm**[-] **[1] Obtained with TrEsp Using HF-CIS and TDDFT/B3LYP Methods for the Strongest Coupled Pigments in Different Antennae and for the Special Pairs in Three Reaction Centers** _**[a]**_ 

||||_V_10,01|
|---|---|---|---|
|complex|dimer|HF-CIS|TDDFT/B3LYP|
|FMO|BChl1-BChl2|-130|-125|
|LH2|BChlR-BChl<br>;|245|211|
|bRC|PM-PL|136|91|
|LHC-II|Chla611-Chla612|167|136|
|PS-II|PD1-PD2|92|68|
|PS-I|PA-PB|51|1|



_a_ The transition charges were rescaled to yield a transition dipole moment magnitude of 4.6 and 6.1 D for Chla and BChla, respectively. 

account in the determination of the transition charges, as shown in the Supporting Information). 

**C. Site Energy Shifts.** Finally, the influence of the Coulomb coupling between charge densities of the pigments on their local transition energies is investigated. The shift _E_ in the transition energy of a pigment due to the presence of the second pigment, 

_J. Phys. Chem. B, Vol. 110, No. 34, 2006_ **17277** 

Intermolecular Coulomb Couplings 

**Figure 11.** Excitonic coupling in the special pair of photosystem I depending on the in-plane rotation angle of PB relative to the native orientation, obtained using different atomic partial charges as explained in detail in the text. 

in first-order perturbation theory, is obtained as the difference between Coulomb couplings of the excited-state charge density of the pigment with the ground-state charge density of the neighboring pigment and the Coulomb coupling of the groundstate charge densities of the two pigments, that is, _E_ ) _V_ 10,10 - _V_ 00,00. The relative shift in site energies of the two pigments ∆ _E_ ) _E_ 1 - _E_ 2 is then obtained as 

**==> picture [162 x 11] intentionally omitted <==**

The above quantity is shown in Table 2 for the dimers investigated, together with the values for _V_ 10,10 and _V_ 01,01, obtained with HF-CIS and with TDDFT/B3LYP partial charges. In the case of light-harvesting antennae, the excitonic couplings in Table 1 are larger than the relative shifts in site energies in Table 2. For the “special pairs” of the reaction centers, the ∆ _E_ values are in the same range or even larger than the excitonic couplings. In general, the deviations between the HF-CIS and the TDDFT/B3LYP results are smaller for the excitonic couplings and larger for the charge density couplings and the resulting relative shifts in site energies. 

## **V. Discussion** 

**A. Calculation of Atomic Partial Charges.** Atomic partial charges derived from electrostatic potentials of molecular ground states are widely used in molecular dynamics and Monte Carlo calculations as well as in electrostatic calculations of physical properties of condensed phase matter.[38,39] Those partial charges provide a more accurate description of the charge density coupling than, for example, the alternative Mulliken charges.[40] In the present study, the idea to represent the electrostatic potential by atomic partial charges is extended to excited states and to optical excitations of molecules. It is known that the atomic partial charges depend strongly on the way the potential points are selected.[32] In the present work, we used the CHELPBOW method[32] where the grid points are selected randomly around the molecule to avoid this dependence. The random choice of the grid points was found to make the determination of partial charges very efficient. We have compared the CHELPBOW program with the widely used RESP program of Kollman and co-workers[41] using the random grid for the electrostatic potential and found that, although there are differences in the partial charges, both programs give very similar results for the Coulomb couplings obtained from those charges. The advantage of CHELP-BOW is that it provides the random grid. 

By comparing the different charge densities and the corre- sponding electrostatic potentials in Figures 1 4, it is seen that the potentials provide additional physical insights. For example, the shape of the electrostatic potential of the transition densities in Figures 1 and 2 suggests the use of an EDA in the calculation 

of excitonic couplings. The electrostatic potential of the difference in charge densities of the excited and ground state in Figures 3 and 4 directly reflects the electrochromic shift of the excitation energy that would result from a charge in the neighborhood of the pigment. The red and blue regions of the potential represent negative and positive values, respectively. An external positive charge near ring I or a negative charge near ring III will therefore lead to a decrease in energy, that is, an electrochromic red shift. Opposite external charges, of course, lead to blue shifts. The same conclusion was drawn earlier from semiempirical calculations of the transition energy depending on the position of an external charge above the BChla macrocycle.[13,42] The present method provides this information directly without external charge, that is, from a single calculation. 

We note that the direction of the calculated difference dipole ∆ **d** ) **d** 11 - **d** 00 does not contain the uncertainty of a 180 ° rotation as the transition dipole moment. The experimental information about the direction of ∆ **d** is obtained from Stark experiments, which measure the angle between ∆ **d** and the optical transition dipole moment **d** 10. For BChla, an angle of 12 ° was reported,[43] which is between the 8 ° calculated by HFCIS and 18 ° obtained with TDDFT/B3LYP. For Chla, an angle of 20 ° was measured,[44] and values of 34 ° and 16 ° were obtained with HF-CIS and TDDFT/B3LYP, respectively. However, in contrast to the BChla calculations in Figure 4, the two quantum chemical calculations for Chla provide ∆ **d** vectors in Figure 3 that differ in the direction of the rotation with respect to **d** 10, - that is, the NB ND axis. Unfortunately, the Stark experiments cannot discriminate between these two directions, and so, at present it is unclear which calculation is more reliable. The present difference in the TDDFT/B3LYP and HF-CIS calculations could reflect larger dynamical correlations of electrons in Chla, which are not well described in HF-CIS. Additional wave function based calculations, including more dynamic correlation, and TDDFT calculations with different XC functionals are planned to investigate this point further. 

**B. Excitonic Couplings.** The present TrEsp method of obtaining transition charges from the electrostatic potential of the transition density allows us to use ab initio quantum chemical - - methods instead of the _semiempirical_ Pariser Parr Pople method used before[4][-][7,10,11] to obtain atomic transition monopoles. Whereas the earlier method did not contain the full 3-D information due to a reduction of the transition density to atomic expansion coefficients of the molecular orbitals involved in the transition, here the atomic partial charges reflect directly the 3-D electrostatic potential of the transition density. This potential appears in the derivation of the Coulomb coupling between two molecular states of a dimer, and we argue that it is, therefore, the most natural and unique choice to define atomic partial charges of an electronic transition. Further support for this argument is obtained by comparing the present TrEsp method with the TDC method.[9] In the latter, the integral of the Coulomb coupling between the transition charges of the molecules is solved by performing a double summation over the volume elements (cubes) containing the transition densities. Of course this method contains the exact coupling in the limit of zero cube volume. 

In practical applications on chlorophylls and bacteriochlorophylls, about 500 000 cubes with a volume of about (0.2 Å)[3] have to be used to reach convergence.[9] It is shown in Figure 10 that the relevant information about the transition density contained in 500 000 cubes of a chlorophyll or bacteriochlorophyll can be stored in just 82 atomic partial charges (given 

**17278** _J. Phys. Chem. B, Vol. 110, No. 34, 2006_ 

Madjet et al. 

**TABLE 2: Charge Density Couplings** _**V**_ **10,10 (** _**V**_ **01,01) between the Excited State of the Left (Right) Pigment and the Ground State of the Right (Left) Pigment in Column 2 Using HF-CIS and TDDFT/B3LYP Methods** _**[a]**_ 

|complex<br>dimer|_V_10,10|_V_01,01|∆_E_)_V_10,10-_V_01,01|
|---|---|---|---|
||HF-CIS<br>TDDFT|HF-CIS<br>TDDFT|HF-CIS<br>TDDFT|
|FMO<br>BChl1-BChl2<br>LH2<br>BChlR-BChl_�_<br>bRC<br>PM-PL<br>LHC-II<br>Chla611-Chla612<br>PS-II<br>PD1-PD2<br>PS-I<br>PA-PB|-981<br>-758<br>-247<br>-165<br>474<br>416<br>-1249<br>-949<br>258<br>373<br>409<br>433|-900<br>-754<br>-287<br>-186<br>418<br>356<br>-1270<br>-935<br>90<br>265<br>451<br>529|-81<br>-4<br>40<br>21<br>56<br>60<br>21<br>-14<br>168<br>108<br>-42<br>-96|



_a_ The last two columns show the resulting shifts in transition energies. The couplings and shifts are given in units of cm-1. 

in the Supporting Information) by fitting the electrostatic potential of the transition charges. The couplings obtained in the TDC method converge against those of the TrEsp method for small cube volume. The TrEsp method is numerically more efficient than the TDC method, since instead of 500 000 × 500 000 cube couplings just 82 × 82 couplings of atomic partial charges have to be evaluated. For example, the evaluation of the couplings in Figure 10 takes about 20 h with TDC and 10 min with TrEsp on the same PC. Since the quantum chemical calculation is identical in both methods, the computation time for the latter was not included in this estimate. In the case of TrEsp, 10 min are needed to determine the partial charges of the two pigments, and 20 h in TDC are used to sum over the cubes. We note that the close agreement between the TrEsp and TDC method is not restricted to HF-CIS calculations but can be expected to be valid for any quantum chemical method. In the Supporting Information, a comparison is given for TDDFT-B3LYP calculations. 

Another advantage of TrEsp is that the partial charges, given in the Supporting Information, that were evaluated for an idealized planar Chla or BChla can be applied to the pigments found in the X-ray structures without the need to repeat the quantum chemical calculations for every pair of pigments, provided that the conformational differences between pigments in different sites can be neglected. If the conformational variations are important, then a partial optimization of the crystal geometries is necessary, because the procedure used by the crystallographers to fit the electron density does not include a geometry optimization in the quantum chemical sense. The value of 51 cm[-][1] obtained for the coupling between PA and PB of PS-I by using the partial charges for the planar chlorophyll (obtained in HF-CIS) is close to the value of 64 cm[-][1] in Figure 10 obtained by optimizing the structure with constraints, keeping the torsional angles as in the crystal structure. We note that some influence of the conformation on the coupling is included also, when the partial charges of the planar chlorophyll are used, simply because those charges are placed at the coordinates of the respective atoms in the crystal structures. 

The atomic partial charges of the planar Chla and BChla were applied in the calculations of excitonic couplings of the reaction center “special pairs” and of the excitonic coupling in dimers containing the strongest coupled pigments of three photosynthetic antennae. The efficient TrEsp method made it easy to vary the mutual orientation and plane-to-plane distance of the two pigments in the dimer and to check deviations of the couplings from those obtained in point-dipole approximation. The point-dipole approximation is found to give a reasonable description of the excitonic couplings in photosynthetic anten- nae. The largest deviations of 30 40% (depending on the quantum chemical method) are obtained for the coupling between the bacteriochlorophylls of the R- and _�_ -subunits of the LH2 complex. The deviations are similar to the 25% reported earlier by Sauer[11] and co-workers. From an analysis of the upper 

exciton component of the so-called B850 bacteriochlorophylls of the LH2 complex, using a B800-free mutant, Koolhaas et al.[45] reported a coupling of about 300 cm[-][1] , based on an analysis of circular dichroism spectra. This value was supported later also by additional experiments and modeling of absorption and circular dichroism experiments of LH2 complexes from various bacteria by Georgakopoulou, van der Zwan, and co-workers.[46] The present TrEsp results of 245 and 211 cm[-][1] obtained with HF-CIS and TDDFT/B3LYP, respectively, are somewhat smaller than the above value. A possible reason for the deviation might be the uncertainty of the vacuum dipole strength of bacteriochlorophyll _a_ . If for the latter instead of 6.1 D, of the empty cavity model,[35] a value of 6.6 D, obtained by applying a linear fit,[35] is used, the HF-CIS coupling amounts 287 cm[-][1] . Another possible source of additional excitonic coupling is given by short-range exchange contributions. Scholes et al.[47] carried out monomer as well as dimer calculations using an HF-CIS method and inferred a short-range coupling of about 55 cm[-][1] . A Fo¨rster type excitonic coupling of 265 cm[-][1] was calculated[47] by the TDC method, rescaling the transition density to yield a transition dipole strength of 6.38 D. For this dipole strength, the TrEsp/ HF-CIS value for the coupling increases from 245 to 268 cm[-][1] , that is, to the value obtained with the TDC method.[47] A similar value (254 cm[-][1] ) was reported by Alden et al.,[12] on the basis of a transition monopole method and simulations of linear absorption and circular dichroism spectra. 

The influence of the polarization of the protein on the excitonic couplings[14,36,48] has been neglected here, but it can lead to a decrease as well as to an increase of the coupling,[36] as discussed before. In the case of the FMO complex, we have carried out electrostatic calculations recently,[14] where the BChls were treated as cavities in the dielectric of the protein (with optical dielectric constant _ϵ_ ) 2). The Coulomb couplings between the transition monopole charges of Chang[5] in the cavities of the strongly coupled pigments were smaller than the respective vacuum couplings by a factor of about 0.8. The PDA was found to be valid,[14] in agreement with the present result. 

An interesting aspect of the present analysis is the dependence of the excitonic couplings on the mutual orientation of the strongly coupled pigments. The orientation appears to be optimized for strong excitonic coupling. This result applies to the dimers of the LH2 and the FMO complex in Figures 6 and 7 and also the LHC-II complex in Figure 9. An obvious way to achieve this orientation is given by the Coulomb coupling between the ground-state charge densities of the pigments. As seen in Figure 7, this coupling leads to a stabilization of the ground-state geometry, which provides maximum excitonic coupling. 

In the case of the reaction center “special pairs”, the PDA breaks down and the orientation of the pigments is not optimized for strong excitonic coupling. This result may not be surprising, since the primary task of reaction centers is to transfer charges 

_J. Phys. Chem. B, Vol. 110, No. 34, 2006_ **17279** 

Intermolecular Coulomb Couplings 

and not excitations. The structure of the electrostatic potential of the transition charges in Figures 1 and 2 explains why the extended dipole approximation gives such excellent results for the excitonic couplings over the whole range of mutual orientations and interplane distances. The extent of about 9 Å, obtained for all three reaction centers, is slightly smaller than the 10 Å found for LH2. This result implies that the extent depends somewhat on the dimer geometry and thus limits the practical use of an EDA. However, it can be expected that the value of 9 Å provides a good first estimate for the deviations from a PDA. This extent is significantly larger than the 6 Å assumed previously.[8] We note that a systematic investigation of the EDA was not attempted before. The large factor of about 5 between the point-dipole coupling and the TrEsp value for the bacterial reaction center is in line with the earlier studies of Warshel and Parson.[6] 

Concerning the coupling between the two “special pair” chlorophylls PA and PB of photosystem I reaction centers, there exist controversial results in the literature. A value of -48 cm[-][1] - - obtained with a Pariser Parr Pople transition monopole approach by Sener, Schulten, and co-workers[37] contrasts with a value of 141 cm[-][1] obtained by Damjanovic, Fleming, and coworkers[15] with the TDC method using HF-CIS. We note that the sign inversion reflects the fact that for the first coupling the high-energy exciton state of the dimer carries most of the oscillator strengths, whereas in the second case the low-energy exciton state does. Such a switch has a strong influence on the optical spectra and therefore needs to be understood. The following points were investigated: (i) How does the coupling depend on the recent corrections[49] of the structure of PA, which was modeled in the original PDB file 1JB0.pdb by mistake as a chlorophyll _a_ and not as the epimer chlorophyll _a_ ′ ?[49] (ii) How do the conformations of the pigments and different levels of geometry optimization influence the coupling calculated? (iii) How does the transition monopole coupling of Sener et al.[37] - - compare with other Pariser Parr Pople transition monopole calculations? (iv) The validity of the approach[37] to use a chlorophyll analogue, which resembles bacteriochlorophyll, for the calculation of couplings between chlorophylls. 

Concerning the corrections in the structure, we found only minor changes in the couplings. The value of 141 cm[-][1] obtained in the TDC method using HF-CIS wave functions[15] can be reproduced, if no geometry optimization is carried out and if the transition charges are not rescaled to match the experimental dipole strength. The TrEsp/HF-CIS coupling obtained by using a geometry optimization with constraints (for the torsional angles) in Figure 10 (at native distance) is very similar to the one obtained by using atomic partial charges of planar chlorophyll _a_ and chlorophyll _a_ ′ in Figure 11 (at native orientation). As discussed earlier, we consider at least a partial geometry optimization necessary to improve the X-ray crystallographic modeling of the electron density. If the transition monopole charges of Chang[5] or Weiss[4] are used, somewhat larger couplings as in TrEsp/HF-CIS are obtained in Figure 11. To investigate the effect of using a chlorophyll analogue, the coupling was in addition calculated using bacteriochlorophyll transition monopole charges of Chang[5] for chlorophyll _a_ and chlorophyll _a_ ′ . The coupling for the native orientation drops from about 70 cm[-][1] , if transition monopoles for chlorophyll _a_ are used, to about 20 cm[-][1] with bacteriochlorophyll transition monopoles, without changing sign in Figure 11. In addition, the dependence of the coupling on the rotation angle changes qualitatively. Taking into account the present HF-CIS result and - - the at least qualitative agreement with earlier Pariser Parr 

Pople transition monopole approaches, we judge the positive sign of the coupling obtained by Damjanovic et al.[15] as the more realistic one. However, as seen also in Figure 11, the coupling obtained by TrEsp using TDDFT/B3LYP is almost zero for the native orientation of the pigments, that is, we cannot rule out the possibility that due to dynamic correlation effects, which are not so well described in HF-CIS and the semiempirical methods, the coupling can indeed change sign. 

Another puzzling result concerns the small excitonic couplings that are obtained in all the “special pairs” in Table 1. Our preliminary quantum chemical calculations on whole “special pair” dimers show that additional charge-transfer interactions are involved. The excitonic coupling between the localized excited states of the monomers contains besides the Fo¨rster type coupling also a Dexter component, which results from electron exchange between the pigments. The latter may be promoted by charge-transfer states acting as bridging units in a superexchange type excitation energy transfer.[3] On the basis of preliminary calculations of optical spectra of reaction centers of photosystems I and II, it seems likely that those additional contributions in the excitonic coupling of the “special pairs” cannot be neglected. The earlier successful calculations of the spectra probably compensated the missing short-range contributions by an overestimation of the Fo¨rster type excitonic coupling. The present TrEsp method allows for an accurate calculation of the latter and in combination with dimer calculations will allow us to estimate the short-range contributions to the excitonic couplings. 

**C. Shift in Local Transition Energies.** The electrostatic coupling between the charge densities of the ground and excited states of the two pigments in a dimer is found to lead to different shifts of the local transition energies of the two dimer halfs, as seen in Table 1. The difference in transition energies contributes to the localization of excited states and should therefore be taken - into account in future calculations of optical spectra of pigment protein complexes. The deviation between the relative shifts in transition energies obtained in HF-CIS and in TDDFT/B3LYP is larger than that for the excitonic couplings, because the respective electrostatic potentials (Figures 3 and 4) of the ground and excited states deviate more than the electrostatic potential of the transition (Figures 1 and 2). We note that rescaling the ground and excited-state partial charges of HF-CIS and TDDFT/ B3LYP to yield the same magnitude of the difference dipole vector does not improve the agreement of the results. 

The calculated shift in excitation energies is the first-order term in a perturbation theory in the intermolecular Coulomb coupling. The higher order terms, which were neglected in the present analysis, describe excitation energy shifts due to polarization effects of the pigments. To take into account those effects, it would be necessary to calculate also the energies and transition dipole moments of the pigment’s higher excited states. 

- In a pigment protein complex, additional shifts in excitation energies occur due to the local protein environments of the pigments, as discussed in the Introduction. An interesting experimental observation concerns the role of hydrogen bonds in shifting optical transition energies of pigments in proteins. A progressive blue shift by 11 and 24 nm of the low-energy - absorption maximum of the LH2 antenna complex of _Rhodo-_ - _bacter sphaeroides_ was observed[16] upon mutation of RTyr44 Tyr45 to Phe-Tyr and to Phe-Leu. From Fourier transform resonance Raman studies, it was seen that the blue shift in the mutants correlates with the loss of hydrogen bonds to the 2-acetylcarbonyl groups of the BChla pair.[17] As seen in Figure 4, there is a strong negative potential of the charge density 

**17280** _J. Phys. Chem. B, Vol. 110, No. 34, 2006_ 

Madjet et al. 

difference at this group, that is, an electrochromic red shift is expected due to the positive partial charge provided by the hydrogen atom in the hydrogen bond, as observed in the above experiments. 

For Chla, a 6 nm blue shift of the low-energy band in the triplet minus singlet spectrum of photosystem I complexes was measured[18] after exchanging Thr739, which forms a hydrogen bond with the 13[1] keto group of the special pair chlorophyll PA, with Val, His, or Tyr, which were shown to not form a hydrogen bond. The electrostatic potential of the difference charge density of Chla in Figure 3, which is strongly negative at the 13[1] keto group, provides an explanation of this result. 

We note that the difference potential at the 13[1] keto group for BChla is different in the TDDFT/B3LYP and HF-CIS calculations, as seen in the lower part of Figure 4. Whereas TDDFT/B3LYP yields a negative potential, in HF-CIS, the potential is almost zero. Hence, TDDFT/B3LYP predicts a red shift of the BChla transition energy by the hydrogen bond to the 13[1] keto group for BChla, whereas according to the HFCIS calculations the transition energy is influenced much weaker, if at all, by such a hydrogen bond. We are only aware of mutation studies on the primary donor of bacterial reaction centers that investigate the effect of a hydrogen bond to the 13[1] keto group of BChla.[50] However, in this case, there is a strong coupling between exciton states and charge-transfer states[6,51] that determines the position of optical lines, and one would have to consider in addition the effect of the hydrogen bond on the quantum mechanical mixing of exciton and chargetransfer states. It would be easier to study the effect of this hydrogen bond in an antenna complex like the FMO complex of green sulfur bacteria, where for 5 out of 7 BChla pigments hydrogen bonds between the 13[1] keto group and the protein were reported.[52] Mutation studies could give important insights into the tuning of BChla transition energies by those hydrogen bonds and would provide a test for the two quantum chemical methods used here. 

In a combined electrostatic/quantum chemical calculation of optical transition energies of the BChla pigments in the FMO protein, that will be presented elsewhere, we find that the TDDFT/B3LYP charge densities lead to transition energies of the BChls that provide a better description of the experimental optical spectra than the respective HF-CIS values. 

## **VI. Conclusions** 

A new method, TrEsp, for ab initio calculations of excitonic Fo¨rster type Coulomb interactions between molecules was developed that takes into account the full 3-D information of the molecular transition densities. The method avoids finite grid errors of the TDC method and is numerically more efficient. It was applied here to study excitonic couplings in pigment dimers of photosynthetic antennae and reaction center complexes. Strong deviations between the PDA and the TrEsp excitonic couplings were obtained for the reaction center “special pairs”. The EDA works surprisingly well, assuming a distance of about 9 Å between the two transition charges, a value that is larger by 30% than assumed in the past.[8] An interesting result, concerning the strongly coupled antennae pigments, is that the mutual orientations of the pigments are optimized for strong excitonic coupling and that a driving force for the dimer formation is given by the Coulomb coupling between the charge densities of the ground states. From the intermolecular charge density coupling, in particular in the “special pairs”, a relative shift in local excitation energies results that can be in the same order of magnitude as the excitonic coupling. 

**Acknowledgment.** M.E.M. would like to thank Emma Sigfridsson for sending the source code of the chargefit program and the QChem developers Yihan Shao and Shawn T. Brown for incorporating the possibility to calculate the electrostatic potential of the transition density and of the charge density of excited states into the QChem program. We thank Andreas Knorr for a stimulating discussion about charge density effects and Frank Mu¨h for various discussions and critical reading of the manuscript. Financial support by Emmy Noether Grant RE1610 of the Deutsche Forschungsgemeinschaft is gratefully acknowledged. 

**Supporting Information Available:** Tables giving atomic partial charges for planar chlorophyll _a_ and bacteriochlorophyll _a_ and transition charges used in the TrEsp (HF-CIS) calculations in Figure 10. Figures showing a comparison of the TDC and TrEsp method for TDDFT-B3LYP calculations, a comparison of the ground state charge density coupling obtained here with the one that follows from a Mulliken charge analysis, and the basis set dependence of the TrEsp excitonic coupling, respectively. This material is available free of charge via the Internet at http://pubs.acs.org. 

## **References and Notes** 

- (1) Fo¨rster, Th. _Ann. Phys._ **1948** , _2_ , 55. 

(2) Dexter, D. L. _J. Chem. Phys._ **1953** , _21_ , 836. 

(3) Scholes, G. D.; Harcourt, R. D.; Ghiggino, K. P. _J. Chem. Phys._ **1995** , _101_ , 10521. 

(4) Weiss, C., Jr. _J. Mol. Spectrosc._ **1972** , _44_ , 37. 

(5) Chang, J. C. _J. Chem. Phys._ **1977** , _67_ , 3901. 

(6) Warshel, A.; Parson, W. W. _J. Am. Chem. Soc._ **1987** , _109_ , 6143. 

(7) Damjanovic, A.; Ritz, T.; Schulten, K. _Phys. Re_ V _. E_ **1999** , _59_ , 3293. 

(8) Pearlstein, R. M. _Theoretical interpretation of antenna spectra in Chlorophylls_ ; Scheer, H., Ed.; CRC Press: Boca Raton, FL, 1991; p 1047. 

(9) Krueger B. P.; Scholes, G. D.; Fleming, G. R. _J. Phys. Chem. B_ **1998** , _102_ , 5378. 

(10) Philipson, K. D.; Tsai, S. C.; Sauer, K. _J. Phys. Chem._ **1971** , _75_ , 1440. 

(11) Sauer, K.; Cogdell, R. J.; Prince, S. M.; Freer, A.; Isaacs, N. W.; Scheer, H. _Photochem. Photobiol._ **1996** , _64_ , 564. 

(12) Alden, R. G.; Johnson, E.; Nagarajan, V.; Parson, W. W.; Law, C. J.; Cogdell, R. G. _J. Phys. Chem. B_ **1997** , _101_ , 4667. 

(13) Eccles, J.; Honig, B. _Proc. Natl. Acad. Sci. U.S.A._ **1983** , _80_ , 4959. (14) Adolphs, J.; Renger, T. _Biophys. J._ **2006** , in press. 

(15) Damjanovic, A.; Vaswani, H. M.; Fromme, P.; Fleming, G. R. _J. Phys. Chem. B_ **2002** , _106_ , 10251. 

(16) Fowler, G. J.; Visschers, R. W.; Grief, G. G.; van Grondelle, R.; Hunter, C. N. _Nature_ **1992** , _355_ , 848. 

(17) Fowler, G. J.; Sockalingum, G. D.; Robert, B.; Hunter, C. N. _Biochem. J._ **1994** , _299_ , 695. 

(18) Witt, H.; Schlodder, E.; Teutloff, C.; Niklas, J.; Bordignon, E.; Carbonera, D.; Kohler, S.; Labahn, A.; Lubitz, W. _Biochemistry_ **2002** , _41_ , 8557. 

(19) Gudowska-Nowak, E.; Newton, M. D.; Fajer, J. _J. Phys. Chem._ **1990** , _94_ , 5795. 

(20) He, Z.; Sundstro¨m, V.; Pullerits, T. _J. Phys. Chem. B_ **2002** , _106_ , 11606. 

(21) Herek, J. L.; Wendling, M.; He, Z.; Polivka, T.; Garcia-Asua, G.; Cogdell, R. J.; Hunter, C. N.; van Grondelle, R.; Sundstro¨m, V.; Pullerits, T. _J. Phys. Chem. B_ **2004** , _108_ , 10398. 

(22) Camara-Artigas, A.; Blankenship, R.; Allen, J. P. _Photosynth. Res._ **2003** , _75_ , 49. 

(23) Papiz, M. Z.; Prince, S. M.; Howard, T.; Cogdell, R. J.; Isaacs, N. W. _J. Mol. Biol._ **2003** , _326_ , 1523. 

(24) Liu, Z.; Yan, H.; Wang, K.; Kuang, T.; Zhang, J.; Gui, L.; An, X.; Chang, W. _Nature_ **2004** , _428_ , 287. 

(25) Yeates, T. O.; Komiya, H.; Chirino, A.; Rees, D. C.; Allen, J. P.; Feher, G. _Proc. Natl. Acad. Sci. U.S.A._ **1988** , _85_ , 7993. 

(26) Jordan, P.; Fromme, P.; Witt, H. T.; Klukas, O.; Saenger, W.; Krauss, N. _Nature_ **2001** , _411_ , 909. 

(27) Loll, B.; Kern, J.; Saenger, W.; Zouni, A.; Biesiadka, J. _Nature_ **2005** , _438_ , 1040. 

(28) McWeeny R. _Methods of Molecular Quantum Mechanics_ ; Academic: London, 1992. 

(29) Scholes, G. D. _Annu. Re_ V _. Phys. Chem._ **2003** , _54_ , 57. 

(30) May, V.; Ku¨hn, O. _Charge and Energy Transfer Dynamics in Molecular Systems_ ; Wiley-VCH: Berlin, 2000; p 372. 

_J. Phys. Chem. B, Vol. 110, No. 34, 2006_ **17281** 

Intermolecular Coulomb Couplings 

(31) _Jaguar 5.5_ . Schroedinger, L.L.C: Portland, OR, 1991-2003. 

(32) Sigfridsson, E.; Ryde, U. _J. Comput. Chem._ **1998** , _19_ , 377. 

- (33) Kong, J.; White, C. A.; Krylov, A. I.; Sherrill, C. D.; Adamson, R. 

- D.; Furlani, T. R.; Lee, M. S.; Lee, A. M.; Gwaltney, S. R.; Adams, T. R.; Ochsenfeld, C.; Gilbert, A. T. B.; Kedziora, G. S.; Rassolov, V. A.; Maurice, D. R.; Nair, N.; Shao, Y.; Besley, N. A.; Maslen, P. E.; Dombroski, J. P.; Dachsel, H.; Zhang, W. M.; Korambath, P. P.; Baker, J.; Byrd, E. F. C.; Van Voorhis, T.; Oumi, M.; Hirata, S.; Hsu, C. P.; Ishikawa, N.; Florian, 

- J.; Warshel, A.; Johnson, B. G.; Gill, P. M. W.; Head-Gordon, M.; Pople, 

- J. A. Q.-Chem. _J. Comput. Chem._ **2000** , _21_ , 1532. 

- (34) Hinsen, K.; Roux, B. _J. Comput. Chem_ . **1997** , _18_ , 368. 

- (35) Knox, R. S.; Spring B. Q. _Photochem. Photobiol._ **2003** , _77_ , 

- 497. 

- (36) Hsu, C.-P.; Fleming, G. R.; Head-Gordon, M.; Head Gordon, T. _J._ 

- _Chem. Phys._ **2001** , _114_ , 3065. 

- (37) Sener, M. K.; Lu, D.; Ritz, T.; Park, S.; Fromme, P.; Schulten, K. 

- _J. Phys. Chem. B_ **2002** , _106_ , 7948. 

- (38) Cornell, W. D.; Cieplak, P.; Bayly, C. I.; Kollman, P. A. _J. Am._ 

- _Chem. Soc._ **1993** , _115_ , 9620. 

- (39) Honig, B.; Nicholls, A. _Science_ **1995** , _268_ , 1144. 

- (40) A comparison of the ground-state charge density coupling obtained 

- with the present method, and the coupling resulting from Mulliken charges is given in the Supporting Information. 

(41) Baly, C. I.; Cieplak, P.; Cornell, W. D.; Kollman, P. A. _J. Phys. Chem._ **1993** , _97_ , 10269. 

(42) Hanson, L. K.; Fajer, J.; Thompson, M. A.; Zerner, M. C. _J. Am. Chem. Soc._ **1987** , _109_ , 4728. 

(43) Lockhart, D. J.; Boxer, S. G. _Proc. Natl. Acad. Sci. U.S.A._ **1988** , _85_ , 107. 

(44) Krawczyk, S. _Biochim. Biophys. Acta_ **1991** , _1056_ , 64. 

(45) Koolhaas, M. H. C.; Frese, R. N.; Fowler, G. J: S.; Bibby, T. S.; Georgakopoulou, S.; van der Zwan, G.; Hunter, C. N.; van Grondelle, R. _Biochemistry_ **1998** , _37_ , 4693. 

(46) Georgakopoulou, S.; Frese, R. N.; Johnson, E.; Koolhaas, C.; Cogdell, R. J.; van Grondelle, R.; van der Zwan, G. _Biophys. J._ **2002** , _82_ , 2184. 

- (47) Scholes, G. D.; Gould, I. R.; Cogdell, R. J.; Fleming, G. R. _J. Phys._ 

- _Chem. B_ **1999** , _103_ , 2543. 

- (48) Iozzi, M. F.; Mennucci, B.; Tomasi, J. _J. Chem. Phys._ **2004** , _120_ , 

- 7029. 

- (49) Krauss, N. Personal communication. 

- (50) Mattioli, T. A.; Lin, X.; Allen, J. P.; Williams, J. C. _Biochemistry_ 

- **1995** , _34_ , 6142. 

- (51) Renger, T. _Phys. Re_ V _. Lett._ **2004** , _93_ , article no. 188101. 

- (52) Li, Y.-F.; Zhou, W.; Blankenship, R. E.; Allen, J. P. _J. Mol. Biol._ 

- **1997** , _271_ , 456. 

