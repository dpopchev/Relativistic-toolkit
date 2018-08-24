import logging
import sympy as sp
import itertools as itools

from sympy import tensorproduct as TP
from sympy import tensorcontraction as TC
from IPython import get_ipython

#~ get_ipython().magic('reset -sf')
#~ sp.init_printing(
    #~ use_latex = True,
    #~ forecolor = "White"
#~ )

class toolkit:
    """
    see README

    LACKING ANY INPUT CHECKS !!!!

    in all examples RT is the instance of toolkit class
    in all examples sp is sympy module
    """

    def __init__(self):

        #~ list of sympy symbols to be used as coordiantes
        self.coords = None

        #~ the metric and etc geometricals
        self.g_co = None
        self.g_contra = None
        self.Christoffel_1st = None
        self.Christoffel_2nd = None
        self.RiemanT = None
        self.RicciT_co = None
        self.RicciS = None
        self.EinsteinT_co = None
        self.u_co = None
        self.u_contra = None
        self.rho = None
        self.p = None
        self.varphi = None
        self.V = None

        #~ the perturbed metric and ect goemtricals
        self.h_co = None
        self.h_contra = None
        self.hChristoffel_2nd = None
        self.hRiemanT = None
        self.hRicciT_co = None
        self.hRicciS = None
        self.hEinsteinT_co = None
        self.hU_co = None
        self.hU_contra = None
        self.hRho = None
        self.hP = None
        self.hVarphi = None
        self.hV = None

        #~ metric with small parameter to expand around and its geometricals
        self.eps = None
        self.epsG_co = None
        self.epsG_contra = None
        self.epsChristoffel_2nd = None
        self.epsRiemanT = None
        self.epsRicciT_co = None
        self.epsRicciS = None
        self.epsEinsteinT_co = None
        self.epsU_co = None
        self.epsU_contra = None
        self.epsRho = None
        self.epsP = None
        self.epsVarphi = None
        self.epsV = None

        return

    def set_coordinates(self, coords):
        """
        sets the coordinates as list of symbolic python symbols
        the amount of coordinates defines the dimension of the manifold

        Parameters
        ----------
        coords : list
            list of sympy symbols to be used as coordinates

        Returns
        -------
        """

        #~ TODO input checks

        self.coords = [ _ for _ in coords ]

        return

    def get_coordinates(self):
        """
        returns the setted coordinates

        Parameters
        ----------

        Returns
        -------
        : list
            the saved list, to make sure it is what we want

        Examples
        --------
        >>> import sympy as sp
        >>> sp.init_printing(pretty_print=False, use_latex=False)

        >>> t, x, y, z = sp.symbols("t,x,y,z")
        >>> RT.set_coordinates( [t,x,y,z] )
        >>> RT.get_coordinates()
        [t, x, y, z]

        >>> x_0, x_1, x_2, x_3= sp.symbols("x_0, x_1, x_2, x_3")
        >>> RT.set_coordinates( [x_0, x_1, x_2, x_3] )
        >>> RT.get_coordinates()
        [x_0, x_1, x_2, x_3]

        >>> t, r, theta, phi = sp.symbols("t,r,\\theta,\phi")
        >>> RT.set_coordinates([t,r,theta,phi])
        >>> RT.get_coordinates()
        [t, r, heta, \phi]
        """

        return self.coords

#################################################################################
#~ GEOMETRICAL FIELD EQUATIONS EXPANDED AROUND SMALL PARAMETER
#################################################################################
#~ public/private  methods to get/set
#~      LHS and RHS of Einstein equations with conservation laws
#~      with expanding around small parameter
#~
#~ depends on geometrical structures on non perturbed part
#################################################################################

    def set_epsGmetric(self, eps, epsG_co):

        #~ TODO input chekcs
        #~ TODO docstring

        self.eps = eps

        self.epsG_co = sp.Array(epsG_co)
        self.epsG_contra = sp.Array( self.epsG_co.tomatrix()**-1 )

        g_contr = TC( TP(self.epsG_co, self.epsG_contra), (0,2) )

        g_contr = g_contr.applyfunc(
            lambda _: sp.series(_, x=self.eps, x0=0, n=2).removeO()
        )

        if g_contr.tomatrix() != sp.eye(len(self.coords)):

            logging.error(
                "Metric contraction does not result in Kronecker symbol!"
            )
            display(g_contracted)
            return

        logging.info("Will compute the Christoffel, Riman, Ricci...")

        self.epsG_co = self.epsG_co.applyfunc(
            lambda _: sp.series(_, x=self.eps, x0=0, n=2).removeO()
        )

        self.epsG_contra = self.epsG_contra.applyfunc(
            lambda _: sp.series(_, x=self.eps, x0=0, n=2).removeO()
        )

        self.epsChristoffel_2nd = self._get_Christoffel_2nd(
            self.coords, self.epsG_co, self.epsG_contra
        ).applyfunc(
            lambda _: sp.series(_, x=self.eps, x0=0, n=2).removeO()
        )

        self.epsRiemanT = self._get_Rieman_tensor(
            self.coords, self.epsG_co, self.epsG_contra, self.epsChristoffel_2nd
        ).applyfunc(
            lambda _: sp.series(_, x=self.eps, x0=0, n=2).removeO()
        )

        self.epsRicciT_co = self._get_Ricci_tensor_co(
            self.coords, self.epsG_co, self.epsG_contra, self.epsRiemanT
        ).applyfunc(
            lambda _: sp.series(_, x=self.eps, x0=0, n=2).removeO()
        )

        self.epsRicciS = sp.series(
            self._get_Ricci_scalar(
                self.coords, self.epsG_co, self.epsG_contra, self.epsRicciT_co
            ),
            x=self.eps, x0=0, n=2
        ).removeO()

        self.epsEinsteinT_co = self._get_Einstein_tensor_co(
            self.coords, self.epsG_co, self.epsG_contra,
            self.epsRicciT_co, self.epsRicciS
        ).applyfunc(
            lambda _: sp.series(_, x=self.eps, x0=0, n=2).removeO()
        )

        return

    def get_epsGmetric(self):

        #~ TODO input chekcs
        #~ TODO docstring

        return self.epsG_co, self.epsG_contra

    def get_epsChristoffel_2nd(self):

        #~ TODO input chekcs
        #~ TODO docstring

        return self.epsChristoffel_2nd

    def get_epsRieman_tensor(self):

        #~ TODO input chekcs
        #~ TODO docstring

        return self.epsRiemanT

    def get_epsRicci_tensor_co(self):

        #~ TODO input chekcs
        #~ TODO docstring

        return self.epsRicciT_co

    def get_epsRicci_scalar(self):

        #~ TODO input chekcs
        #~ TODO docstring

        return self.epsRicciS

    def get_epsEinstein_tensor_co(self):

        #~ TODO input chekcs
        #~ TODO docstring

        return self.epsEinsteinT_co

    def set_epsScalarField(self, epsVarphi, epsV):

        #~ TODO inpucht checks
        #~ TODO docstring

        self.epsVarphi = epsVarphi
        self.epsV = 0

        return

    def get_epsScalarField(self):

        #~ TODO inpucht checks
        #~ TODO docstring

        return self.epsVarphi, self.epsV

    def get_epsScalarField_field_eq(self):

        #~ TODO inpucht checks
        #~ TODO docstring

        scalarField_lhs = self._get_lhs_scalarField(
            self.epsG_co, self.epsG_contra, self.epsVarphi,
            self.epsChristoffel_2nd,
            self.coords
        )

        scalarField_lhs = sp.series(
            scalarField_lhs,
            x = self.eps,
            x0 = 0,
            n = 2
        ).removeO()

        return sp.simplify(scalarField_lhs)

#################################################################################
#~ PERTURBED GEOMETRICAL FIELD EQUATIONS
#################################################################################
#~ public/private  methods to get/set
#~      LHS and RHS of PERTURBED Einstein equations with conservation laws
#~
#~ depends on geometrical structures on non perturbed part
#################################################################################

    def set_hmetric(self, h_co):

        #~ TODO input chekcs
        #~ TODO docstring

        self.h_co = sp.Array(h_co)

        self.h_contra = (-1)*TP(self.g_contra, self.g_contra, self.h_co)
        self.h_contra = TC( self.h_contra, (1,4))
        self.h_contra = TC( self.h_contra, (1,3))

        self.hChristoffel_2nd = self._get_hChristoffel_2nd(
            self.coords, self.g_co, self.g_contra, self.h_co, self.Christoffel_2nd
        )

        self.hRiemanT = self._get_hRieman_tensor(
            self.coords, self.g_co, self.g_contra, self.h_co,
            self.hChristoffel_2nd,
            self.Christoffel_2nd
        )

        self.hRicciT_co = self._get_hRicci_tensor_co( self.hRiemanT )

        self.hRicciS = self._get_hRicciS(
            self.g_co, self.g_contra,
            self.h_co, self.h_contra,
            self.RicciT_co, self.hRicciT_co
        )

        self.hEinsteinT_co = self._get_hEinstein_tensor_co(
            self.hRicciT_co, self.h_co, self.RicciS, self.g_co, self.hRicciS
        )

        return

    def get_hmetric(self):

        #~ TODO input chekcs
        #~ TODO docstring

        return self.h_co, self.h_contra

    def get_hChristoffel_2nd(self):

        #~ TODO inpucht checks
        #~ TODO docstring

        return self.hChristoffel_2nd

    def get_hRieman_tensor(self):

        #~ TODO inpucht checks
        #~ TODO docstring

        return self.hRiemanT

    def get_hRicci_tensor_co(self):

        #~ TODO inpucht checks
        #~ TODO docstring

        return self.hRicciT_co

    def get_hRicci_scalar(self):

        #~ TODO inpucht checks
        #~ TODO docstring

        return self.hRicciS

    def get_hEinstein_tensor_co(self):

        #~ TODO inpucht checks
        #~ TODO docstring

        return self.hEinsteinT_co

    def set_hU_contra(self, xi_contra):

        #~ TODO inpucht checks
        #~ TODO docstring

        self.hU_contra = TP(
            self.u_contra,
            self._nabla_T_1contra(xi_contra, self.Christoffel_2nd, self.coords)
        )

        self.hU_contra = TC(self.hU_contra, (0,1))

        self.hU_co = (-1)*TP(
            self.u_co,
            self._nabla_T_1contra(xi_contra, self.Christoffel_2nd, self.coords)
        )

        self.hU_co = TC(self.hU_co, (0,2))

        return

    def get_hU(self):

        #~ TODO inpucht checks
        #~ TODO docstring

        return self.hU_co, self.hU_contra

    def set_hPressure(self, hP):

        #~ TODO inpucht checks
        #~ TODO docstring

        self.hP = hP

        return

    def get_hPressure(self):

        #~ TODO inpucht checks
        #~ TODO docstring

        return self.hP

    def set_hDensity(self, hRho):

        #~ TODO inpucht checks
        #~ TODO docstring

        self.hRho = hRho

        return

    def get_hDensity(self):

        #~ TODO inpucht checks
        #~ TODO docstring

        return self.hRho

    def set_hScalarField(self, hVarphi, hV):

        #~ TODO inpucht checks
        #~ TODO docstring

        self.hVarphi = hVarphi
        self.hV = hV

        return

    def get_hScalarField(self):

        #~ TODO inpucht checks
        #~ TODO docstring

        return self.hVarphi, self.hV

    def get_hDiv_tensor_energy_momentum(self):

        #~ TODO docstring
        #~ TODO input check

        hT_fluid_co = self._get_hTensor_perfect_fluid_co(
            self.rho, self.hRho,
            self.p, self.hP,
            self.u_co, self.hU_co,
            self.g_co, self.h_co,
            self.coords
        )

        hT_scalarField_co = self._get_hTensor_scalarField_co(
            self.varphi, self.hVarphi,
            self.V, self.hV,
            self.g_contra, self.h_contra,
            self.g_co, self.h_co,
            self.coords
        )

        hT_co = hT_fluid_co + hT_scalarField_co

        T_fluid_co = self._get_tensor_perfect_fluid_co(
            self.rho, self.p, self.u_co, self.g_co
        )

        T_scalarField_co = self._get_tensor_scalarField_co(
            self.varphi, self.V, self.g_co, self.g_contra, self.coords
        )

        T_co = T_fluid_co + T_scalarField_co

        hT = (
            TC( TP( self.g_contra, hT_co ), (1,3) )
            + TC( TP( self.h_contra, T_co ), (1,3) )
        )

        nabla_hT = self._nabla_T_1co_1contra(
            hT, self.Christoffel_2nd, self.coords
        )

        return sp.simplify( TC( nabla_hT, (0,2) ))

    def get_hScalarField_field_eq(self):

        hT_fluid_co = self._get_hTensor_perfect_fluid_co(
            self.rho, self.hRho,
            self.p, self.hP,
            self.u_co, self.hU_co,
            self.g_co, self.h_co,
            self.coords
        )

        T_fluid_co = self._get_tensor_perfect_fluid_co(
            self.rho, self.p, self.u_co, self.g_co
        )

        hScalarField_lhs = self._get_hLHS_scalarField(
            self.g_co, self.g_contra, self.h_co, self.h_contra,
            self.varphi, self.hVarphi,
            self.Christoffel_2nd, self.coords, self.hChristoffel_2nd
        )

        hScalarField_rhs = self._get_hRHS_scalarField(
            self.g_co, self.g_contra, self.h_co, self.h_contra,
            self.varphi, self.V, self.hVarphi, self.hV, T_fluid_co, hT_fluid_co,
            self.Christoffel_2nd, self.coords
        )

        return hScalarField_lhs, hScalarField_rhs

    @staticmethod
    def _get_hChristoffel_2nd(coords, g_co, g_contra, h_co, Chris_2):

        #~ TODO inpucht checks
        #~ TODO docstring

        nabla_h_co = toolkit._nabla_T_2co(h_co, Chris_2, coords)

        div_nabla_h_co = TP(g_contra, nabla_h_co)

        term_1 = TC(div_nabla_h_co, (1,4))
        term_2 = TC(div_nabla_h_co, (1,4))
        term_3 = TC(div_nabla_h_co, (1,2))

        res = sp.Array( [
            [
                [
                    term_1[contra_1, co_1, co_2]
                    + term_2[contra_1, co_2, co_1]
                    - term_3[contra_1, co_1, co_2]
                    for co_2 in range(len(coords))
                ] for co_1 in range(len(coords))
            ] for contra_1 in range(len(coords))
        ] )

        return sp.simplify((1/2)*res)

    @staticmethod
    def _get_hRieman_tensor(coords, g_co, g_contra, h_co, hChris2, Chris_2):

        #~ TODO inpucht checks
        #~ TODO docstring

        nabla_hChris2 = toolkit._nabla_T_1contra_2co(hChris2, Chris_2, coords)

        hRiemanT = sp.Array( [
            [
                [
                    [
                        nabla_hChris2[co_2, contra_1, co_1, co_3 ]

                        - nabla_hChris2[co_3, contra_1, co_1, co_2]

                        for co_3 in range(len(coords))
                    ] for co_2 in range(len(coords))
                ] for co_1 in range(len(coords))
            ] for contra_1 in range(len(coords))
        ] )

        return sp.simplify(hRiemanT)

    @staticmethod
    def _get_hRicci_tensor_co(hRiemanT):

        #~ TODO inpucht checks
        #~ TODO docstring

        return sp.simplify( TC(hRiemanT, (0,2)) )

    @staticmethod
    def _get_hRicciS(g_co, g_contra, h_co, h_contra, RicciT_co, hRicciT_co):

        term1 = TP(RicciT_co, h_contra)
        term1 = TC( term1, (0,2) )
        term1 = TC( term1, (0,1) )

        term2 = TP(g_contra, hRicciT_co)
        term2 = TC( term2, (0,2) )
        term2 = TC( term2, (0,1) )

        return sp.simplify(term1 + term2)

    @staticmethod
    def _get_hEinstein_tensor_co( hRicciT_co, h_co, RicciS, g_co, hRicciS ):

        #~ TODO inpucht checks
        #~ TODO docstring

        term1 = hRicciT_co

        term2 = (-1/2)*RicciS*h_co

        term3 = (-1/2)*hRicciS*g_co

        return sp.simplify(term1 + term2 + term3)

#################################################################################
#~ TENSOR ENERGY AND MOMENTUM PERTURBED
#################################################################################
#~ private methods to compute
#~      tensor for perfect fluid and tensor of scalar field
#~      PERTURBED
#################################################################################

    @staticmethod
    def _get_hTensor_perfect_fluid_co(
        rho, hRho,
        p, hP,
        u_co, hU_co,
        g_co, h_co,
        coords
    ):

        term1 = (hP + hRho)*TP(u_co, u_co)

        tmp = TP( u_co, hU_co )

        term2 = sp.Array( [
            [
                tmp[co_1, co_2] + tmp[co_2, co_1]

                for co_2 in range(len(coords))
            ] for co_1 in range(len(coords))
        ] )

        term2 *= (rho + p)

        term3 = p*h_co + g_co*hP

        return sp.simplify(term1 + term2 + term3)

    @staticmethod
    def _get_hTensor_scalarField_co(
        varphi, hVarphi,
        V, hV,
        g_contra, h_contra,
        g_co, h_co,
        coords
    ):

        tmp_varphi_varphi = TP(
            toolkit._nabla_scalar(varphi, coords),
            toolkit._nabla_scalar(varphi, coords)
        )

        tmp_hVarphi_varphi = TP(
            toolkit._nabla_scalar(hVarphi, coords),
            toolkit._nabla_scalar(varphi, coords)
        )

        term1 = sp.Array( [
            [
                tmp_hVarphi_varphi[co_1, co_2] + tmp_hVarphi_varphi[co_2, co_1]

                for co_2 in range(len(coords))
            ] for co_1 in range(len(coords))
        ] )

        term2 = TP(g_contra, tmp_varphi_varphi, h_co)
        term2 = TC( term2, (0,2))
        term2 = TC( term2, (0,1))

        term3 = TP(g_co, h_contra, tmp_varphi_varphi)
        term3 = TC(term3, (2,4))
        term3 = TC(term3, (2,3))

        term4 = TP(g_co, g_contra, term1)
        term4 = TC(term4, (2,4))
        term4 = TC(term4, (2,3))

        term5 = 2*V*h_co

        term6 = 2*g_co*sp.diff(V, varphi)*hVarphi

        return sp.simplify(
            term1 - term2 - term3 - term4 - term5 - term6
        )

################################################################################
#~ PERTURBED SCALAR FIELD EQUATION
#################################################################################
#~ private methods to compute
#~      LHS and RHS of scalar field equation PERTUREBED
#~
#~ requires
#~      set_hScalarField()
#~ needed for
#~      get_hdiv_tensor_energy_momentum()
#~      get_hu_div_tensor_energy_momentum()
#~      get_hSTT_field_eq()
#################################################################################

    @staticmethod
    def _get_hLHS_scalarField(
        g_co, g_contra, h_co, h_contra,
        varphi, hVarphi,
        Chris_2, coords, hChris_2
    ):

        #~ TODO inpucht checks
        #~ TODO docstring

        lhs_1 = TP(
            h_contra,
            toolkit._nabla_T_1co(
                toolkit._nabla_scalar(varphi, coords),
                Chris_2,
                coords
            )
        )

        lhs_1 = TC( lhs_1, (0,2))
        lhs_1 = TC( lhs_1, (0,1))

        lhs_2 = TP(
            g_contra,
            toolkit._nabla_T_1co(
                toolkit._nabla_scalar(hVarphi, coords),
                Chris_2,
                coords
            )
        )

        lhs_2 = TC( lhs_2, (0,2) )
        lhs_2 = TC( lhs_2, (0,1) )

        lhs_3 = TP(
            hChris_2,
            toolkit._nabla_scalar(varphi, coords)
        )

        lhs_3 = TC( lhs_3, (0,3) )

        lhs_3 = TP( g_contra, lhs_3 )
        lhs_3 = TC( lhs_3, (0,2) )
        lhs_3 = TC( lhs_3, (0,1))


        return sp.simplify(lhs_1 + lhs_2 - lhs_3)

    @staticmethod
    def _get_hRHS_scalarField(
        g_co, g_contra, h_co, h_contra,
        varphi, V, hVarphi, hV, T_co, hT_co,
        Chris_2, coords
    ):

        #TODO inpucht checks
        #TODO docstring

        #var_tmp = sp.Function("\\varphi")(coords[1])
        a = sp.Function("a")(varphi)

        T_trace = TC( TP(g_contra, T_co), (0,2) )
        T_trace = TC( T_trace, (0,1) )

        rhs_1 = sp.diff(a, varphi)*T_trace*hVarphi

        rhs_2 = TP(h_contra, T_co)
        rhs_2 = TC( rhs_2, (0,2))
        rhs_2 = TC( rhs_2, (0,1))
        rhs_2 *= a

        rhs_3 = TP(g_contra, hT_co)
        rhs_3 = TC( rhs_3, (0,2) )
        rhs_3 = TC( rhs_3, (0,1) )
        rhs_3 *= a

        rhs_4 = sp.diff(V, varphi)
        rhs_4 = sp.diff(rhs_4, varphi)*hVarphi

        return sp.simplify(
            -4*sp.pi*( rhs_1 + rhs_2 + rhs_3 ) + rhs_4
        )
#################################################################################
#~ GEOMETRICAL FIELD EQUATIONS
#################################################################################
#~ public methods to get/set
#~      LHS and RHS of Einstein equations with conservation laws
#################################################################################

    def set_metric(self, g_co):
        """
        sets the full covariant metric
        they can include sympy functions of the coordinates !

        checks if the contracted metric is the Kronecker Symbol (the eye metric)

        CURRENTLY NO INPUT CHECKING IS MADE !!!!

        Parameters
        ----------
        g_co : list
            2d list to define the covariant metric

        Returns
        -------
        """

        #~ TODO input checks

        self.g_co = sp.Array(g_co)

        self.g_contra = sp.Array( self.g_co.tomatrix()**-1 )

        g_contracted = TC( TP(self.g_co, self.g_contra), (0,2) )

        if g_contracted.tomatrix() != sp.eye(len(self.coords)):
            logging.error(
                "Metric contraction does not result in Kronecker symbol!"
            )
            display(g_contracted)
            return

        logging.info("Will compute the Christoffel, Riman, Ricci...")

        self.Christoffel_2nd = self._get_Christoffel_2nd(
            self.coords, self.g_co, self.g_contra
        )

        self.RiemanT = self._get_Rieman_tensor(
            self.coords, self.g_co, self.g_contra, self.Christoffel_2nd
        )

        self.RicciT_co = self._get_Ricci_tensor_co(
            self.coords, self.g_co, self.g_contra, self.RiemanT
        )

        self.RicciS = self._get_Ricci_scalar(
            self.coords, self.g_co, self.g_contra, self.RicciT_co
        )

        self.EinsteinT_co = self._get_Einstein_tensor_co(
            self.coords, self.g_co, self.g_contra, self.RicciT_co, self.RicciS
        )

        return

    def get_metric(self):
        """
        returns the setted co and covariant metricies to see and admire

        Parameters
        ----------

        Returns
        -------
        g_co : sympy array
            2d sympy array to see if covariant metric is as expected

        g_contra : sympy array
            2d sympy array to see if contravariant metric is as expected

        Examples
        --------
        >>> import sympy as sp
        >>> sp.init_printing(pretty_print=False, use_latex=False)

        >>> t, r, theta, phi = sp.symbols("t,r,\theta,\phi")
        >>> RT.set_coordinates([t,r,theta,phi])

        >>> g_co = [
        ...    [ -1, 0, 0, 0 ],
        ...    [ 0, 1, 0, 0 ],
        ...    [ 0, 0, 1, 0 ],
        ...    [ 0, 0, 0, 1 ]
        ... ]

        >>> RT.set_metric(g_co)
        >>> g_co, g_contra = RT.get_metric()
        >>> g_co.tolist()
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        >>> g_contra.tolist()
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        >>> Phi = sp.Function("\Phi")(r)
        >>> Lambda = sp.Function("\Lambda")(r)
        >>> g_co = [
        ...     [ -sp.exp(2*Phi), 0, 0, 0 ],
        ...     [ 0, sp.exp(2*Lambda), 0, 0 ],
        ...     [ 0, 0, r**2, 0 ],
        ...     [ 0, 0, 0, (r*sp.sin(theta))**2 ]
        ... ]

        >>> RT.set_metric(g_co)
        >>> g_co, g_contra = RT.get_metric()
        >>> g_co.tolist()
        [[-exp(2*\Phi(r)), 0, 0, 0], [0, exp(2*\Lambda(r)), 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(heta)**2]]
        >>> g_contra.tolist()
        [[-exp(-2*\Phi(r)), 0, 0, 0], [0, exp(-2*\Lambda(r)), 0, 0], [0, 0, r**(-2), 0], [0, 0, 0, 1/(r**2*sin(heta)**2)]]
        """

        return self.g_co, self.g_contra

    def get_Christoffel_2nd(self):

        #~ TODO docstirng

        return self.Christoffel_2nd

    def get_Rieman_tensor(self):

        #~ TODO docstring

        return self.RiemanT

    def get_Ricci_tensor_co(self):

        #~ TODO docstring

        return self.RicciT_co

    def get_Ricci_scalar(self):

        #~ TODO docstring

        return self.RicciS

    def get_Einstein_tensor_co(self):

        #~ TODO docstring

        return self.EinsteinT_co

    def set_u_contra(self, u_contra):
        """
        set the components of the contravariant 4 velocicty U

        it automaticly sets the covariant veloctiy

        which may include functions of the coordinates

        CURRENTLY NO INPUT CHECKING IS MADE !!!!

        Parameters
        ----------
        u_contra : list
            1d list which will be converter to rank 1 array

        Returns
        -------
        """

        #~ TODO input checks
        #~ TODO include examples

        self.u_contra = sp.Array(u_contra)

        self.u_co = TC( TP(self.g_co, self.u_contra), (1,2))

        return

    def get_u(self):
        """
        return the 4 velocity as sympy arrays

        Returns
        -------
        u_co : sp array of rank 1

        u_contra : sp array of rank 1
        """

        #~ TODO finish docstring
        #~ TODO include examples

        return self.u_co, self.u_contra

    def set_pressure(self, p):

        #~ TODO NO DOCSTRING

        self.p = p

        return

    def get_pressure(self):

        return self.p

    def set_density(self, rho):

        #~ TODO NO DOCSTRING

        self.rho = rho

        return

    def get_density(self):

        return self.rho

    def set_scalarField(self, varphi, V):
        """
        give functions for the scalar field

        Parameters
        ----------
        varphi : sympy function

        V : sympy function
        """

        #~ TODO finish the docstring

        self.varphi = varphi
        self.V = V

        return

    def get_scalarField(self):

        #~ TODO NO DOCSTRING

        return self.varphi, self.V

    def get_div_tensor_energy_momentum(self):

        #~ TODO docstring
        #~ TODO input check

        T_fluid_co = self._get_tensor_perfect_fluid_co(
            self.rho, self.p, self.u_co, self.g_co
        )

        T_scalarField_co = self._get_tensor_scalarField_co(
            self.varphi, self.V, self.g_co, self.g_contra, self.coords
        )

        T_co = T_fluid_co + T_scalarField_co
        T = TC( TP(self.g_contra, T_co), (1,3) )

        nabla_T = self._nabla_T_1co_1contra(
            T, self.Christoffel_2nd, self.coords
        )

        return sp.simplify(TC(nabla_T, (0,2)))

    def get_u_div_tensor_energy_momentum(self):

        #~ TODO docstring
        #~ TODO input check

        div_T = self.get_div_tensor_energy_momentum()

        print("\n Terms of div(T) parallel to u \n")

        u_par_div_T = TP(self.u_contra, div_T)

        for _ in sp.expand( TC(u_par_div_T, (0,1) ) ).args:
            display(_)

        print("\n Terms of div(T) perpendicular to u \n")

        kronecker = sp.Array(sp.eye(4))

        u_perp_div_T_term_1 = TC( TP(kronecker, div_T), (0,2) )

        u_perp_div_T_term_2 = TC( TP(self.u_co, self.u_contra, div_T), (1,2))

        u_perp_div_T = sp.simplify(u_perp_div_T_term_1 + u_perp_div_T_term_2 )

        for i, v in enumerate(u_perp_div_T):
            display(self.coords[i], v)

        return u_par_div_T, u_perp_div_T

    def get_STT_field_eq(self):

        #~ TODO NO DOCSTRING

        T_fluid_co = 8*sp.pi*self._get_tensor_perfect_fluid_co(
            self.rho, self.p, self.u_co, self.g_co
        )

        T_scalarField_co = self._get_tensor_scalarField_co(
            self.varphi, self.V, self.g_co, self.g_contra, self.coords
        )

        T_energy_momentum = T_fluid_co + T_scalarField_co

        if self.Einstein_co:
            G_co = self.Einstein_co
        else:
            G_co = self._get_Einstein_tensor_co(
                self.coords, self.g_co, self.g_contra, self.RicciT, self.RicciS
            )

        scalarField_lhs = self._get_lhs_scalarField(
            self.g_co, self.g_contra, self.varphi, self.V,
            self.Christoffel_2nd,
            self.coords
        )

        scalarField_rhs = self._get_rhs_scalarField(
            self.g_co, self.g_contra, self.varphi, self.V,
            self._get_tensor_perfect_fluid_co(
                self.rho, self.p, self.u_co,
                self.g_co
            ),
            self.Christoffel_2nd,
            self.coords
        )

        print("\n LHS of Geometrical part (Einstein tenosor) \n")

        for i_comb in itools.product(
            [_ for _ in range(len(self.coords))], repeat=2
        ):
            if G_co[i_comb[0], i_comb[1]]:
                print("\n Einstein tensor non zero combination \n")

                display(self.coords[i_comb[0]], self.coords[i_comb[1]])

                print("\n term by term \n")

                for _ in sp.expand(G_co[i_comb[0], i_comb[1]]).args:
                    display(_)

                print("\n searching next non zero Einstein tensor... ")

        print("\n RHS of Geometrical part (Tensor energy momentum) \n")
        for i_comb in itools.product(
            [_ for _ in range(len(self.coords))], repeat=2
        ):
            if T_energy_momentum[i_comb[0], i_comb[1]]:
                print("\n Tensor energy momentum non zero combination \n")

                display(self.coords[i_comb[0]], self.coords[i_comb[1]])

                print("\n term by term \n")

                for _ in sp.expand(T_energy_momentum[i_comb[0], i_comb[1]]).args:
                    display(_)

                print("\n searching next non zero energy momentum combination \n")

        print("\n LHS Scalar field \n")
        for _ in sp.expand(scalarField_lhs).args:
            display(_)

        print("\n RHS Scalar field \n")
        for _ in sp.expand(scalarField_rhs).args:
            display(_)

        return G_co, \
        T_energy_momentum, \
        scalarField_lhs, scalarField_rhs

#################################################################################
#~ SCALAR FIELD EQUATION
#################################################################################
#~ private methods to compute
#~      LHS and RHS of scalar field equation
#~
#~ requires
#~      set_scalarField()
#~ needed for
#~      get_div_tensor_energy_momentum()
#~      get_u_div_tensor_energy_momentum()
#~      get_STT_field_eq()
#################################################################################

    @staticmethod
    def _get_lhs_scalarField( g_co, g_contra, varphi, Chris_2, coords ):

        #~ TODO DOCSTRING

        lhs = TP(
            g_contra,
            toolkit._nabla_T_1co(
                toolkit._nabla_scalar(varphi, coords),
                Chris_2,
                coords
            )
        )

        lhs = TC( lhs, (1,3) )
        lhs = TC( lhs, (0,1) )

        return sp.simplify(lhs)

    @staticmethod
    def _get_rhs_scalarField( g_co, g_contra, varphi, V, T_co, Chris_2, coords ):

        #~ TODO DOCSTRING

        #~ var_tmp = sp.Function("\\varphi")(coords[1])

        a = sp.Function("a")(varphi)

        T_trace = TC( TP(g_contra, T_co), (0,2) )

        T_trace = TC( T_trace, (0,1) )

        rhs = -4*sp.pi*a*T_trace + sp.diff( V, varphi )

        return sp.simplify(rhs)

#################################################################################
#~ TENSOR ENERGY AND MOMENTUM
#################################################################################
#~ private methods to compute
#~      tensor for perfect fluid and tensor of scalar field
#################################################################################

    @staticmethod
    def _get_tensor_scalarField_co(varphi, V, g_co, g_contra, coords):

        #~ TODO NO DOCSTRING

        term1 = 2*TP(
            toolkit._nabla_scalar(varphi, coords),
            toolkit._nabla_scalar(varphi, coords)
        )

        term2 = TP(
            g_contra,
            toolkit._nabla_scalar(varphi, coords),
            toolkit._nabla_scalar(varphi, coords)
        )
        term2 = TC( term2, (0,2) )
        term2 = TC( term2, (0,1) )
        term2 *= g_co

        term3 = 2*V*g_co

        return sp.simplify(
            term1 - term2 - term3
        )

    @staticmethod
    def _get_tensor_perfect_fluid_co(rho, p, u_co, g_co):

        #~ TODO NO DOCSTRING

        return sp.simplify(
            (rho + p)*TP(u_co, u_co) + p*g_co
        )

#################################################################################
#~ GEOMETRY CURVATURE
#################################################################################
#~ private methods to compute
#~      Christoffel 1st, 2nd; Riemann tensor, Ricci tensor and scalar and G
#################################################################################

    @staticmethod
    def _comp_Christoffel_1st(coords, g_co, co):
        """
        for provided coordinates as list, full covariant metric
        and covarinat indexe reutrns the corresponding Christoffel symbol
        of first kind using the following formula written in latex

        \Gamma_{\lambda \mu \nu} =
            = 1/2(
                \partial_\mu g_{\nu \lambda}
                + \partial_\nu g_{\mu \lambda}
                - \partial_\lambda g_{\mu \nu}

        Parameters
        ----------
        coords : list
            list of sympy symbols to use for coordinates

        g : sympy Array
            rank 2 Array containing the full covariant metric

        co : list
            list contain the indices corresponding for Christoffel symbol entry
            co ---> [ \lambda, \mu, \nu ]

        Returns
        -------
        : sympy expr
            returns simplified symbolic python expression for the corresponding
            entry

        Examples
        --------
        >>> import sympy as sp
        >>> sp.init_printing(pretty_print=False, use_latex=False)
        >>> t, r, theta, phi = sp.symbols("t,r,\theta,\phi")
        >>> RT.set_coordinates([t,r,theta,phi])
        >>> g_co = [
        ...    [ -1, 0, 0, 0 ],
        ...    [ 0, 1, 0, 0 ],
        ...    [ 0, 0, 1, 0 ],
        ...    [ 0, 0, 0, 1 ]
        ... ]
        >>> RT.set_metric(g_co)
        >>> g_co, g_contra = RT.get_metric()
        >>> RT._comp_Christoffel_1st([t,r,theta,phi], g_co, [0,0,0])
        0
        >>> RT._comp_Christoffel_1st([t,r,theta,phi], g_co, [1,1,1])
        0

        >>> Phi = sp.Function("\Phi")(r)
        >>> Lambda = sp.Function("\Lambda")(r)
        >>> g_co = [
        ...     [ -sp.exp(2*Phi), 0, 0, 0 ],
        ...     [ 0, sp.exp(2*Lambda), 0, 0 ],
        ...     [ 0, 0, r**2, 0 ],
        ...     [ 0, 0, 0, (r*sp.sin(theta))**2 ]
        ... ]
        >>> RT.set_metric(g_co)
        >>> g_co, g_contra = RT.get_metric()
        >>> RT._comp_Christoffel_1st([t,r,theta,phi], g_co, [0,0,0])
        0
        >>> RT._comp_Christoffel_1st([t,r,theta,phi], g_co, [1,1,1])
        1.0*exp(2*\Lambda(r))*Derivative(\Lambda(r), r)
        >>> RT._comp_Christoffel_1st([t,r,theta,phi], g_co, [2,3,3])
        -0.5*r**2*sin(2*heta)
        """

        #~ TODO input checks

        return sp.simplify(
            1/2 * (
                sp.diff( g_co[ co[2], co[0] ], coords[co[1]] )
                + sp.diff( g_co[ co[1], co[0] ], coords[co[2]] )
                - sp.diff( g_co[ co[1], co[2] ], coords[co[0]] )
            )
        )

    @staticmethod
    def _get_Christoffel_1st(coords, g_co):
        """
        computes each term of Christoffel symbols of 1st kind with
        _comp_Christoffel_1st static method provided in the same class for
        provided coords and covariant metric;
        returns the object as rank 3 array

        Parameters
        ----------
        coords : list
            list of sympy symbols to use as coordinates, the len is the dim

        g : sympy array
            the full covariant metric as sympy array of rank 2

        Returns
        -------
        Christoffel_1st : sympy array
            sympy array of rank 3, see _comp_Christoffel_1st for the formula
            and arrangement of the indices

        Examples
        --------
        >>> import sympy as sp
        >>> sp.init_printing(pretty_print=False, use_latex=False)
        >>> t, r, theta, phi = sp.symbols("t,r,\theta,\phi")
        >>> RT.set_coordinates([t,r,theta,phi])

        >>> Phi = sp.Function("\Phi")(r)
        >>> Lambda = sp.Function("\Lambda")(r)
        >>> g_co = [
        ...     [ -sp.exp(2*Phi), 0, 0, 0 ],
        ...     [ 0, sp.exp(2*Lambda), 0, 0 ],
        ...     [ 0, 0, r**2, 0 ],
        ...     [ 0, 0, 0, (r*sp.sin(theta))**2 ]
        ... ]
        >>> RT.set_metric(g_co)
        >>> g_co, g_contra = RT.get_metric()
        >>> Chris_1 = RT._get_Christoffel_1st([t,r,theta,phi], g_co)

        >>> Chris_1[0,0,0]
        0
        >>> Chris_1[1,1,1]
        1.0*exp(2*\Lambda(r))*Derivative(\Lambda(r), r)
        >>> Chris_1[2,3,3]
        -0.5*r**2*sin(2*heta)
        >>> Chris_1[0,0,1]
        -1.0*exp(2*\Phi(r))*Derivative(\Phi(r), r)
        >>> Chris_1[0,1,0]
        -1.0*exp(2*\Phi(r))*Derivative(\Phi(r), r)
        """

        #~ TODO input checks

        Christoffel_1st = sp.Array( [
            [
                [
                    toolkit._comp_Christoffel_1st(
                        coords, g_co, [co_1, co_2, co_3]
                    ) for co_3 in range(len(coords))
                ] for co_2 in range(len(coords))
            ] for co_1 in range(len(coords))
        ] )

        return Christoffel_1st

    @staticmethod
    def _get_Christoffel_2nd(coords, g_co, g_contra, Christoffel_1st = None):
        """
        returns Christoffel symbol of 2nd kind by contracting with g_contra
        the firstly computed Christoffel symbol of 1st kind using g_co

        latex formula
            \Gamma^{\sigma}_{\mu\nu} = g^{\sigma\lambda}\Gamma_{\lambda\mu\nu}

        Parameters
        ----------
        coords : list
            list of sympy symbols to use as coordinates, the len is the dim

        g_co : sympy array
            the full covariant metric as sympy array of rank 2

        g_contra : sympy array
            the full contravariant metric as sympy array of rank 2

        Returns
        -------
        Christoffel_2nd : sympy array
            sympy array of rank 3, in contracts Christoffel_1st symbols wrt
            1st index with the contravariant metric, see Christoffel_1st for
            definitions

        Examples
        --------
        >>> import sympy as sp
        >>> sp.init_printing(pretty_print=False, use_latex=False)
        >>> t, r, theta, phi = sp.symbols("t,r,\theta,\phi")
        >>> RT.set_coordinates([t,r,theta,phi])

        >>> Phi = sp.Function("\Phi")(r)
        >>> Lambda = sp.Function("\Lambda")(r)
        >>> g_co = [
        ...     [ -sp.exp(2*Phi), 0, 0, 0 ],
        ...     [ 0, sp.exp(2*Lambda), 0, 0 ],
        ...     [ 0, 0, r**2, 0 ],
        ...     [ 0, 0, 0, (r*sp.sin(theta))**2 ]
        ... ]
        >>> RT.set_metric(g_co)
        >>> g_co, g_contra = RT.get_metric()
        >>> Chris_2 = RT._get_Christoffel_2nd([t,r,theta,phi], g_co, g_contra)

        >>> Chris_2[0,0,0]
        0
        >>> Chris_2[1,1,1]
        1.0*Derivative(\Lambda(r), r)
        >>> Chris_2[2,3,3]
        -0.5*sin(2*heta)
        >>> Chris_2[0,0,1]
        1.0*Derivative(\Phi(r), r)
        >>> Chris_2[0,1,0]
        1.0*Derivative(\Phi(r), r)
        """

        #~ TODO input checks

        if not Christoffel_1st:

            Christoffel_1st = toolkit._get_Christoffel_1st(coords, g_co)

        Christoffel_2nd = TC( TP(g_contra, Christoffel_1st), (1,2) )

        return sp.simplify(Christoffel_2nd)

    @staticmethod
    def _get_Rieman_tensor(coords, g_co, g_contra, Christoffel_2nd = None):
        """
        return Rieman tensor for provided coordinates, covariant and
        contravariant metrices
        first computes Christoffel_1st kind, after that Christoffel_2nd
        and using the latter one and the definitions above, by contractions
        calculates the Rieman tensor components wrt coords, g_co, g_contra

        latex formula:
            R^{\alpha}_{\beta\mu\nu} =
                \partial_{\mu} \Gamma^{\alpha}_{\beta\nu}
                - \partial_{\nu} \Gamma^{\alpha}_{\beta\mu}
                + \Gamma^{\alpha}_{\sigma\mu}\Gamma^{\sigma}_{\beta\nu}
                - \Gamma^{\alpha}_{\sigma\nu}\Gamma^{\sigma}_{\beta\mu}

        Parameters
        ----------
        coords : list
            list of sympy symbols to use as coordinates, the len is the dim

        g_co : sympy array
            the full covariant metric as sympy array of rank 2

        g_contra : sympy array
            the full contravariant metric as sympy array of rank 2

        Returns
        -------
        RiemanT : sympy array
            sympy array of rank 4 to include the components of Rieman tensor
            with 1st index contravariant, the next 3 covariant, see def above

        Examples
        --------
        >>> import sympy as sp
        >>> sp.init_printing(pretty_print=False, use_latex=False)
        >>> t, r, theta, phi = sp.symbols("t,r,\theta,\phi")
        >>> RT.set_coordinates([t,r,theta,phi])
        >>> Phi = sp.Function("\Phi")(r)
        >>> Lambda = sp.Function("\Lambda")(r)
        >>> g_co = [
        ...     [ -sp.exp(2*Phi), 0, 0, 0 ],
        ...     [ 0, sp.exp(2*Lambda), 0, 0 ],
        ...     [ 0, 0, r**2, 0 ],
        ...     [ 0, 0, 0, (r*sp.sin(theta))**2 ]
        ... ]
        >>> RT.set_metric(g_co)
        >>> g_co, g_contra = RT.get_metric()
        >>> RiemanT = RT._get_Rieman_tensor([t,r,theta,phi], g_co, g_contra)

        >>> RiemanT[0,2,2,0]
        1.0*r*exp(-2*\Lambda(r))*Derivative(\Phi(r), r)
        >>> RiemanT[1,2,2,1]
        -1.0*r*exp(-2*\Lambda(r))*Derivative(\Lambda(r), r)
        >>> RiemanT[1,3,3,1]
        -1.0*r*exp(-2*\Lambda(r))*sin(heta)**2*Derivative(\Lambda(r), r)
        >>> RiemanT[3,0,0,3]
        -1.0*exp(-2*\Lambda(r) + 2*\Phi(r))*Derivative(\Phi(r), r)/r
        >>> RiemanT[0,0,0,0]
        0
        >>> RiemanT[3,2,2,3]
        -1.0 + 1.0*exp(-2*\Lambda(r))
        """

        #~ TODO input checks

        if not Christoffel_2nd:
            Christoffel_2nd = toolkit._get_Christoffel_2nd(coords, g_co, g_contra)

        Christoffel_2nd_deriv = sp.derive_by_array(Christoffel_2nd, coords)

        Christoffel_2nd_contr = TC(
            TP(Christoffel_2nd, Christoffel_2nd), (1,3)
        )

        RiemanT = sp.Array( [
            [
                [
                    [
                        Christoffel_2nd_deriv[
                            co_2, contra_1, co_1, co_3
                        ]
                        - Christoffel_2nd_deriv[
                            co_3, contra_1, co_1, co_2
                        ]
                        + Christoffel_2nd_contr[
                            contra_1, co_2, co_1, co_3
                        ]
                        - Christoffel_2nd_contr[
                            contra_1, co_3, co_1, co_2
                        ]

                        for co_3 in range(len(coords))
                    ] for co_2 in range(len(coords))
                ] for co_1 in range(len(coords))
            ] for contra_1 in range(len(coords))
        ] )

        return sp.simplify(RiemanT)

    @staticmethod
    def _get_Ricci_tensor_co(coords, g_co, g_contra, RiemanT = None):
        """
        return full covariant Ricci tensor for provided coordinates, covariant
        and contravariant metrices
        first computes Christoffel_1st kind, after that Christoffel_2nd
        and using the latter one and the definitions above, by contractions
        calculates the Rieman tensor components wrt coords, g_co, g_contra
        after that contracts rimena first index(contravariant) and third index
        (covariant)

        latex formula:
            R_{\beta\nu} = R^{\sigma}_{\beta\sigma\nu}

        Parameters
        ----------
        coords : list
            list of sympy symbols to use as coordinates, the len is the dim

        g_co : sympy array
            the full covariant metric as sympy array of rank 2

        g_contra : sympy array
            the full contravariant metric as sympy array of rank 2

        Returns
        -------
        RicciT_co : sympy array
            sympy array of rank 2 to include the components of full covariant
            Ricci tensor

        Examples
        --------
        >>> import sympy as sp
        >>> sp.init_printing(pretty_print=False, use_latex=False)
        >>> t, r, theta, phi = sp.symbols("t,r,\theta,\phi")
        >>> RT.set_coordinates([t,r,theta,phi])
        >>> Phi = sp.Function("\Phi")(r)
        >>> Lambda = sp.Function("\Lambda")(r)
        >>> g_co = [
        ...     [ -sp.exp(2*Phi), 0, 0, 0 ],
        ...     [ 0, sp.exp(2*Lambda), 0, 0 ],
        ...     [ 0, 0, r**2, 0 ],
        ...     [ 0, 0, 0, (r*sp.sin(theta))**2 ]
        ... ]
        >>> RT.set_metric(g_co)
        >>> g_co, g_contra = RT.get_metric()
        >>> RicciT = RT._get_Ricci_tensor_co([t,r,theta,phi], g_co, g_contra)

        >>> RicciT[0,0]
        (1.0*r*(-Derivative(\Lambda(r), r)*Derivative(\Phi(r), r) + Derivative(\Phi(r), r)**2 + Derivative(\Phi(r), r, r)) + 2.0*Derivative(\Phi(r), r))*exp(-2*\Lambda(r) + 2*\Phi(r))/r

        >>> RicciT[1,1]
        (1.0*r*(Derivative(\Lambda(r), r)*Derivative(\Phi(r), r) - Derivative(\Phi(r), r)**2 - Derivative(\Phi(r), r, r)) + 2.0*Derivative(\Lambda(r), r))/r

        >>> RicciT[2,2]
        1.0*(r*Derivative(\Lambda(r), r) - r*Derivative(\Phi(r), r) + exp(2*\Lambda(r)) - 1)*exp(-2*\Lambda(r))

        >>> RicciT[3,3]
        1.0*(r*Derivative(\Lambda(r), r) - r*Derivative(\Phi(r), r) + exp(2*\Lambda(r)) - 1)*exp(-2*\Lambda(r))*sin(heta)**2

        >>> RicciT[1,2]
        0
        """

        #~ TODO input checks
        if not RiemanT:
            RiemanT = toolkit._get_Rieman_tensor(coords, g_co, g_contra)

        RicciT = TC( RiemanT, (0,2) )

        return sp.simplify(RicciT)

    @staticmethod
    def _get_Ricci_scalar(coords, g_co, g_contra, RicciT_co = None):
        """
        return Ricci scalar for provided coordinates, covariant
        and contravariant metrices
        first computes Christoffel_1st kind, after that Christoffel_2nd
        and using the latter one and the definitions above, by contractions
        calculates the Rieman tensor components wrt coords, g_co, g_contra
        after that contracts rimena first index(contravariant) and third index
        (covariant)
        after that contracts with contra variant metric the Ricci tensor

        latex formula:
            R = g^{\beta\nu} R_{\beta\nu}

        Parameters
        ----------
        coords : list
            list of sympy symbols to use as coordinates, the len is the dim

        g_co : sympy array
            the full covariant metric as sympy array of rank 2

        g_contra : sympy array
            the full contravariant metric as sympy array of rank 2

        Returns
        -------
        RicciS : sympy array
            sympy expression for the Ricci scalar

        Examples
        --------
        >>> import sympy as sp
        >>> sp.init_printing(pretty_print=False, use_latex=False)
        >>> t, r, theta, phi = sp.symbols("t,r,\theta,\phi")
        >>> RT.set_coordinates([t,r,theta,phi])
        >>> Phi = sp.Function("\Phi")(r)
        >>> Lambda = sp.Function("\Lambda")(r)
        >>> g_co = [
        ...     [ -sp.exp(2*Phi), 0, 0, 0 ],
        ...     [ 0, sp.exp(2*Lambda), 0, 0 ],
        ...     [ 0, 0, r**2, 0 ],
        ...     [ 0, 0, 0, (r*sp.sin(theta))**2 ]
        ... ]
        >>> RT.set_metric(g_co)
        >>> g_co, g_contra = RT.get_metric()
        >>> RicciS = RT._get_Ricci_scalar([t,r,theta,phi], g_co, g_contra)

        >>> RicciS
        1.0*(2.0*r**2*Derivative(\Lambda(r), r)*Derivative(\Phi(r), r) - 2.0*r**2*Derivative(\Phi(r), r)**2 - 2.0*r**2*Derivative(\Phi(r), r, r) + 4.0*r*Derivative(\Lambda(r), r) - 4.0*r*Derivative(\Phi(r), r) + 2.0*exp(2*\Lambda(r)) - 2.0)*exp(-2*\Lambda(r))/r**2
        """

        #~ TODO input checks
        if not RicciT_co:
            RicciT_co = toolkit._get_Ricci_tensor_co(coords, g_co, g_contra)

        RicciS = TP( g_contra, RicciT_co )

        RicciS = TC( RicciS, (0,2) )
        RicciS = TC( RicciS, (0,1) )

        return sp.simplify(RicciS)

    @staticmethod
    def _get_Einstein_tensor_co(
        coords, g_co, g_contra, RicciT = None, RicciS = None
    ):
        """
        return full covariant Einstein tensor for provided coordinates,
        covariant and contravariant metrices
        first computes Christoffel_1st kind, after that Christoffel_2nd
        and using the latter one and the definitions above, by contractions
        calculates the Rieman tensor components wrt coords, g_co, g_contra
        after that contracts rimena first index(contravariant) and third index
        (covariant)
        after that contracts with contra variant metric the Ricci tensor
        after that use definition for computing the covariant einstein tensor

        latex formula:
            G_{\mu\nu} = R_{\mu\nu} - \dfrac{1}{2}g_{\mu\nu}R

        Parameters
        ----------
        coords : list
            list of sympy symbols to use as coordinates, the len is the dim

        g_co : sympy array
            the full covariant metric as sympy array of rank 2

        g_contra : sympy array
            the full contravariant metric as sympy array of rank 2

        Returns
        -------
        EinsteinT_co : sympy array
            rank 2 sympy array which contains the components for Einstein tensor

        Examples
        --------
        >>> import sympy as sp
        >>> sp.init_printing(pretty_print=False, use_latex=False)
        >>> t, r, theta, phi = sp.symbols("t,r,\theta,\phi")
        >>> RT.set_coordinates([t,r,theta,phi])
        >>> Phi = sp.Function("\Phi")(r)
        >>> Lambda = sp.Function("\Lambda")(r)
        >>> g_co = [
        ...     [ -sp.exp(2*Phi), 0, 0, 0 ],
        ...     [ 0, sp.exp(2*Lambda), 0, 0 ],
        ...     [ 0, 0, r**2, 0 ],
        ...     [ 0, 0, 0, (r*sp.sin(theta))**2 ]
        ... ]
        >>> RT.set_metric(g_co)
        >>> g_co, g_contra = RT.get_metric()
        >>> EinsteinT = RT._get_Einstein_tensor_co([t,r,theta,phi], g_co, g_contra)

        >>> EinsteinT[0,0]
        (2.0*r*Derivative(\Lambda(r), r) + 1.0*exp(2*\Lambda(r)) - 1.0)*exp(-2*\Lambda(r) + 2*\Phi(r))/r**2

        >>> EinsteinT[1,1]
        (2.0*r*Derivative(\Phi(r), r) - 1.0*exp(2*\Lambda(r)) + 1.0)/r**2

        >>> EinsteinT[2,2]
        1.0*r*(-r*Derivative(\Lambda(r), r)*Derivative(\Phi(r), r) + r*Derivative(\Phi(r), r)**2 + r*Derivative(\Phi(r), r, r) - Derivative(\Lambda(r), r) + Derivative(\Phi(r), r))*exp(-2*\Lambda(r))

        >>> EinsteinT[3,3]
        1.0*r*(-r*Derivative(\Lambda(r), r)*Derivative(\Phi(r), r) + r*Derivative(\Phi(r), r)**2 + r*Derivative(\Phi(r), r, r) - Derivative(\Lambda(r), r) + Derivative(\Phi(r), r))*exp(-2*\Lambda(r))*sin(heta)**2
        """

        #~ TODO input checks

        if not RicciT:
            RicciT = toolkit._get_Ricci_tensor_co(coords, g_co, g_contra)

        if not RicciS:
            RicciS = toolkit._get_Ricci_scalar(coords, g_co, g_contra)

        EinsteinT = RicciT - (1/2)*RicciS*g_co

        return sp.simplify(EinsteinT)

#################################################################################
#~ NABLA PART
#################################################################################
#~ private methods to compute covariant derivative in some cases
#################################################################################

    @staticmethod
    def _nabla_scalar( phi, coords ):
        """
        return the covariant derivative of scalar field phi wrt provided list
        of sympy symbols which are the basis

        Parameters
        ----------
        phi : sympy epxr
            the scalar field, which may be function of the coords

        coords : list
            list containing the coordinates as sympy symbols

        Returns
        -------
            : sympy array of rank 1

        """

        #~ TODO include examples

        return sp.simplify(
            sp.derive_by_array( phi, coords )
        )

    @staticmethod
    def _nabla_T_1contra( T, Chris_2, coords ):
        """
        return the covariant derivative of tensor of rank 1 and type (1,0)

        Parameters
        ----------
        T : sympy array of rank 1

        Chris_2 : sympy array of rank 3
            Christoffel symbol of 2nd kind wrt to coords of interenst,

        coords : list
            sympy symbols coordinates in list

        Returns
        -------
            : sympy array of rank 2
        """

        #~ TODO include examples

        T_deriv = sp.derive_by_array(T, coords)

        Chris_T_contra_1 = TC( TP(Chris_2, T), (2,3) )

        res = sp.Array([
            [
                T_deriv[nabla_i, contra_1]

                + Chris_T_contra_1[ contra_1, nabla_i ]

                for contra_1 in range(len(coords))
            ] for nabla_i in range(len(coords))
        ] )

        return sp.simplify( res )

    @staticmethod
    def _nabla_T_1co( T, Chris_2, coords ):
        """
        return the covariant derivative of tensor of rank 1 and type (0,1)

        Parameters
        ----------
        T : sympy array of rank 1

        Chris_2 : sympy array of rank 3
            Christoffel symbol of 2nd kind wrt to coords of interenst,

        coords : list
            sympy symbols coordinates in list

        Returns
        -------
            : sympy array of rank 2
        """

        #~ TODO include examples

        T_deriv = sp.derive_by_array(T, coords)

        Chris_T_co_1 = TC( TP(Chris_2, T), (0,3) )

        res = sp.Array( [
            [
                T_deriv[nabla_i, co_1]

                - Chris_T_co_1[ nabla_i, co_1 ]

                for co_1 in range(len(coords))
            ] for nabla_i in range(len(coords))
        ] )

        return sp.simplify( res )

    @staticmethod
    def _nabla_T_1contra_1co( T, Chris_2, coords ):
        """
        return the covariant derivative of tensor of rank 1 and type (1,1),
        where first index is contravariant, second is covariant

        T^{a}_{b}

        INDEX PLACEMENT IS FIXED, for other arrengment see other _nabla funcs

        Parameters
        ----------
        T : sympy array of rank 2 of type T^{a}_{b}

        Chris_2 : sympy array of rank 3
            Christoffel symbol of 2nd kind wrt to coords of interenst,

        coords : list
            sympy symbols coordinates in list

        Returns
        -------
            : sympy array of rank 2
        """

        #~ TODO include examples

        T_deriv = sp.derive_by_array(T, coords)

        Chris_T_contra_1 = TC( TP(Chris_2, T), (2,3))
        Chris_T_co_1 = TC( TP(Chris_2, T), (0,4) )

        res = sp.Array( [
            [
                [
                    T_deriv[nabla_i, contra_1, co_1]

                    + Chris_T_contra_1[contra_1, nabla_i, co_1 ]

                    - Chris_T_co_1[nabla_i, co_1, contra_1 ]

                    for co_1 in range(len(coords))
                ] for contra_1 in range(len(coords))
            ] for nabla_i in range(len(coords))
        ] )

        return sp.simplify( res )

    @staticmethod
    def _nabla_T_1co_1contra( T, Chris_2, coords ):
        """
        return the covariant derivative of tensor of rank 1 and type (1,1),
        where first index is covariant, second is contravariant

        T_{a}^{b}

        INDEX PLACEMENT IS FIXED, for other arrengment see other _nabla funcs

        Parameters
        ----------
        T : sympy array of rank 2 of type T_{a}^{b}

        Chris_2 : sympy array of rank 3
            Christoffel symbol of 2nd kind wrt to coords of interenst,

        coords : list
            sympy symbols coordinates in list

        Returns
        -------
            : sympy array of rank 2
        """

        #~ TODO include examples

        T_deriv = sp.derive_by_array(T, coords)

        Chris_T_co_1 = TC( TP(Chris_2, T), (0,3) )
        Chris_T_contra_1 = TC( TP(Chris_2, T), (2,4))

        res = sp.Array( [
            [
                [
                    T_deriv[nabla_i, co_1, contra_1]

                    - Chris_T_co_1[nabla_i, co_1, contra_1 ]

                    + Chris_T_contra_1[contra_1, nabla_i, co_1 ]

                    for contra_1 in range(len(coords))
                ] for co_1 in range(len(coords))
            ] for nabla_i in range(len(coords))
        ] )

        return sp.simplify( res )

    @staticmethod
    def _nabla_T_2co( T, Chris_2, coords ):
        """
        return the covariant derivative of tensor of rank 1 and type (0,2),
        where first index is covariant, second is contravariant

        T_{ab}

        INDEX PLACEMENT IS FIXED, for other arrengment see other _nabla funcs

        Parameters
        ----------
        T : sympy array of rank 2 of type T_{ab}

        Chris_2 : sympy array of rank 3
            Christoffel symbol of 2nd kind wrt to coords of interenst,

        coords : list
            sympy symbols coordinates in list

        Returns
        -------
            : sympy array of rank 2
        """

        #~ TODO include examples

        T_deriv = sp.derive_by_array(T, coords)

        Chris_T_co_1 = TC( TP(Chris_2, T), (0,3) )
        Chris_T_co_2 = TC( TP(Chris_2, T), (0,4))

        res = sp.Array( [
            [
                [
                    T_deriv[nabla_i, co_1, co_2]

                    - Chris_T_co_1[nabla_i, co_1, co_2 ]

                    - Chris_T_co_2[nabla_i, co_2, co_1 ]

                    for co_2 in range(len(coords))
                ] for co_1 in range(len(coords))
            ] for nabla_i in range(len(coords))
        ] )

        return sp.simplify( res )

    @staticmethod
    def _nabla_T_2contra( T, Chris_2, coords ):
        """
        return the covariant derivative of tensor of rank 1 and type (2,0),
        where first index is covariant, second is contravariant

        T^{ab}

        INDEX PLACEMENT IS FIXED, for other arrengment see other _nabla funcs

        Parameters
        ----------
        T : sympy array of rank 2 of type T^{ab}

        Chris_2 : sympy array of rank 3
            Christoffel symbol of 2nd kind wrt to coords of interenst,

        coords : list
            sympy symbols coordinates in list

        Returns
        -------
            : sympy array of rank 2
        """

        #~ TODO include examples

        T_deriv = sp.derive_by_array(T, coords)

        Chris_T_contra_1 = TC( TP(Chris_2, T), (2,3) )
        Chris_T_contra_2 = TC( TP(Chris_2, T), (2,4))

        res = sp.Array( [
            [
                [
                    T_deriv[nabla_i, contra_1, contra_2]

                    + Chris_T_contra_1[contra_1, nabla_i, contra_2 ]

                    + Chris_T_contra_2[contra_2, nabla_i, contra_1 ]

                    for contra_2 in range(len(coords))
                ] for contra_1 in range(len(coords))
            ] for nabla_i in range(len(coords))
        ] )

        return sp.simplify( res )

    @staticmethod
    def _nabla_T_3co( T, Chris_2, coords ):
        """
        return the covariant derivative of tensor of rank 1 and type (2,0),
        where first index is covariant, second is contravariant

        T_{abc}

        INDEX PLACEMENT IS FIXED, for other arrengment see other _nabla funcs

        Parameters
        ----------
        T : sympy array of rank 3 of type T_{abc}

        Chris_2 : sympy array of rank 3
            Christoffel symbol of 2nd kind wrt to coords of interenst,

        coords : list
            sympy symbols coordinates in list

        Returns
        -------
            : sympy array of rank 2
        """

        #~ TODO include examples

        T_deriv = sp.derive_by_array(T, coords)

        Chris_T_co_1 = TC( TP(Chris_2, T), (0,3) )
        Chris_T_co_2 = TC( TP(Chris_2, T), (0,4) )
        Chris_T_co_3 = TC( TP(Chris_2, T), (0,5) )

        res = sp.Array( [
            [
                [
                    [

                        T_deriv[nabla_i, co_1, co_2, co_3 ]

                        - Chris_T_co_1[nabla_i, co_1, co_2, co_3 ]

                        - Chris_T_co_2[nabla_i, co_2, co_1, co_3 ]

                        - Chris_T_co_3[nabla_i, co_3, co_1, co_2 ]

                        for co_3 in range(len(coords))
                    ] for co_2 in range(len(coords))
                ] for co_1 in range(len(coords))
            ] for nabla_i in range(len(coords))
        ] )

        return sp.simplify( res )

    @staticmethod
    def _nabla_T_1contra_2co( T, Chris_2, coords ):
        """
        return the covariant derivative of tensor of rank 3 and type (1,2),
        where first index is contravariant, the rest are covariant

        T^{a}_{bc}

        INDEX PLACEMENT IS FIXED, for other arrengment see other _nabla funcs

        Parameters
        ----------
        T : sympy array of rank 3 of type T^{a}_{bc}

        Chris_2 : sympy array of rank 3
            Christoffel symbol of 2nd kind wrt to coords of interenst,

        coords : list
            sympy symbols coordinates in list

        Returns
        -------
            : sympy array of rank 2
        """

        #~ TODO include examples

        T_deriv = sp.derive_by_array(T, coords)

        Chris_T_contra_1 = TC( TP(Chris_2, T), (2,3) )
        Chris_T_co_1 = TC( TP(Chris_2, T), (0,4) )
        Chris_T_co_2 = TC( TP(Chris_2, T), (0,5) )

        res = sp.Array( [
            [
                [
                    [

                        T_deriv[nabla_i, contra_1, co_1, co_2 ]

                        + Chris_T_contra_1[contra_1, nabla_i, co_1, co_2 ]

                        - Chris_T_co_1[nabla_i, co_1, contra_1, co_2 ]

                        - Chris_T_co_2[nabla_i, co_2, contra_1, co_1 ]

                        for co_2 in range(len(coords))
                    ] for co_1 in range(len(coords))
                ] for contra_1 in range(len(coords))
            ] for nabla_i in range(len(coords))
        ] )

        return sp.simplify( res )

if __name__ == "__main__":

    import doctest

    doctest.testmod(
        extraglobs = {
            "RT": toolkit(),
        }
    )


    #~ imp.reload(RTm);
    #~ RT = RTm.toolkit();

    #~ t, r, thetha, phi = sp.symbols("t,r,\\theta, \phi");

    #~ coords = [t,r,thetha, phi]

    #~ RT.set_coordinates(coords)

    #~ Phi = sp.Function("\Phi")(r)

    #~ Lambda = sp.Function("\Lambda")(r)

    #~ g_co = [
            #~ [ -sp.exp(2*Phi), 0, 0, 0 ],
            #~ [ 0, sp.exp(2*Lambda), 0, 0 ],
            #~ [ 0, 0, r**2, 0 ],
            #~ [ 0, 0, 0, (r*sp.sin(theta))**2 ]
    #~ ];

    #~ RT.set_metric(g_co);
    #~ g_co, g_contra = RT.get_metric();

    #~ G_co = RT.get_Einstein_tensor_co();

    #~ rho = sp.Function("\\rho")(r)
    #~ RT.set_density(rho)

    #~ p = sp.Function("p")(rho)
    #~ RT.set_pressure(p)

    #~ varphi = sp.Function("\\varphi")(r)
    #~ V = sp.Function("V")(varphi)
    #~ RT.set_scalarField(varphi, V)

    #~ RT.set_u_contra([sp.exp(-Phi), 0, 0, 0]);
    #~ u_co, u_contra = RT.get_u();

    #~ delta_Phi = sp.Function("\delta\Phi")(r,t)
    #~ delta_Lambda = sp.Function("\delta\Lambda")(r,t)

    #~ h_co = [
        #~ [ -2*sp.exp(2*Phi)*delta_Phi, 0, 0, 0 ],
        #~ [ 0, 2*sp.exp(2*Lambda)*delta_Lambda, 0, 0 ],
        #~ [ 0, 0, 0, 0 ],
        #~ [ 0, 0, 0, 0 ]
    #~ ]

    #~ RT.set_hmetric(h_co)
    #~ h_co, h_contra = RT.get_hmetric()

    #~ hG_co = RT.get_hEinstein_tensor_co()

    #~ eps = sp.symbols("\\varepsilon")

    #~ epsG_co = [
            #~ [ -sp.exp(2*( Phi + eps*delta_Phi)), 0, 0, 0 ],
            #~ [ 0, sp.exp(2*(Lambda + eps*delta_Lambda)), 0, 0 ],
            #~ [ 0, 0, r**2, 0 ],
            #~ [ 0, 0, 0, (r*sp.sin(theta))**2 ]
    #~ ];

    #~ RT.set_epsGmetric(eps, epsG_co);
    #~ epsG_co, epsG_contra = RT.get_epsGmetric();

    #~ epsG_co = RT.get_epsEinstein_tensor_co();

    #~ display(G_co[0,0] + eps*hG_co[0,0]);
    #~ display(epsG_co[0,0]);

    #~ imp.reload(RTm);

    #~ RT = RTm.toolkit();

    #~ t, r, thetha, phi = sp.symbols("t,r,\\theta, \phi");

    #~ coords = [t,r,thetha, phi]

    #~ RT.set_coordinates(coords)

    #~ Phi = sp.Function("\Phi")(r)

    #~ Lambda = sp.Function("\Lambda")(r)

    #~ g_co = [
            #~ [ -sp.exp(2*Phi), 0, 0, 0 ],
            #~ [ 0, sp.exp(2*Lambda), 0, 0 ],
            #~ [ 0, 0, r**2, 0 ],
            #~ [ 0, 0, 0, (r*sp.sin(theta))**2 ]
    #~ ];

    #~ RT.set_metric(g_co);
    #~ g_co, g_contra = RT.get_metric();

    #~ G_co = RT.get_Einstein_tensor_co();

    #~ rho = sp.Function("\\rho")(r)
    #~ RT.set_density(rho)

    #~ p = sp.Function("p")(rho)
    #~ RT.set_pressure(p)

    #~ varphi = sp.Function("\\varphi")(r)
    #~ V = sp.Function("V")(varphi)
    #~ RT.set_scalarField(varphi, V)

    #~ RT.set_u_contra([sp.exp(-Phi), 0, 0, 0]);
    #~ u_co, u_contra = RT.get_u();

    #~ delta_Phi = sp.Function("\delta\Phi")(r,t)
    #~ delta_Lambda = sp.Function("\delta\Lambda")(r,t)

    #~ h_co = [
        #~ [ -2*sp.exp(2*Phi)*delta_Phi, 0, 0, 0 ],
        #~ [ 0, 2*sp.exp(2*Lambda)*delta_Lambda, 0, 0 ],
        #~ [ 0, 0, 0, 0 ],
        #~ [ 0, 0, 0, 0 ]
    #~ ]

    #~ RT.set_hmetric(h_co)
    #~ h_co, h_contra = RT.get_hmetric()

    #~ xi = sp.Function("\\xi")(r,t)
    #~ xi_contra = sp.Array([ 0, xi, 0, 0 ])
    #~ RT.set_hU_contra(xi_contra)
    #~ hU_co, hU_contra = RT.get_hU()

    #~ delta_rho = sp.Function("\delta\\rho")(r)
    #~ RT.set_hDensity(delta_rho)

    #~ delta_p = sp.Function("\delta p")(delta_rho)
    #~ RT.set_hPressure(delta_p)

    #~ delta_varphi = sp.Function("\delta\\varphi")(r,t)
    #~ delta_V = sp.Function("\delta V")(varphi)
    #~ RT.set_hScalarField(delta_varphi, delta_V)

    #~ lhs, rhs = RT.get_hScalarField_field_eq()

    #~ eps = sp.symbols("\\varepsilon")
    #~ epsG_co = [
        #~ [ -sp.exp(2*( Phi + eps*delta_Phi)), 0, 0, 0 ],
        #~ [ 0, sp.exp(2*(Lambda + eps*delta_Lambda)), 0, 0 ],
        #~ [ 0, 0, r**2, 0 ],
        #~ [ 0, 0, 0, (r*sp.sin(theta))**2 ]
    #~ ]
    #~ RT.set_epsGmetric(eps, epsG_co);
    #~ epsG_co, epsG_contra = RT.get_epsGmetric();

    #~ epsVarphi = sp.Function("\delta\\varphi")(r,t)
    #~ RT.set_epsScalarField(varphi + eps*epsVarphi, 0)
    #~ lhs_scalar = RT.get_epsScalarField_field_eq()
