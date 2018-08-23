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
        self.Christoffel_2nd = None
        self.RiemanT = None
        self.RicciT_co = None
        self.RicciS = None
        self.Einstein_co = None

        #~ the perturbed metric and ect goemtricals
        self.h_co = None
        self.h_contra = None
        self.hChristoffel_2nd = None
        self.hRiemanT = None
        self.hRicciT_co = None
        self.hRicciS = None
        self.hEinstein_co = None

        #~ metric with small parameter to expand around and its geometricals
        self.eps_g_co = None
        self.eps_g_contra = None
        self.epsChristoffel_2nd = None
        self.epsRiemanT = None
        self.epsRicciT_co = None
        self.epsRicciS = None
        self.epsEinstein_co = None

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

    @staticmethod
    def _comp_Christoffel_1st(coords, g, co):
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
                sp.diff( g[ co[2], co[0] ], coords[co[1]] )
                + sp.diff( g[ co[1], co[0] ], coords[co[2]] )
                - sp.diff( g[ co[1], co[2] ], coords[co[0]] )
            )
        )

    @staticmethod
    def _get_Christoffel_1st(coords, g):
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
                        coords, g, [co_1, co_2, co_3]
                    ) for co_3 in range(len(coords))
                ] for co_2 in range(len(coords))
            ] for co_1 in range(len(coords))
        ] )

        return Christoffel_1st

    @staticmethod
    def _get_Christoffel_2nd(coords, g_co, g_contra):
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

        Christoffel_1st = toolkit._get_Christoffel_1st(coords, g_co)

        Christoffel_2nd = TC( TP(g_contra, Christoffel_1st), (1,2) )

        return sp.simplify(Christoffel_2nd)

    @staticmethod
    def _get_Rieman_tensor(coords, g_co, g_contra):
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
    def _get_Ricci_tensor_co(coords, g_co, g_contra):
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

        RiemanT = toolkit._get_Rieman_tensor(coords, g_co, g_contra)

        RicciT = TC( RiemanT, (0,2) )

        return sp.simplify(RicciT)

    @staticmethod
    def _get_Ricci_scalar(coords, g_co, g_contra):
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

        RicciT = toolkit._get_Ricci_tensor_co(coords, g_co, g_contra)

        RicciS = TP( g_contra, RicciT )

        RicciS = TC( RicciS, (0,2) )
        RicciS = TC( RicciS, (0,1) )

        return sp.simplify(RicciS)

    @staticmethod
    def _get_Einstein_tensor_co(coords, g_co, g_contra):
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

        RicciT = toolkit._get_Ricci_tensor_co(coords, g_co, g_contra)
        RicciS = toolkit._get_Ricci_scalar(coords, g_co, g_contra)

        EinsteinT = RicciT - (1/2)*RicciS*g_co

        return sp.simplify(EinsteinT)

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


if __name__ == "__main__":

    import doctest

    doctest.testmod(
        extraglobs = {
            "RT": toolkit(),
        }
    )
