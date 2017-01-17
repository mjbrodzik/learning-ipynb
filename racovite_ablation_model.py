#!/usr/bin/env python
import hypsometry
import pandas as pd
import numpy as np

class racovite_ablation_modelError( Exception ): pass

def run( clean_ice_km2, ela_m, ablation_gradient_m_per_100m, verbose=False ):
    """
    Run the ablation gradient melt model as described by:
    
    Racoviteanu et al., 2014, Evaluation of an ice ablation model to estimate the
    contribution of melting glacier ice to annual discharge in the Nepal Himalaya.  WRR.

    clean_ice_km2 : a clean ice Hypsometry DataFrame e.g. (MODICE hypsometry for a basin).
    ela_m : elevation line altitude (m)
    ablation_gradient_m_per_100m : clean_ice ablation gradient, (m/100m)

    returns melt_by_elevation DataFrame.

    2014-09-30 M. J. Brodzik brodzik@nsidc.org 303-492-8263
    National Snow & Ice Data Center, Boulder CO
    Copyright (C) 2014 Regents of the University of Colorado at Boulder
    
    """

    try:

        # This implementation assumes input contours are 100 m,
        # So verify that input contours are 100 m
        # (otherwise I'll need to do a little more work for the ablation gradient to work)
        contour_size_m = 100.
        contours_m = clean_ice_km2.data.columns.values.astype( np.float )
        diff = contours_m - np.roll( contours_m, 1 )
        diff = diff[ 1: ]
        if contour_size_m != np.amin( diff ) or contour_size_m != np.amax( diff ):
            raise racovite_ablation_modelError( "clean_ice_km2 contours should be 100 m." )

        # Calculate the ablation gradient for each contour below the ELA
        # For contours above the ELA, set b to 0.
        b = -1. * ablation_gradient_m_per_100m * ( contours_m - ela_m ) / contour_size_m
        b[ contours_m > ela_m ] = 0.

        m2_per_km2 = 10.0 ** 6.0
        m3_per_km3 = 10.0 ** 9.0
        clean_ice_melt_km3 = b * ( clean_ice_km2.data * m2_per_km2 ) / m3_per_km3

        if verbose:
            print ( __name__ + ": input clean ice:" )
            print ( clean_ice_km2.data )
            print ( b )
            print ( clean_ice_melt_km3 )

        # Return a new Hypsometry
        return( hypsometry.Hypsometry( comments=["Ablation gradient melt by elevation, km^3"],
                                       data=pd.DataFrame( data=clean_ice_melt_km3,
                                                          index=clean_ice_km2.data.index ) ) )
    
    except racovite_ablation_modelError:
        raise
        
if __name__ == '__main__':
    def selftest():
        print( "racovite_ablation_model: performing regression test:" ),
        modice_filename = "modice_area_by_elev/IN_Indus_at_Kotri.0100m.modicev03_area_by_elev.txt"
        modice = hypsometry.Hypsometry()
        modice.read( modice_filename )
        melt = run( modice, 5000., 0.6 )

        compare_filename = "clean_ice_melt_by_elev/compare.IN_Indus_at_Kotri.0100m.racovite_clean_ice_melt_by_elev.txt"
        compare = hypsometry.Hypsometry()
        compare.read( compare_filename )

        from pandas.util.testing import assert_frame_equal
        try:
            if None == assert_frame_equal( melt.data, compare.data, check_names=True ):
                print( "OK" )
        except AssertionError:
            print( "***** Failed *****" )

    selftest()

        











