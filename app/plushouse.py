import pickle
import warnings
import inflection
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

class PlusHouse( object ):
    def __init__( self ):
        self.lin_model = pickle.load( open( 'linear_model.pkl', 'rb' ) )
        
        self.overall_qual_scaler     = pickle.load( open( 'overall_qual_scaler.pkl', 'rb' ) )
        self.total_scaler            = pickle.load( open( 'total_scaler.pkl', 'rb' ) )
        self.total_abv_grade_scaler  = pickle.load( open( 'total_abv_scaler.pkl', 'rb' ) )
        self.garage_multy_car_scaler = pickle.load( open( 'garage_multy_car_scaler.pkl', 'rb' ) )
    
        return None

    # def linear_fillna( self, df, na_col, ref_col ):  # Old Version for Documentation
    #     df11 = df[[ref_col, na_col]]

    #     x_train = np.reshape( df11[~df11[na_col].isna()][ref_col].tolist(), (-1, 1) )
    #     y_train = np.reshape( df11[~df11[na_col].isna()][na_col].tolist(), (-1, 1) )
    #     x_test  = np.reshape( df11[df11[na_col].isna()][ref_col].tolist(), (-1, 1) )

    #     lin   = self.lin_model.fit( x_train, y_train )
    #     y_hat = lin.predict( x_test )

    #     df.loc[df[na_col].isna(), na_col] = y_hat

    #     return df

    
    def clean_dataset( self, df ):
        df.columns = [inflection.underscore( p ) for p in df.columns.tolist()]
        
        #df = self.linear_fillna( df, 'lot_frontage', 'lot_area' )
        df['garage_yr_blt'] = df['garage_yr_blt'].fillna( 0 )
        df['fireplace_qu']  = df['fireplace_qu'].fillna('DontHave')
        df.loc[df['garage_yr_blt'] == 0, 'garage_yr_blt'] = 1900
        
        df['total_sqft'] = df.total_bsmt_sf + df.gr_liv_area
        df['total_size_porch']  = df.wood_deck_sf    + df.open_porch_sf + df.enclosed_porch + df['3_ssn_porch'] + df.screen_porch   
        df['garage_multy_car']  = df.garage_area     * df.garage_cars
        df['total_bath']        = df.bsmt_full_bath  + df.full_bath + df.bsmt_half_bath + df.half_bath
        df['total_abv_grade']   = df.tot_rms_abv_grd + df.kitchen_abv_gr + df.bedroom_abv_gr 
        df['floors_sqft']       = df.low_qual_fin_sf + df['2nd_flr_sf']  + df['1st_flr_sf'] + df.total_bsmt_sf
        
        
        cols_selected = ['overall_qual', 'exter_qual', 'total_sqft', 'total_abv_grade', 'total_bath', 'garage_multy_car',
                         'land_slope', 'condition2', 'bldg_type', 'exter_cond', 'neighborhood', 'central_air', 'garage_finish',
                         'foundation', 'bsmt_cond', 'heating_qc', 'paved_drive', 'fireplace_qu']
        
        df = df[cols_selected]
        
        return df
    
    def data_preparation( self, df ):
        
        df['overall_qual']     = self.overall_qual_scaler.fit_transform( df[['overall_qual']].values )
        df['total_sqft']       = self.total_scaler.fit_transform( df[['total_sqft']].values )
        df['total_abv_grade']  = self.total_abv_grade_scaler.fit_transform( df[['total_abv_grade']].values )
        df['garage_multy_car'] = self.garage_multy_car_scaler.fit_transform( df[['garage_multy_car']].values )

        df.exter_qual    = df['exter_qual'].map( {'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0} )
        df.foundation    = df['foundation'].map( {'BrkTil':5, 'CBlock':4, 'PConc':3, 'Slab':2, 'Stone':1, 'Wood':0 } )
        df.land_slope    = df['land_slope'].map( {'Gtl':2, 'Mod':1, 'Sev':0} )
        df.bldg_type     = df['bldg_type'].map( {'1Fam':0, '2fmCon':1, 'Duplex':2, 'Twnhs':3, 'TwnhsE':4 } )
        df.exter_cond    = df['exter_cond'].map( {'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0} )
        df.central_air   = df['central_air'].map( {'Y':1, 'N':0} )
        df.garage_finish = df['garage_finish'].map( {'Fin':3, 'RFn':2, 'Unf':1, 'DontHave':0} )
        df.bsmt_cond     = df['bsmt_cond'].map( {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'DontHave':0} )
        df.heating_qc    = df['heating_qc'].map( {'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0} )
        df.paved_drive   = df['paved_drive'].map( {'Y':2, 'P':1, 'N':0} )
        df.fireplace_qu  = df['fireplace_qu'].map( {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'DontHave':0} )

        f_conditi2 = df.groupby( ['condition2'] ).size() / len(df)
        f_neighbor = df.groupby( ['neighborhood'] ).size() / len(df)

        df.condition2 = df['condition2'].apply(  lambda x: f_conditi2[x] )
        df.neighborhood  = df['neighborhood'].apply(lambda x: f_neighbor[x] )

        return df
    
    def get_prediction( self, model, original_data, test_data ):
        yhat = model.predict( test_data )
        original_data['prediction'] = yhat**3
        original_data['prediction'] = np.round( original_data['prediction'], 3 )

        return original_data