import pickle
import warnings
import inflection
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

class PlusHouse( object ):
    def __init__( self ):
        self.lin_model = pickle.load( open( 'linear_model.pkl', 'rb' ) )
        
        self.overall_qual_scaler = pickle.load( open( 'overall_qual_scaler.pkl', 'rb' ) )
        self.st_scaler           = pickle.load( open( '1st_scaler.pkl', 'rb' ) )
        self.garage_scaler       = pickle.load( open( 'garage_scaler.pkl', 'rb' ) )
        self.total_scaler        = pickle.load( open( 'total_scaler.pkl', 'rb' ) )
        self.frontage_scaler     = pickle.load( open( 'frontage_scaler.pkl', 'rb' ) )
        self.bsmt_scaler         = pickle.load( open( 'bsmt_scaler.pkl', 'rb' ) )
        self.gr_scaler           = pickle.load( open( 'gr_scaler.pkl', 'rb' ) )
    
        return None

    def linear_fillna( self, df, na_col, ref_col ):
        df11 = df[[ref_col, na_col]]

        x_train = np.reshape( df11[~df11[na_col].isna()][ref_col].tolist(), (-1, 1) )
        y_train = np.reshape( df11[~df11[na_col].isna()][na_col].tolist(), (-1, 1) )
        x_test  = np.reshape( df11[df11[na_col].isna()][ref_col].tolist(), (-1, 1) )

        lin   = self.lin_model.fit( x_train, y_train )
        y_hat = lin.predict( x_test )

        df.loc[df[na_col].isna(), na_col] = y_hat

        return df

    
    def clean_dataset( self, df ):
        df.columns = [inflection.underscore( p ) for p in df.columns.tolist()]
        
        df = self.linear_fillna( df, 'lot_frontage', 'lot_area' )
        df['garage_yr_blt'] = df['garage_yr_blt'].fillna( 0 )
        df['fireplace_qu']  = df['fireplace_qu'].fillna('DontHave')
        
        df['total_sqft'] = df.total_bsmt_sf + df.gr_liv_area
        df.loc[df['garage_yr_blt'] == 0, 'garage_yr_blt'] = 1900
        cols_selected = ['overall_qual', 'exter_qual', 'total_sqft', '1st_flr_sf', 'total_bsmt_sf', 'gr_liv_area', 'year_built', 'lot_frontage', 'garage_yr_blt', 'condition1', 'fireplace_qu']
        
        df = df[cols_selected]
        
        return df
    
    def data_preparation( self, df ):
        
        df['overall_qual']    = self.overall_qual_scaler.fit_transform( df[['overall_qual']].values )
        df['1st_flr_sf']      = self.st_scaler.fit_transform( df[['1st_flr_sf']].values )
        df['garage_yr_blt']   = self.garage_scaler.fit_transform( df[['garage_yr_blt']].values )
        df['total_sqft']      = self.total_scaler.fit_transform(  df[['total_sqft']].values )
        df['lot_frontage']    = self.frontage_scaler.fit_transform( df[['lot_frontage']].values )
        df['total_bsmt_sf']   = self.bsmt_scaler.fit_transform( df[['total_bsmt_sf']].values )
        df['gr_liv_area']     = self.gr_scaler.fit_transform(   df[['gr_liv_area']].values )

        f_conditi1 = df.groupby( ['condition1'] ).size() / len(df)
        df.condition1   = df['condition1'].apply(  lambda x: f_conditi1[x] )
        df.fireplace_qu = df['fireplace_qu'].map( {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'DontHave':0} )
        df.exter_qual   = df['exter_qual'].map( {'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0} )

        return df
    
    def get_prediction( self, model, original_data, test_data ):
        yhat = model.predict( test_data )
        original_data['prediction'] = yhat**3
        original_data['prediction'] = np.round( original_data['prediction'], 3 )

        return original_data