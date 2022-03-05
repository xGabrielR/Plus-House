import re
import pickle
import inflection
import pandas as pd
import streamlit as st
import plotly.express as px

from datetime import datetime
from plushouse import PlusHouse

@st.cache
def convert(df):
    return df.to_csv().encode('utf-8')

class PlusApp():

    def __init__( self ):
        self.pl = PlusHouse()
        self.model = pickle.load( open( 'skl/xgb_tunned.pkl', 'rb' ) )
        
    def pre_html( self ): # Try to Use a External Style
        html='''
        <style>
            * { padding: 1.5px; }
            p {color: #428df5; }
            ::selection { color: #b950ff; }
            h1 {color: #7033ff; text-align: center; }
            h2 {color: #8d5dfc}
        </style>
        '''
        st.markdown( html, unsafe_allow_html=True )

    def plot_line( self ):
        simple_line = '''
        <style>
        .line { position: absolute; width: 330px; left: 50px; opacity: 15%; height: 20px; top: 80px; background: rgb(253, 250, 255 ); } 
        </style>
        <div class='line'></div>'''
        st.markdown( simple_line, unsafe_allow_html=True )

        return None

    def prepare_base_dataset( self ):
        df_on_base = pd.read_csv( 'test.csv' )

        df = self.pl.clean_dataset( df_on_base )
        df = self.pl.data_preparation( df )
        df = self.pl.get_prediction( self.model, df_on_base, df )
        df.prediction = df.prediction.apply( lambda x: 'R$ ' + str( int(x) ) + ',00' )

        results = {'df_raw': df_on_base, 'df_pred': df}

        return results

    def app_header( self ):
        st.title('Hello CEO!!')
        st.write('\n\n')

        hour = int( datetime.now().hour )
        if (hour <= 12) & (hour >= 0):
            st.subheader('Have a Good Morning 😁')
        if (hour > 12) & (hour <= 17):
            st.subheader('Have a Good Afternoon 😊')
        if (hour > 17):
            st.subheader('Have a Good Night 😴')

        st.write('What would you like to do ?')


    def dataset_app( self, df_raw, df ):
        self.plot_line()
        header = '''<h2>❏ Current Dataset on App Base</h2>'''
        st.markdown( header, unsafe_allow_html=True )

        st.dataframe( df_raw.head() )

        st.subheader( 'Predictions on This Dataset' )

        house_id = st.text_input('Please CEO, Provide House Id')

        try:
            house_id = int( house_id )

        except ValueError:
            st.write('Please, Provide a Valid Property Id or Blank for Random 5 Properties')

        dfx = df[df.id == house_id]

        if dfx.empty:
            st.dataframe( df.sample(5)[['id', 'prediction', 'lot_area', 'condition1', 'neighborhood']] )
                
        else:
            st.dataframe( dfx[['id', 'prediction', 'lot_area', 'condition1', 'neighborhood']] )

        df['prediction'] = df['prediction'].apply( lambda x: int(re.match( '\d+', x[3:] ).group(0)) )
        dfaux = df[['neighborhood', 'prediction']].groupby('neighborhood').sum().reset_index()

        fig = px.bar( dfaux, x='neighborhood', y='prediction', text_auto='.2s', \
                    title='Predictions for each Neighborhood', width=700, height=500, \
                    color_discrete_sequence=["black"], hover_name='prediction' )

        fig.update_traces( textfont_size=20, textangle=0, textposition="outside", cliponaxis=False )
        fig.update_layout( plot_bgcolor='#0e1117' )
        st.plotly_chart( fig, use_container_width=True )

        return None

    def sender_dataset( self ):
        self.plot_line()
        st.header('❏ Drag and Drop Your Dataset Here')

        file = st.file_uploader( '', type=['csv'])

        if file is None:

            st.write('')

        else:

            df1 = pd.read_csv( file )
            df1.columns = [inflection.underscore( col_name ) for col_name in df1.columns.tolist()]

            cols_requested = ['overall_qual', 'exter_qual', '1st_flr_sf', 
                            'total_bsmt_sf', 'gr_liv_area', 'year_built', 'lot_frontage', 
                            'garage_yr_blt', 'condition1', 'fireplace_qu']

            list_len = [i for i in df1.columns.tolist() if i in cols_requested]

            if not len(list_len) == 10:
                st.write('Please, Provide a Correct Dataset, check Documentation!')
            
            else:
                st.title('Predictions for Your Dataset')

                df = self.pl.clean_dataset( df1 )
                df = self.pl.data_preparation( df )
                df = self.pl.get_prediction( self.model, df1, df )
                df.prediction = df.prediction.apply( lambda x: 'R$ ' + str( int(x) ) + ',00' )

                st.dataframe( df.loc[:5, ['id', 'prediction', 'lot_area', 'condition1', 'neighborhood']] )
                
                csv = convert(df)

                st.download_button('Download The Dataset', csv, 'NewDataset.csv', 'text/csv')

        return None

if __name__ == '__main__':
    
    pa = PlusApp()

    pa.pre_html()

    pa.app_header()

    df = pa.prepare_base_dataset()

    pa.dataset_app( df['df_raw'], df['df_pred'] )

    pa.sender_dataset()