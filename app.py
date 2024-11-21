import streamlit as st
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def parse_year(filename):
    import re
    match = re.search(r'\d+', filename)
    return match.group() if match else None

def add_file_to_df(df, filename):
    # year = parse_year(filename)
    with open('./namesbystate/' + filename,"r") as file:
         for line in file:
            fields = line.split(",")
            df = pd.concat([df, pd.DataFrame(data={"state": str(fields[0]), "gender": str(fields[1]), "yob": str(fields[2]), "name": str(fields[3]), "count": int(fields[4])}, index=[1])], ignore_index=True)
    return df

def get_top_gender_names(df, year_range, gender):
    full_year_range = [str(x) for x in range(year_range[0], year_range[1] + 1, 1)]
    subset = df[df['year'].isin(full_year_range) & (df['gender'] == gender)]
    return subset.groupby(['year']).apply(lambda x: x.sort_values(by='count', ascending=False).head(1))

def update_top_gender_names():
    top_male_names = get_top_gender_names(st.session_state.df_us, st.session_state.year_range, 'M')
    top_female_names = get_top_gender_names(st.session_state.df_us, st.session_state.year_range, 'F')

def get_data_from_text_files(directory_name, out_name):
    # get the social security baby names dataset for the last 10 years
    directory = directory_name
    df = pd.DataFrame()
    states = ['UT', 'VA']

    for state in states:
        filename = f"{state}.TXT"
        df = add_file_to_df(df, filename)  
    df.to_csv(out_name, index=False)

# get_data_from_text_files('./namesbystate', 'namesbystate.csv')

st.title("SS Name Data 2014-2023")

# get the data
df_us = pd.read_csv('names.csv', index_col=False)
df_us['year'] = df_us['year'].astype(str)

df_states = pd.read_csv('namesbystate.csv', index_col=False)
df_states['latitude'] = df_states.apply(lambda x: 39.3200 if x['state'] == 'UT' else 37.7749 if x['state'] == 'VA' else None, axis=1)
df_states['longitude'] = df_states.apply(lambda x: -110.9333 if x['state'] == 'UT' else -78.1649 if x['state'] == 'VA' else None, axis=1)

# initialize session state
if 'df_us' not in st.session_state:
    st.session_state['df_us'] = df_us
if 'year_range' not in st.session_state:
    st.session_state['year_range'] = (2014, 2023)
if 'top_male_names' not in st.session_state:
    st.session_state['top_male_names'] = pd.DataFrame()
if 'top_female_names' not in st.session_state:
    st.session_state['top_female_names'] = pd.DataFrame()

tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

with st.sidebar:
    st.write('Select a name and a year to get the count for that name and year.')
    name = st.selectbox("Select name", df_us['name'].unique())
    year = st.slider("Select year", 2014, 2023)
    st.write(f'{name} in {year}')
    sidebar_df = df_us[(df_us['name'] == name) & (df_us['year'] == str(year))]
    st.write(sidebar_df)


with tab1:
    col1, col2 = st.columns(2)
    top_male_names = get_top_gender_names(df_us, st.session_state.year_range, 'M')
    st.session_state['top_male_names'] = top_male_names.drop(columns=['gender'])
    top_female_names = get_top_gender_names(df_us, st.session_state.year_range, 'F')
    st.session_state['top_female_names'] = top_female_names.drop(columns=['gender'])

    with col1:
        st.header('Top Male Names')
        st.write(st.session_state['top_male_names'])
        fig1, ax1 = plt.subplots(1, 1)
        sns.barplot(x='year', y='count', data=st.session_state['top_male_names'], hue='name', ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.header('Top Female Names')
        st.write(st.session_state['top_female_names'])
        fig2, ax2 = plt.subplots(1, 1)
        sns.barplot(x='year', y='count', data=st.session_state['top_female_names'], hue='name', ax=ax2)
        st.pyplot(fig2)

    # slider below the columns
    st.slider("Select year range", 2014, 2023, key="year_range", on_change=update_top_gender_names)


with tab2:
    st.header("State Name Popularity Comparison")
    st.write('I chose Virginia and Utah because I lived in both states. A red dot means there are more babies with that name in the state where the dot appears, blue means there are fewer.')
    name = st.selectbox("Select name", df_states['name'].unique())
    state_name_df = df_states[df_states['name'] == name]
    utah_count = state_name_df[state_name_df['state'] == 'UT']['count'].values[0]
    virginia_count = state_name_df[state_name_df['state'] == 'VA']['count'].values[0]
    if utah_count > virginia_count:
        state_name_df['color'] = np.where(state_name_df['state'] == 'UT', '#FF0000', '#0000FF')
    else:
        state_name_df['color'] = np.where(state_name_df['state'] == 'VA', '#FF0000', '#0000FF')

    st.write(f'Utah: {utah_count}, Virginia: {virginia_count}')
    st.map(state_name_df, latitude='latitude', longitude='longitude', color='color')
