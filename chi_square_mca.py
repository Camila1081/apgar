#!/usr/bin/env python
# coding: utf-8

pip install pandasql

pip install mca


pip install prince==0.7.1

pip install jinja2==3.1.2
pip install matplotlib
pip install scipy

pip install --upgrade jinja2

pip install numpy

pip install seaborn

get_ipython().system(' pip install factor_analyzer==0.3.2')

pip install scipy


import pandas as pd
import jinja2
import scipy
import prince
#import mca
from scipy.stats import chi2_contingency
from pandasql import sqldf
import numpy as np
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import seaborn as sns
plt.style.use('fivethirtyeight')
from sklearn.datasets import load_digits
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer

import pandas as pd


df4=pd.read_fwf(filepath_or_buffer="./Preliminar/Nat2022us/Nat2022PublicUS.c20230504.r20230822.txt", header=None,nrows=150000,colspecs=[(49,50),(32,33),(78,79),(106,107),(119,120),(123,124),(181,182),(226,227),(268,269),(260,261),(261,262),(262,263),(263,264),(286,287),(305,306),(312,313),(313,314),(314,315),(317,318),(400,401),(401,402),(330,331),(502,503),(382,383),(387,388),(445,446)])


df4 = df4.sample(n=30000, random_state=1)


df4 = df4.rename(columns={
0:'BFACIL3',
1:'F_BFACIL',
2:'MAGER9',
3:'MRACE6',
4:'DMAR',
5:'MEDUC',
6:'TBO_REC',    
7:'PRECARE5',    
8:'CIG_REC',
9:'CIG0_R',
10:'CIG1_R',
11:'CIG2_R',
12:'CIG3_R',
13:'BMI_R',
14:'WTGAIN_REC',
15:'RF_PDIAB',
16:'RF_GDIAB' ,
17:'RF_PHYPE',
18:'RF_PPTERM',
19:'ME_PRES',
20:'ME_ROUT',
21:'RF_CESAR',
22:'OEGest_R3',
23:'LD_INDL',
24:'LD_ANES',
25:'APGAR5R'
    
})


df4.head(4)


#Substituindo valores para facilitar na apresentação do resultado
df4=df4.dropna()
df4=df4.replace('Y',1)
df4=df4.replace('N',0)
df4=df4.replace('U','99')

#Deletando categorias com menos de 10 observações totais
n=df4.size
def replace_low_frequency(icol,df4,threshold=20):
    value_counts = df4[icol].value_counts()
    to_replace = value_counts[value_counts < threshold].index
    lista=to_replace[:].tolist()
    print(lista[:])
    print(f"to_replace: {lista}")
    print(df4[df4[icol].isin(lista)])
    
    #df4=df4[df4[icol]!=lista[0]]
    df4 = df4[~df4[icol].isin(lista)]
    df4= df4[~df4[icol].isin(['U', '999', '99',99,999,''])]
    return df4

for icol in df4.columns:
    print(f"\nColuna é: {icol}")
    df4=replace_low_frequency(icol,df4, 10)

df4['RF_PDIAB'] = pd.to_numeric(df4['RF_PDIAB'],errors = 'coerce')
df4['RF_GDIAB'] = pd.to_numeric(df4['RF_GDIAB'],errors = 'coerce')
df4['RF_PHYPE'] = pd.to_numeric(df4['RF_PHYPE'],errors = 'coerce')
df4['TBO_REC'] = pd.to_numeric(df4['TBO_REC'],errors = 'coerce')
df4['CIG0_R'] = pd.to_numeric(df4['CIG0_R'],errors = 'coerce')
df4['CIG1_R'] = pd.to_numeric(df4['CIG1_R'],errors = 'coerce')
df4['CIG2_R'] = pd.to_numeric(df4['CIG2_R'],errors = 'coerce')
df4['CIG3_R'] = pd.to_numeric(df4['CIG3_R'],errors = 'coerce')
df4['OEGest_R3'] = pd.to_numeric(df4['OEGest_R3'],errors = 'coerce')
df4['RF_PPTERM'] = pd.to_numeric(df4['RF_PPTERM'],errors = 'coerce')
df4['RF_CESAR'] = pd.to_numeric(df4['RF_CESAR'],errors = 'coerce')
df4['CIG_REC'] = pd.to_numeric(df4['CIG_REC'],errors = 'coerce')
df4['LD_ANES'] = pd.to_numeric(df4['LD_ANES'],errors = 'coerce')
df4['LD_INDL'] = pd.to_numeric(df4['LD_INDL'] ,errors = 'coerce')
df4.head(5)


#Eliminando as observações sem informação de APGAR5
df4=df4[df4['APGAR5R'] != 5]
#Eliminando linhas com local de nascimento desconhecido
df4=df4[df4['BFACIL3'] != 3]
df4=df4[df4['MEDUC'] != 9]
df4=df4[df4['BMI_R'] != 9]
#Eliminando dados desconhecidos de fumantes
df4=df4[df4['CIG0_R'] != 6]
df4=df4[df4['CIG1_R'] != 6]
df4=df4[df4['CIG2_R'] != 6]
df4=df4[df4['CIG3_R'] != 6]
#Elimnando observacoes sem informacao de local de nascimento
df4=df4[df4['F_BFACIL']==1]
#Eliminando
df4=df4[df4['MAGER9']!='99']
df4=df4[df4['WTGAIN_REC']!='9']
df4=df4[df4['WTGAIN_REC']!=9]
df4=df4[df4['PRECARE5']!=5]

#Agregando resultado de APGAR
#df4['APGAR5R'] = df4['APGAR5R'].replace(1,'RUIM')
#df4['APGAR5R'] = df4['APGAR5R'].replace(2,'RUIM')
#df4['APGAR5R'] = df4['APGAR5R'].replace(3,'BOM')
#df4['APGAR5R'] = df4['APGAR5R'].replace(4,'BOM')



pd.set_option('display.max_columns', None)
df4.describe().round(2)

df4.drop(['F_BFACIL'], axis='columns', inplace=True)

#checando total de valores nulos
df4.count() 

df4=df4.dropna()

df4[df4['BFACIL3']=='3']


df4.hist(bins=10, figsize=(30,30))


# Variáveis Categóricas
f,ax=plt.subplots(1,2,figsize=(18,8))
df4[['DMAR','MEDUC']].groupby(['DMAR']).count().plot.bar(ax=ax[0])
ax[0].set_title('MEDUC quantity per DMAR')
sns.violinplot(data=df4,x='DMAR',hue='MEDUC')
ax[1].set_title('DMAR vs MEDUC')
plt.show()


f,ax=plt.subplots(1,2,figsize=(18,8))
df4[['ME_PRES','ME_ROUT']].groupby(['ME_ROUT']).count().plot.bar(ax=ax[0])
ax[0].set_title('ME_PRES quantity per ME_ROUT')
sns.violinplot(data=df4,x='ME_PRES',hue='ME_ROUT')
ax[1].set_title('ME_PRES vs ME_ROUT')
plt.show()

sns.heatmap(df4.corr(method='spearman', min_periods=10), 
           # annot=True, 
            cmap='RdYlGn', 
            linewidths=0.8, 
            xticklabels=True,  # Show all x-axis labels
            yticklabels=True)  # Show all y-axis labelsfig=plt.gcf()
# Adjust the font size for x and y tick labels
plt.xticks(fontsize=10)  # Set x-axis label font size
plt.yticks(fontsize=10)  # Set y-axis label font size
fig.set_size_inches(80,80)
plt.show()


# 
# **Teste Chi Quadrado das variáveis categóricas**
# 

# Extraindo nome das colunas 
column_names=df4.columns
print(column_names)
# Associando colunas aos index
chisqmatrix=pd.DataFrame(df4,columns=column_names,index=column_names)
perexpected_matrix=pd.DataFrame(df4,columns=column_names,index=column_names)


#df4=df4.astype('category').head(5000)
df4=df4.astype('category')
df4[column_names].head(2)


# Para realizar um teste qui-quadrado em uma amostra com muitas colunas categóricas em Python, você pode usar a função scipy.stats.chisquare() da biblioteca SciPy. 
# A tabela de contingência é representada pela matriz table, onde cada valor representa a contagem de ocorrências de uma categoria em uma determinada coluna. A função chi2_contingency() calcula o valor qui-quadrado, o valor p, os graus de liberdade e os valores esperados para uma tabela de contingência.
# Além disso, a tabela de frequências esperadas (expected) mostra o que seria esperado se as variáveis fossem independentes. Comparar as frequências observadas com as esperadas pode ajudar a entender a natureza da associação entre as variáveis.
# Nesse caso foram descartadas as relações onde a frequência esperada era menos de 10 observacoes


# Setting counters to zero
import prince
outercnt=0
innercnt=0
keep_var=['APGAR5R']
matriz_p=pd.DataFrame(columns=['variavel','p-value'])
adjusted_residuals_df2=pd.DataFrame(columns=['APGAR5R_1','APGAR5R_2','APGAR5R_3','APGAR5R_4'])
inercias=pd.DataFrame(columns=['Variavel','Inercia'])

for icol in column_names: # Outer loop
    for jcol in column_names: # inner loop 
       # print(f"\nVariáveis: {icol} vs {jcol}")
        
     # Construindo tabela de contingencia
        mycrosstab=pd.crosstab(df4[icol],df4[jcol])        
        
        
        #dof: grau de liberdade
        #p: p-value 
        #expected: frequencia esperada        
        stat,p,dof,expected=scipy.stats.chi2_contingency(mycrosstab)  
        
        p=round(p,5)
        
        #Frequencia residual
        residuals=(mycrosstab-expected)
        
        #Guardando p-valor a cada ciclo
        chisqmatrix.iloc[outercnt,innercnt]=p

        #Se o valor p for menor que o nível de significância escolhido (no caso p<=0.05), 
        #rejeita-se a hipótese nula de independência entre as variáveis.
 
        if icol==jcol:
            chisqmatrix.iloc[outercnt,innercnt]=1.00
            
        
        if jcol=='APGAR5R' and icol!='APGAR5R':
            
            print(f"\n\nVariáveis: {icol} vs {jcol}")
            print(f"Valor qui-quadrado: {stat}")
            print(f"Valor p: {p}")
            print(f"Graus de liberdade: {dof}")
            
            print(f"Valores frequencia observada:") 
            print(mycrosstab)
            print(f"Valores frequencia esperados:")
            print(expected)
            print(f"Valores frequencia residual:")          
            print(residuals)
            
            print(f"Valores frequencia esperada padronizada")
            freq=residuals/np.sqrt(expected)
            print(freq)           
            
        
            #Calculando frequencia padronizada e adjustada
            n_total = mycrosstab.sum().sum()

            row_totals = mycrosstab.sum(axis=1).values
            row_perc=(1-row_totals/n_total)

            col_totals = mycrosstab.sum(axis=0).values
            col_perc=(1-col_totals/n_total)

            total_perc=np.outer(row_perc,col_perc)
            total_perc_sqrt=np.sqrt(total_perc)
                       
            print(f"\nValores frequencia residual padronizada ajustado")
            expected_std_adjust=freq/np.sqrt(total_perc_sqrt)
            print(expected_std_adjust)
            
            
            freq_abs=abs(expected_std_adjust)
            
            teste=freq_abs.ge(1.96)            
            
            matriz_p.loc[len(matriz_p)]=[icol,p]
            
            #Checa se tem algum valor residual siginificativo >1.96
            signif=freq_abs.ge(1.96)
           
            expected_std_adjust['Combined_Index'] = icol+ '_'+freq.index.astype(str) 
            
            expected_std_adjust.set_index("Combined_Index", inplace=True)
            
            
            print("Print colunas da variavel")            
            expected_std_adjust = expected_std_adjust.rename(columns={1: 'APGAR5R_1', 2: 'APGAR5R_2', 3: 'APGAR5R_3', 4: 'APGAR5R_4'})
            adjusted_residuals_df2.columns.name='APGAR5R'
            
            adjusted_residuals_df = pd.DataFrame(expected_std_adjust, index=expected_std_adjust.index, 
                                                 columns=expected_std_adjust.columns)
            print(adjusted_residuals_df.columns)
            
            if p<0.05 and signif.any().any():
                
                #Valores frequencia residual ajustado acumulada
                adjusted_residuals_df2 = pd.concat([adjusted_residuals_df,adjusted_residuals_df2]) 
                
                keep_var.append(icol)
                
                print(f'frequency size {expected_std_adjust.size}')

                # Perform MCA
                mca = prince.MCA()
                mca_results = mca.fit(df4[[icol,jcol]])
                
                # Get the coordinates of the categories
                coordinates = mca.row_coordinates(df4[[icol,jcol]])

                # Plot the results
                plt.figure(figsize=(10, 8))
                mca_results.plot_coordinates(df4[[icol,jcol]],
                                     ax=plt.gca(),
                                     show_row_points=False,
                                     show_row_labels=False,
                                     show_column_points=True,
                                     column_points_size=100,
                                     show_column_labels=True,
                                     legend_n_cols=5,)
                plt.title(f"MCA - Category Coordinates")
                plt.show()

                # Print eigenvalues and explained inertia ratios
                print("Eigenvalues:")
                print(mca_results.eigenvalues_)
                
                print("Explained inertia ratios:")
                print(mca_results.explained_inertia_)
                
                print("Inercia Total")
                total_inertia = sum(mca.eigenvalues_)
                
                print(total_inertia)
                inputs=pd.DataFrame({'Variavel': [icol],
                   'Inercia_Total': [total_inertia]})                
                inercias=pd.concat([inercias,inputs]) 
                
                
  
        innercnt=innercnt+1
        innercnt
    outercnt=outercnt+1
    outercnt
    innercnt=0
print(inercias)



pd.set_option("display.max_rows", None, "display.max_columns", None)
display(adjusted_residuals_df2)



matriz_p



df4 = df4[keep_var]


df4.columns
#keep_var



adjusted_residuals_df3=adjusted_residuals_df2[(abs(adjusted_residuals_df2['APGAR5R_1']) >= 1.96)|(abs(adjusted_residuals_df2['APGAR5R_2']) >= 1.96)|(abs(adjusted_residuals_df2['APGAR5R_3']) >= 1.96)|(abs(adjusted_residuals_df2['APGAR5R_4']) >= 1.96)]
adjusted_residuals_df4=adjusted_residuals_df2[(abs(adjusted_residuals_df2['APGAR5R_1']) >= 3)|(abs(adjusted_residuals_df2['APGAR5R_2']) >= 3)|(abs(adjusted_residuals_df2['APGAR5R_3']) >= 3)|(abs(adjusted_residuals_df2['APGAR5R_4']) >= 3)]
adjusted_residuals_df3


def plot_pretty_heatmap(df, figsize_factor=0.5, annot_size=10, title='Mapa de calor dos valores residuais ajustados'):
    # Automatically adjust figure size based on the dataframe's shape
    fig_width = df.shape[1] * figsize_factor  # Width based on number of columns
    fig_height = df.shape[0] * figsize_factor  # Height based on number of rows
    
    # Create the figure and axis
    plt.figure(figsize=(fig_width, fig_height), facecolor='#f7f7f7')  # light background for contrast
    
    # Create a custom colormap (you can change it if needed)
    cmap = sns.diverging_palette(20, 220, as_cmap=True)
    
    # Create the heatmap with prettier aesthetics
    heatmap = sns.heatmap(df, 
                          annot=True, 
                          cmap=cmap, 
                          linewidths=0.5, 
                          annot_kws={"size": annot_size, "weight": 'bold'},  # Bold text annotations
                          fmt='.1f', 
                          cbar_kws={"shrink": 0.8, "aspect": 30, 'ticks': [df.min().min(), df.mean().mean(), df.max().max()]},
                          square=True,  # To make each cell square
                          linecolor='white',  # White lines between cells for clarity
                          vmin=df.min().min(), vmax=df.max().max(),  # Ensure consistent scaling
                          edgecolor='black',  # Darker edge for contrast
                          )
    
    # Add a bold title with a larger font size and padding
    plt.title(title, fontsize=22, fontweight='bold', pad=20, color='#333333')  # Dark gray title
    
    # Improve axis label sizes and rotations
    plt.xlabel('APGAR5R [1-4]', fontsize=16, labelpad=15, color='#333333')
    plt.ylabel('Variávale_Categoria', fontsize=16, labelpad=15, color='#333333')
    
    # Rotate x-ticks for better readability if necessary
    plt.xticks(rotation=45, ha='right', fontsize=12, color='#333333')
    plt.yticks(rotation=0, fontsize=12, color='#333333')
    
    # Get the figure and adjust its size
    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)
    
    # Add color bar with rounded corners and better positioning
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12, color='#333333')  # Adjust color bar font size and color
    cbar.outline.set_linewidth(0)  # Remove the border around the color bar for a cleaner look

    # Show the plot with a tight layout for aesthetics
    plt.tight_layout()
    plt.show()

# Example usage
plot_pretty_heatmap(adjusted_residuals_df3)



