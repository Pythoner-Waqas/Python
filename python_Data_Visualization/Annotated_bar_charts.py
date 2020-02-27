
import matplotlib.pyplot as plt
import matplotlib as mpl
print(plt.style.available)
mpl.style.use(['ggplot']) # optional: for ggplot-like style
import pandas as pd

df_can = pd.read_excel('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Canada.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2)
print('Data downloaded and read into a dataframe!')
df_can.head(1)
print(df_can.shape)


# clean up the dataset to remove unnecessary columns (eg. REG) 
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis = 1, inplace = True)
# let's rename the columns so that they make sense
df_can.rename (columns = {'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace = True)
# for sake of consistency, let's also make all column labels of type string
df_can.columns = list(map(str, df_can.columns))
# set the country name as index - useful for quickly looking up countries using .loc method
df_can.set_index('Country', inplace = True)
# add total column
df_can['Total'] =  df_can.sum (axis = 1)
# years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))
print ('data dimensions:', df_can.shape)

df_iceland = df_can.loc['Iceland', years]

df_iceland.plot(kind='bar', figsize=(10, 6), rot=90) 

plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.title('Icelandic Immigrants to Canada from 1980 to 2013')

# Annotate arrow
plt.annotate('',                      # s: str. will leave it blank for no text
             xy=(32, 70),             # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),         # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',         # will use the coordinate system of the object being annotated 
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
            )

# Annotate Text
plt.annotate('2008 - 2011 Financial Crisis', # text to display
             xy=(28, 30),                    # start the text at at point (year 2008 , pop 30)
             rotation=72.5,                  # based on trial and error to match the arrow
             va='bottom',                    # want the text to be vertically 'bottom' aligned
             ha='left',                      # want the text to be horizontally 'left' algned.
            )

plt.show()


#Annotate horizontal bar charts
df_can.sort_values(by='Total', ascending=True, inplace=True)
df_top15 = df_can['Total'].tail(15)

df_top15.plot(kind='barh', figsize=(12, 12), color='steelblue')
plt.xlabel('Number of Immigrants')
plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')
for index, value in enumerate(df_top15): 
    label = format(int(value), ',') # format int with commas
    
    #print(value)
    # place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
    plt.annotate(label, xy=(value - 0, index - 0), color='black')
    #print(value)


#Part 2 (bar charts for multiple columns)
################################
    
import numpy as np

df_sr = pd.read_csv('https://cocl.us/datascience_survey_data',index_col = 0)

df_sr.sort_values(['Very interested'], ascending=False, axis=0, inplace=True)
# Taking the percentage of the responses and rounding it to 2 decimal places 
val = df_sr.sum(axis=1).values.reshape(-1,1)
df_sr = round((df_sr/val)*100,2)

ax = df_sr.plot(kind='bar', 
                figsize=(15, 8),
                rot=90,color = ['#5cb85c','#5bc0de','#d9534f'],
                width=.8,
                fontsize=14)
# Setting plot title
ax.set_title('Percentage of Respondents Interest in Data Science Areas',fontsize=16)
# Setting figure background color
ax.set_facecolor('white')
# setting legend font size
ax.legend(fontsize=14,facecolor = 'white') 
# Removing the Border 
ax.get_yaxis().set_visible(False)

# Creating a function to display the percentage.
for p in ax.patches:
    #print(p.get_x())
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14
               )
    
plt.show()