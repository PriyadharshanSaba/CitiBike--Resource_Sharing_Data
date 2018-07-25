import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import math
import pyspark
from sklearn.model_selection import train_test_split

#beg_time = datetime.time(datetime.now())

def round_tens(x):
    return int(math.ceil(x/10.0)*10)

def dayNight(x):
    if x>=0 and x<4:
        return "midnight"
    elif x>=4 and x<12:
        return "morning"
    elif x>=12 and x<16:
        return "afternoon"
    elif x>=16 and x<21:
        return "evening"
    elif x>=21 and x<=23:
        return "night"


sample_per = 60

df = pd.read_csv('DataSet/201805-citibike-tripdata.csv')
#df = pd.read_csv("https://bitbucket.org/PriyadharshanSaba/dataset/raw/dcc580a37c7354bc4094cd475515e3574fa6beeb/CitiBikeNYC/201805-citibike-tripdata.csv")

#dropping duplicates
df  = df.dropna(axis=0)
#drop data with no gender specified
df=df[df['gender']!=0]


#calculating the age
currentYear = int(datetime.now().strftime('%Y'))
df['age'] = df['birth year'].apply(lambda x: currentYear - x )
#removing data wrt 'birth year' which is older than 90 years
df = df[df['age']<=90]


df['duration'] = (df['tripduration']/60)
df['month'] = df['starttime'].apply(lambda x: int(datetime.strptime(x,"%Y-%m-%d %H:%M:%S").strftime('%m')))
df['hour'] = df['starttime'].apply(lambda x: int(datetime.strptime(x,"%Y-%m-%d %H:%M:%S").strftime('%H')))
df['dweek'] = df['starttime'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S").strftime('%A'))
#df['day_night'] = df['hour'].apply(lambda x: dayNight(x))
df = df.drop(columns=['starttime','stoptime','start station id','start station name','start station latitude','start station longitude','end station id','end station name','end station latitude','end station longitude','name_localizedValue0','birth year'])
#removing data wrt 'birth year' which is older than 90 years
df = df[df['age']<=90]

#splitting data for sample
df, test_df = train_test_split(df, test_size=0.2)
test_df.to_csv('sample_data.csv')

no_cus = list(df.groupby(['usertype'],as_index=False).count()['tripduration'])[1]
no_sub = list(df.groupby(['usertype'],as_index=False).count()['tripduration'])[0]
total_people = no_cus+no_sub

##plothere
#objects = ('Customers', 'Subscribers')
#y_pos = np.arange(len(objects))
#perc = [no_cus,no_sub]
#plt.bar(y_pos, perc, align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
#plt.ylabel('Figures')
#plt.show()

avg_trip_length = int(round(df['duration'].sum()/total_people))

x = list((df.groupby(['usertype'],as_index=False)['duration'].sum())['duration'])

avg_trip_cus = int(round(x[0]/no_cus))
avg_trip_sub = int(round(x[1]/no_sub))

print(avg_trip_cus,"\t",avg_trip_sub)

#busy days
x = df.groupby(['dweek'],as_index=False).count() #['bikeid'].sort_values()
x['dweek'] = pd.Categorical(x['dweek'], categories=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'], ordered=True)

day_trip_count=list(x.sort_values('dweek')['bikeid'])
day_trip_count

##plothere
#objects = ('Su', 'M','Tu','W','Th','F','Sa')
#y_pos = np.arange(len(objects))
#perc = day_trip_count
#plt.bar(y_pos, perc, align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
#plt.show()


df['duration_band'] = pd.qcut(df['duration'],5)
#durband = pd.qcut(df['duration'],5).drop_duplicates().sort_values()
durband = df['duration_band'].drop_duplicates().sort_values()


#---Trips longer than mean---

#mean_duration = int(roundup(df[['duration']].mean()))
cus_df = df['duration'].where(df['usertype']=="Customer").dropna().reset_index(drop=True)
sub_df = df['duration'].where(df['usertype']=="Subscriber").dropna().reset_index(drop=True)

print(cus_df)
mean_duration_c = round_tens(cus_df.mean())
mean_duration_s = round_tens(sub_df.mean())
print(mean_duration_c,mean_duration_s)

x= cus_df.where(cus_df >= mean_duration_c).count()
y= sub_df.where(sub_df >=mean_duration_s).count()
#x=x*100/total_people
#y=y*100/total_people
#print("\n\nPeople who take longer rides (>=20)\nCustomers\t: ",x,"\nSubs\t: ",y,"\n")


##plothere
#objects = ('Custoemrs','Subscribers')
#y_pos = np.arange(len(objects))
#perc = [x,y]
#plt.bar(y_pos, perc, align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
#plt.ylabel('Percentage')
#plt.show()


##plothere
#plt.hist(sub_df['duration'],  range = (0,75), bins = 16)
#plt.title('Distribution of Trip Durations for NYC')
#plt.xlabel('Duration (m)')
#plt.show()

dawWeek_peak = df[['dweek','bikeid']].groupby(['dweek'],as_index=False).count()
dayNight_peak = df[['day_night','bikeid']].groupby(['day_night'],as_index=False).count()

day = "Wednesday"   #taken as wednessday as it is viewed to be the busiest day of the week
time = "evening"    #assumption made

day_df = df[df['dweek'] == day]
ridesForDay = day_df['bikeid'].count()
day_df = day_df[day_df['day_night'] == time]
day_df.groupby(['hour'],as_index=False).count()[['hour','bikeid']]
busyHour = (day_df['bikeid'].count())*100/ ridesForDay

#--REGRESSION--
regr = linear_model.LinearRegression()

#x = list(df.groupby(['hour']).count()['bikeid'])
x = np.array(x)
x=x.reshape(-1,1)

y = df[df['dweek']==day]
y = list(y.groupby(['hour']).count()['bikeid'])
y=np.array(y)

regr.fit(x,y)

x_test = test_df[test_df['dweek']==day]
x_test = list(x_test.groupby(['hour']).count()['bikeid'])
x_test = np.array(x_test)
x_test = x_test.reshape(-1,1)

pred = regr.predict(x_test)

y_test = test_df[test_df['dweek']==day]
y_test = list(test_df.groupby(['hour']).count()['bikeid'])
y_test=np.array(y_test)

#plothere
plt.scatter(x_test,y_test,color='black')
plt.plot(x_test, pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

print(regr.coef_)
